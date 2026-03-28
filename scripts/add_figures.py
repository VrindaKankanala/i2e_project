"""
scripts/add_figures.py

Incrementally adds figure/diagram chunks to the existing knowledge base.
Runs without re-embedding the 538 text chunks or 23 table chunks.

Steps:
  1. Find figure captions in PDF    -> ~52 unique figure pages
  2. Render each page + Vision API  -> gpt-4o-mini describes each diagram
  3. Embed descriptions             -> OpenAI text-embedding-3-small
  4. Add to ChromaDB                -> alongside existing text + table chunks
  5. Save to disk                   -> data/chunks/figure_chunks.json
  6. Rebuild BM25 index             -> now covers text + table + figure chunks

Cost estimate:
  ~52 Vision API calls x ~500 tokens each = ~26K tokens
  gpt-4o-mini input: ~$0.15/1M tokens -> ~$0.004
  Plus image tokens: ~1000 tokens/image -> ~$0.008
  Total: ~$0.012 (less than 2 cents)

Usage:
  uv run python scripts/add_figures.py
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.structure_extractor import extract_structure
from app.ingestion.image_extractor import extract_figures, FigureChunk
from app.knowledge_base.vector_store import load_vector_store, _get_client, COLLECTION_NAME
from app.knowledge_base.bm25_index import save_bm25_index, _tokenize
from rank_bm25 import BM25Okapi

load_dotenv()

PDF_PATH    = Path(__file__).parent.parent / "nasa_systems_engineering_handbook_0.pdf"
CHUNKS_DIR  = Path(__file__).parent.parent / "data" / "chunks"
CHROMA_DIR  = Path(__file__).parent.parent / "data" / "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"


def embed_figure_chunks(figure_chunks: list[FigureChunk], client: OpenAI) -> list[tuple]:
    """Embed all figure chunks in one batch."""
    texts = []
    for fc in figure_chunks:
        header = f"Section: {fc.section_path}\n{fc.section_id} {fc.section_title}\n\n"
        texts.append(header + fc.text)

    print(f"Embedding {len(texts)} figure chunks...")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [(fc, obj.embedding) for fc, obj in zip(figure_chunks, response.data)]


def add_to_chromadb(figure_embeddings: list[tuple], chroma_dir: Path) -> None:
    """Add figure chunks to the existing ChromaDB collection."""
    client = _get_client(chroma_dir)
    collection = client.get_collection(COLLECTION_NAME)

    ids, embeddings, documents, metadatas = [], [], [], []
    for fc, emb in figure_embeddings:
        ids.append(fc.chunk_id)
        embeddings.append(emb)
        documents.append(fc.text)
        metadatas.append({
            "section_id":     fc.section_id,
            "section_number": fc.section_id,
            "section_title":  fc.section_title,
            "section_path":   fc.section_path,
            "level":          fc.level,
            "page_start":     fc.page,
            "page_end":       fc.page,
            "parent_id":      fc.parent_id,
            "parent_title":   fc.parent_title,
            "chunk_index":    0,
            "total_chunks":   1,
            "cross_refs":     "",
            "token_count":    fc.token_count,
            "chunk_type":     "figure",
        })

    # Remove existing figure chunks first (in case of re-run)
    existing_ids = collection.get(where={"chunk_type": "figure"})["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"Removed {len(existing_ids)} existing figure chunks")

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"Added {len(ids)} figure chunks to ChromaDB (total: {collection.count()})")


def rebuild_bm25_with_figures(figure_chunks: list[FigureChunk]) -> None:
    """
    Rebuild BM25 combining text + table + figure chunks.
    Loads existing chunks from disk to avoid re-running full pipeline.
    """
    from types import SimpleNamespace
    import pickle
    from app.knowledge_base.bm25_index import BM25_DIR, INDEX_FILE

    corpus, chunk_ids, chunk_lookup = [], [], {}

    def _dict_to_ns(d: dict) -> SimpleNamespace:
        ns = SimpleNamespace(**d)
        if not hasattr(ns, "cross_refs") or ns.cross_refs is None:
            ns.cross_refs = []
        elif isinstance(ns.cross_refs, str):
            ns.cross_refs = [r for r in ns.cross_refs.split(",") if r]
        return ns

    # Load text chunks
    text_file = CHUNKS_DIR / "all_chunks.json"
    with open(text_file, encoding="utf-8") as f:
        text_dicts = json.load(f)
    for c in text_dicts:
        full_text = f"{c.get('section_path','')} {c.get('section_title','')} {c.get('text','')}"
        corpus.append(_tokenize(full_text))
        chunk_ids.append(c["chunk_id"])
        chunk_lookup[c["chunk_id"]] = _dict_to_ns(c)

    # Load table chunks
    table_file = CHUNKS_DIR / "table_chunks.json"
    if table_file.exists():
        with open(table_file, encoding="utf-8") as f:
            table_dicts = json.load(f)
        for c in table_dicts:
            full_text = f"{c.get('section_path','')} {c.get('section_title','')} {c.get('text','')}"
            corpus.append(_tokenize(full_text))
            chunk_ids.append(c["chunk_id"])
            chunk_lookup[c["chunk_id"]] = _dict_to_ns(c)
        print(f"  Loaded {len(table_dicts)} table chunks from disk")

    # Load box chunks
    box_file = CHUNKS_DIR / "box_chunks.json"
    if box_file.exists():
        with open(box_file, encoding="utf-8") as f:
            box_dicts = json.load(f)
        for c in box_dicts:
            full_text = f"{c.get('section_path','')} {c.get('section_title','')} {c.get('text','')}"
            corpus.append(_tokenize(full_text))
            chunk_ids.append(c["chunk_id"])
            chunk_lookup[c["chunk_id"]] = _dict_to_ns(c)
        print(f"  Loaded {len(box_dicts)} box chunks from disk")

    # Add figure chunks
    for fc in figure_chunks:
        full_text = f"{getattr(fc,'section_path','')} {getattr(fc,'section_title','')} {getattr(fc,'text','')}"
        corpus.append(_tokenize(full_text))
        chunk_ids.append(fc.chunk_id)
        chunk_lookup[fc.chunk_id] = fc

    print(f"Rebuilding BM25 with {len(chunk_ids)} total documents...")
    bm25 = BM25Okapi(corpus)

    BM25_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids, "chunk_lookup": chunk_lookup}, f)
    print(f"BM25 rebuilt: {len(chunk_ids)} total documents")


def main():
    total_start = time.time()
    print("=" * 55)
    print("  Adding Figure Chunks to Knowledge Base")
    print("=" * 55)

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CHUNKS_DIR / "figure_chunks.json"

    client = OpenAI()

    # If figure_chunks.json already exists, load from disk (skip Vision API)
    if out_file.exists():
        print("\n[1/5] figure_chunks.json found — loading from disk (skipping Vision API)...")
        from types import SimpleNamespace
        with open(out_file, encoding="utf-8") as f:
            dicts = json.load(f)
        figure_chunks = [SimpleNamespace(**d) for d in dicts]
        print(f"      {len(figure_chunks)} figure chunks loaded")
    else:
        print("\n[1/5] Extracting section structure...")
        blocks = parse_pdf(PDF_PATH)
        _, by_id = extract_structure(blocks)

        print("\n[2/5] Extracting figure descriptions via Vision API...")
        figure_chunks = extract_figures(PDF_PATH, by_id, client, rate_limit_delay=0.3)
        print(f"      {len(figure_chunks)} figure chunks created")

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([asdict(fc) for fc in figure_chunks], f, indent=2, ensure_ascii=False)
        print(f"      Saved to {out_file}")

    print("\n[3/5] Embedding figure chunks...")
    figure_embeddings = embed_figure_chunks(figure_chunks, client)

    print("\n[4/5] Adding to ChromaDB...")
    add_to_chromadb(figure_embeddings, CHROMA_DIR)

    print("\n[5/5] Rebuilding BM25 index...")
    rebuild_bm25_with_figures(figure_chunks)

    print(f"\n{'='*55}")
    print(f"  Done in {time.time()-total_start:.1f}s")
    print(f"  Figure chunks added: {len(figure_chunks)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
