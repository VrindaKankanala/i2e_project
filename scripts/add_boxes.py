"""
scripts/add_boxes.py

Incrementally adds callout box chunks to the existing knowledge base.
Uses the Table of Boxes index (page 7) to find all 31 boxes by page number,
then extracts text from bordered rectangles using PyMuPDF drawing detection.

Steps:
  1. Parse Table of Boxes index  -> 31 box entries with page numbers
  2. Extract box text via rects  -> 31 BoxChunk objects
  3. Embed them                  -> OpenAI text-embedding-3-small
  4. Add to ChromaDB             -> alongside existing chunks
  5. Save to disk                -> data/chunks/box_chunks.json
  6. Rebuild BM25 index          -> now covers text + table + figure + box

Usage:
  uv run python scripts/add_boxes.py
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
from app.ingestion.box_extractor import extract_boxes, BoxChunk
from app.knowledge_base.vector_store import _get_client, COLLECTION_NAME
from app.knowledge_base.bm25_index import _tokenize
from rank_bm25 import BM25Okapi

load_dotenv()

PDF_PATH    = Path(__file__).parent.parent / "nasa_systems_engineering_handbook_0.pdf"
CHUNKS_DIR  = Path(__file__).parent.parent / "data" / "chunks"
CHROMA_DIR  = Path(__file__).parent.parent / "data" / "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"


def embed_box_chunks(box_chunks: list[BoxChunk], client: OpenAI) -> list[tuple]:
    """Embed all box chunks in one batch."""
    texts = []
    for bc in box_chunks:
        header = f"Section: {bc.section_path}\n{bc.section_id} {bc.section_title}\n\n"
        texts.append(header + bc.text)

    print(f"Embedding {len(texts)} box chunks...")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [(bc, obj.embedding) for bc, obj in zip(box_chunks, response.data)]


def add_to_chromadb(box_embeddings: list[tuple], chroma_dir: Path) -> None:
    """Add box chunks to the existing ChromaDB collection."""
    client = _get_client(chroma_dir)
    collection = client.get_collection(COLLECTION_NAME)

    ids, embeddings, documents, metadatas = [], [], [], []
    for bc, emb in box_embeddings:
        ids.append(bc.chunk_id)
        embeddings.append(emb)
        documents.append(bc.text)
        metadatas.append({
            "section_id":     bc.section_id,
            "section_number": bc.section_id,
            "section_title":  bc.section_title,
            "section_path":   bc.section_path,
            "level":          bc.level,
            "page_start":     bc.page,
            "page_end":       bc.page,
            "parent_id":      bc.parent_id,
            "parent_title":   bc.parent_title,
            "chunk_index":    0,
            "total_chunks":   1,
            "cross_refs":     "",
            "token_count":    bc.token_count,
            "chunk_type":     "box",
        })

    # Remove existing box chunks first (in case of re-run)
    existing_ids = collection.get(where={"chunk_type": "box"})["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"Removed {len(existing_ids)} existing box chunks")

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"Added {len(ids)} box chunks to ChromaDB (total: {collection.count()})")


def rebuild_bm25_with_boxes(box_chunks: list[BoxChunk]) -> None:
    """Rebuild BM25 combining text + table + figure + box chunks."""
    import pickle
    from types import SimpleNamespace
    from app.knowledge_base.bm25_index import BM25_DIR, INDEX_FILE

    corpus, chunk_ids, chunk_lookup = [], [], {}

    def _dict_to_ns(d: dict) -> SimpleNamespace:
        ns = SimpleNamespace(**d)
        if not hasattr(ns, "cross_refs") or ns.cross_refs is None:
            ns.cross_refs = []
        elif isinstance(ns.cross_refs, str):
            ns.cross_refs = [r for r in ns.cross_refs.split(",") if r]
        return ns

    # Text chunks
    with open(CHUNKS_DIR / "all_chunks.json", encoding="utf-8") as f:
        for c in json.load(f):
            corpus.append(_tokenize(f"{c.get('section_path','')} {c.get('section_title','')} {c.get('text','')}"))
            chunk_ids.append(c["chunk_id"])
            chunk_lookup[c["chunk_id"]] = _dict_to_ns(c)

    # Table chunks
    table_file = CHUNKS_DIR / "table_chunks.json"
    if table_file.exists():
        with open(table_file, encoding="utf-8") as f:
            table_dicts = json.load(f)
        for c in table_dicts:
            corpus.append(_tokenize(f"{c.get('section_path','')} {c.get('section_title','')} {c.get('text','')}"))
            chunk_ids.append(c["chunk_id"])
            chunk_lookup[c["chunk_id"]] = _dict_to_ns(c)
        print(f"  Loaded {len(table_dicts)} table chunks")

    # Figure chunks
    figure_file = CHUNKS_DIR / "figure_chunks.json"
    if figure_file.exists():
        with open(figure_file, encoding="utf-8") as f:
            figure_dicts = json.load(f)
        for c in figure_dicts:
            corpus.append(_tokenize(f"{c.get('section_path','')} {c.get('section_title','')} {c.get('text','')}"))
            chunk_ids.append(c["chunk_id"])
            chunk_lookup[c["chunk_id"]] = _dict_to_ns(c)
        print(f"  Loaded {len(figure_dicts)} figure chunks")

    # Box chunks (new)
    for bc in box_chunks:
        corpus.append(_tokenize(f"{bc.section_path} {bc.section_title} {bc.text}"))
        chunk_ids.append(bc.chunk_id)
        chunk_lookup[bc.chunk_id] = bc

    print(f"Rebuilding BM25 with {len(chunk_ids)} total documents...")
    bm25 = BM25Okapi(corpus)

    BM25_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids, "chunk_lookup": chunk_lookup}, f)
    print(f"BM25 rebuilt: {len(chunk_ids)} total documents")


def main():
    total_start = time.time()
    print("=" * 55)
    print("  Adding Box Chunks to Knowledge Base")
    print("=" * 55)

    print("\n[1/5] Extracting section structure...")
    blocks = parse_pdf(PDF_PATH)
    _, by_id = extract_structure(blocks)

    print("\n[2/5] Extracting callout boxes from PDF...")
    box_chunks = extract_boxes(PDF_PATH, by_id)
    print(f"      {len(box_chunks)} box chunks extracted")

    # Save to disk
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CHUNKS_DIR / "box_chunks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([asdict(bc) for bc in box_chunks], f, indent=2, ensure_ascii=False)
    print(f"      Saved to {out_file}")

    print("\n[3/5] Embedding box chunks...")
    client = OpenAI()
    box_embeddings = embed_box_chunks(box_chunks, client)

    print("\n[4/5] Adding to ChromaDB...")
    add_to_chromadb(box_embeddings, CHROMA_DIR)

    print("\n[5/5] Rebuilding BM25 index...")
    rebuild_bm25_with_boxes(box_chunks)

    print(f"\n{'='*55}")
    print(f"  Done in {time.time()-total_start:.1f}s")
    print(f"  Box chunks added: {len(box_chunks)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
