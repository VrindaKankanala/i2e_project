"""
scripts/add_tables.py

Incrementally adds table chunks to the existing knowledge base.
Runs without re-embedding the 538 text chunks (saves API cost + time).

Steps:
  1. Extract tables from PDF  → 23 table chunks
  2. Embed them               → OpenAI (cheap: ~23 chunks)
  3. Add to ChromaDB          → alongside existing 538 text chunks
  4. Save to disk             → data/chunks/table_chunks.json
  5. Rebuild BM25 index       → now covers text + table chunks

Usage:
  uv run python scripts/add_tables.py
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
from app.ingestion.table_extractor import extract_tables, TableChunk
from app.knowledge_base.vector_store import load_vector_store, _get_client, COLLECTION_NAME
from app.knowledge_base.bm25_index import build_bm25_index, save_bm25_index, _tokenize
from rank_bm25 import BM25Okapi

load_dotenv()

PDF_PATH   = Path(__file__).parent.parent / "nasa_systems_engineering_handbook_0.pdf"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"


def embed_table_chunks(table_chunks: list[TableChunk], client: OpenAI) -> list[tuple]:
    """Embed all table chunks in one batch."""
    texts = []
    for tc in table_chunks:
        header = f"Section: {tc.section_path}\n{tc.section_id} {tc.section_title}\n\n"
        texts.append(header + tc.text)

    print(f"Embedding {len(texts)} table chunks...")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [(tc, obj.embedding) for tc, obj in zip(table_chunks, response.data)]


def add_to_chromadb(table_embeddings: list[tuple], chroma_dir: Path) -> None:
    """Add table chunks to the existing ChromaDB collection."""
    client = _get_client(chroma_dir)
    collection = client.get_collection(COLLECTION_NAME)

    ids, embeddings, documents, metadatas = [], [], [], []
    for tc, emb in table_embeddings:
        ids.append(tc.chunk_id)
        embeddings.append(emb)
        documents.append(tc.text)
        metadatas.append({
            "section_id":     tc.section_id,
            "section_number": tc.section_id,
            "section_title":  tc.section_title,
            "section_path":   tc.section_path,
            "level":          tc.level,
            "page_start":     tc.page,
            "page_end":       tc.page,
            "parent_id":      tc.parent_id,
            "parent_title":   tc.parent_title,
            "chunk_index":    0,
            "total_chunks":   1,
            "cross_refs":     "",
            "token_count":    tc.token_count,
            "chunk_type":     "table",
        })

    # Remove existing table chunks first (in case of re-run)
    existing_ids = collection.get(where={"chunk_type": "table"})["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"Removed {len(existing_ids)} existing table chunks")

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"Added {len(ids)} table chunks to ChromaDB (total: {collection.count()})")


def rebuild_bm25_with_tables(table_chunks: list[TableChunk]) -> None:
    """
    Rebuild BM25 index combining existing text chunks + new table chunks.
    Loads text chunks from disk to avoid re-running the full pipeline.
    """
    # Load existing text chunks from disk
    text_chunks_file = CHUNKS_DIR / "all_chunks.json"
    with open(text_chunks_file, encoding="utf-8") as f:
        text_chunk_dicts = json.load(f)

    print(f"Rebuilding BM25 with {len(text_chunk_dicts)} text + {len(table_chunks)} table chunks...")

    corpus, chunk_ids, chunk_lookup = [], [], {}

    # Text chunks — wrap dicts as simple objects so attribute access works
    from types import SimpleNamespace

    def _dict_to_ns(d: dict) -> SimpleNamespace:
        ns = SimpleNamespace(**d)
        if not hasattr(ns, "cross_refs") or ns.cross_refs is None:
            ns.cross_refs = []
        elif isinstance(ns.cross_refs, str):
            ns.cross_refs = [r for r in ns.cross_refs.split(",") if r]
        return ns

    for c in text_chunk_dicts:
        full_text = f"{c.get('section_path','')} {c.get('section_title','')} {c.get('text','')}"
        corpus.append(_tokenize(full_text))
        chunk_ids.append(c["chunk_id"])
        chunk_lookup[c["chunk_id"]] = _dict_to_ns(c)

    # Table chunks
    for tc in table_chunks:
        full_text = f"{tc.section_path} {tc.section_title} {tc.text}"
        corpus.append(_tokenize(full_text))
        chunk_ids.append(tc.chunk_id)
        chunk_lookup[tc.chunk_id] = tc

    bm25 = BM25Okapi(corpus)

    import pickle
    from app.knowledge_base.bm25_index import BM25_DIR, INDEX_FILE
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids, "chunk_lookup": chunk_lookup}, f)
    print(f"BM25 rebuilt: {len(chunk_ids)} total documents")


def main():
    total_start = time.time()
    print("=" * 50)
    print("  Adding Table Chunks to Knowledge Base")
    print("=" * 50)

    print("\n[1/4] Extracting section structure...")
    blocks = parse_pdf(PDF_PATH)
    _, by_id = extract_structure(blocks)

    print("\n[2/4] Extracting tables from PDF...")
    table_chunks = extract_tables(PDF_PATH, by_id)
    print(f"      {len(table_chunks)} table chunks extracted")

    # Save to disk
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_DIR / "table_chunks.json", "w", encoding="utf-8") as f:
        json.dump([asdict(tc) for tc in table_chunks], f, indent=2, ensure_ascii=False)

    print("\n[3/4] Embedding + adding to ChromaDB...")
    client = OpenAI()
    table_embeddings = embed_table_chunks(table_chunks, client)
    add_to_chromadb(table_embeddings, CHROMA_DIR)

    print("\n[4/4] Rebuilding BM25 index...")
    rebuild_bm25_with_tables(table_chunks)

    print(f"\n{'='*50}")
    print(f"  Done in {time.time()-total_start:.1f}s")
    print(f"  Table chunks added: {len(table_chunks)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
