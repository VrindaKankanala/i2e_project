"""
scripts/ingest.py

One-shot ingestion pipeline. Run this ONCE to build the full knowledge base.

Steps:
  1. Parse PDF             → raw text blocks
  2. Extract structure     → section hierarchy tree
  3. Build chunks          → 538 section-boundary chunks
  4. Generate embeddings   → OpenAI text-embedding-3-small
  5. Build vector store    → ChromaDB (semantic search)
  6. Build BM25 index      → keyword search
  7. Build acronym dict    → 130 NASA acronyms extracted

After this script completes, the knowledge base is ready and the Streamlit
app can be started with: uv run streamlit run app/ui/streamlit_app.py

Usage:
  uv run python scripts/ingest.py
  uv run python scripts/ingest.py --reset   # rebuild from scratch
"""

import argparse
import sys
import time
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.structure_extractor import extract_structure
from app.ingestion.chunker import build_chunks, save_chunks
from app.ingestion.box_extractor import get_box_rects
from app.ingestion.image_extractor import get_diagram_rects
from app.knowledge_base.embedder import generate_embeddings
from app.knowledge_base.vector_store import build_vector_store
from app.knowledge_base.bm25_index import build_bm25_index, save_bm25_index
from app.knowledge_base.acronym_store import build_acronym_dict, save_acronym_dict

load_dotenv()

PDF_PATH    = Path(__file__).parent.parent / "nasa_systems_engineering_handbook_0.pdf"
CHUNKS_DIR  = Path(__file__).parent.parent / "data" / "chunks"
CHROMA_DIR  = Path(__file__).parent.parent / "data" / "chroma_db"


def main(reset: bool = False) -> None:
    total_start = time.time()

    print("=" * 55)
    print("  NASA Handbook Ingestion Pipeline")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Step 1 — Parse PDF
    # ------------------------------------------------------------------
    print("\n[1/6] Parsing PDF...")
    t = time.time()
    blocks = parse_pdf(PDF_PATH)
    print(f"      {len(blocks)} blocks extracted  ({time.time()-t:.1f}s)")

    # ------------------------------------------------------------------
    # Step 2 — Extract structure
    # ------------------------------------------------------------------
    print("\n[2/6] Extracting section structure...")
    t = time.time()
    sections, by_id = extract_structure(blocks)
    print(f"      {len(sections)} sections detected  ({time.time()-t:.1f}s)")

    # ------------------------------------------------------------------
    # Step 2b — Get box rects (used by chunker to exclude box content)
    # ------------------------------------------------------------------
    print("\n[2b]  Loading box + diagram rects for deduplication...")
    t = time.time()
    box_rects     = get_box_rects(PDF_PATH)
    diagram_rects = get_diagram_rects(PDF_PATH)
    print(f"      {len(box_rects)} box pages, {len(diagram_rects)} diagram pages  ({time.time()-t:.1f}s)")

    # ------------------------------------------------------------------
    # Step 3 — Build chunks (box + diagram label content excluded)
    # ------------------------------------------------------------------
    print("\n[3/6] Building chunks (box + diagram labels excluded)...")
    t = time.time()
    chunks = build_chunks(blocks, sections, by_id,
                          box_rects=box_rects, diagram_rects=diagram_rects)
    save_chunks(chunks, CHUNKS_DIR)
    print(f"      {len(chunks)} chunks built  ({time.time()-t:.1f}s)")

    # ------------------------------------------------------------------
    # Step 4 — Generate embeddings
    # ------------------------------------------------------------------
    print("\n[4/6] Generating embeddings (OpenAI)...")
    t = time.time()
    client = OpenAI()
    chunk_embeddings = generate_embeddings(chunks, client)
    print(f"      {len(chunk_embeddings)} embeddings generated  ({time.time()-t:.1f}s)")

    # ------------------------------------------------------------------
    # Step 5 — Build vector store
    # ------------------------------------------------------------------
    print("\n[5/6] Building ChromaDB vector store...")
    t = time.time()
    collection = build_vector_store(chunk_embeddings, CHROMA_DIR, reset=reset)
    print(f"      {collection.count()} documents stored  ({time.time()-t:.1f}s)")

    # ------------------------------------------------------------------
    # Step 6 — Build BM25 index
    # ------------------------------------------------------------------
    print("\n[6/6] Building BM25 keyword index...")
    t = time.time()
    bm25, chunk_ids = build_bm25_index(chunks)
    save_bm25_index(bm25, chunk_ids, chunks)
    print(f"      Index saved  ({time.time()-t:.1f}s)")

    # ------------------------------------------------------------------
    # Step 7 — Build acronym dictionary
    # ------------------------------------------------------------------
    print("\n[7/7] Building acronym dictionary...")
    t = time.time()
    acronyms = build_acronym_dict(chunks)
    save_acronym_dict(acronyms)
    print(f"      {len(acronyms)} acronyms extracted  ({time.time()-t:.1f}s)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total = time.time() - total_start
    print(f"\n{'='*55}")
    print(f"  Ingestion complete in {total:.1f}s")
    print(f"  Chunks   : {len(chunks)}")
    print(f"  Vectors  : {collection.count()}")
    print(f"  BM25     : {len(chunk_ids)} documents")
    print(f"  Acronyms : {len(acronyms)}")
    print(f"{'='*55}")
    print("\nReady! Start the app with:")
    print("  uv run streamlit run app/ui/streamlit_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and rebuild ChromaDB collection from scratch",
    )
    args = parser.parse_args()
    main(reset=args.reset)
