"""
bm25_index.py

Builds and queries a BM25 keyword index over all chunks.

Why BM25 alongside vector search:
  - Vector search is great for semantic similarity ("what does X mean")
  - BM25 is great for exact keyword matches ("TRL-4", "PDR", "6.3.2.1")
  - Acronyms like "KDP", "SRR", "CDR" have very specific embeddings that may
    not cluster well — BM25 catches them exactly
  - Hybrid search (vector + BM25) consistently outperforms either alone

How BM25 works (briefly):
  - Each chunk is tokenized into words
  - BM25 scores a query by: how often query terms appear in a chunk (TF)
    adjusted by how rare those terms are across all chunks (IDF)
  - Result: chunks that contain rare, specific terms rank higher

Persistence:
  - Index saved to data/bm25/bm25_index.pkl
  - Chunk IDs saved alongside so results can be matched back to chunks
"""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.ingestion.chunker import Chunk

BM25_DIR = Path(__file__).parent.parent.parent / "data" / "bm25"
INDEX_FILE = BM25_DIR / "bm25_index.pkl"


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercases everything, splits on non-alphanumeric chars.
    Keeps hyphenated terms together (e.g. "trade-off", "TRL-4").
    """
    text = text.lower()
    # Split on whitespace and common punctuation, but keep hyphens
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    return [t for t in tokens if len(t) > 1]  # drop single-char tokens


def build_bm25_index(chunks: list[Chunk]) -> tuple[BM25Okapi, list[str]]:
    """
    Build a BM25 index from all chunks.

    Returns:
        bm25       : the BM25Okapi index
        chunk_ids  : list of chunk_id in the same order as the index
                     (needed to map BM25 result indices back to chunks)
    """
    print(f"Building BM25 index over {len(chunks)} chunks...")

    corpus = []
    chunk_ids = []

    for chunk in chunks:
        # Combine section path + text for richer keyword matching
        full_text = f"{chunk.section_path} {chunk.section_title} {chunk.text}"
        tokens = _tokenize(full_text)
        corpus.append(tokens)
        chunk_ids.append(chunk.chunk_id)

    bm25 = BM25Okapi(corpus)

    print(f"BM25 index built with {len(corpus)} documents")
    return bm25, chunk_ids


def save_bm25_index(
    bm25: BM25Okapi,
    chunk_ids: list[str],
    chunks: list[Chunk],
) -> None:
    """Save the BM25 index and supporting data to disk."""
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        # Save a minimal chunk lookup by id for retrieval
        "chunk_lookup": {c.chunk_id: c for c in chunks},
    }
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(payload, f)

    print(f"BM25 index saved to {INDEX_FILE}")


def load_bm25_index() -> tuple[BM25Okapi, list[str], dict[str, Chunk]]:
    """Load the BM25 index from disk."""
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {INDEX_FILE}. Run scripts/ingest.py first."
        )
    with open(INDEX_FILE, "rb") as f:
        payload = pickle.load(f)

    print(f"BM25 index loaded: {len(payload['chunk_ids'])} documents")
    return payload["bm25"], payload["chunk_ids"], payload["chunk_lookup"]


def query_bm25(
    bm25: BM25Okapi,
    chunk_ids: list[str],
    chunk_lookup: dict[str, Chunk],
    query: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Keyword search: return top n_results chunks matching the query.

    Returns list of dicts with:
        chunk_id, text, metadata (as dict), score (higher = better match)
    """
    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)

    # Get top n indices sorted by score descending
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    results = []
    for idx in top_indices:
        if scores[idx] == 0:
            break  # no match at all
        cid = chunk_ids[idx]
        chunk = chunk_lookup.get(cid)
        if chunk is None:
            continue
        results.append({
            "chunk_id": cid,
            "text":     chunk.text,
            "metadata": {
                "section_id":    chunk.section_id,
                "section_title": chunk.section_title,
                "section_path":  chunk.section_path,
                "page_start":    chunk.page_start,
                "page_end":      chunk.page_end,
                "level":         chunk.level,
                "cross_refs":    ",".join(chunk.cross_refs),
            },
            "score": round(float(scores[idx]), 4),
        })
    return results
