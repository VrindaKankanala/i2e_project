"""
hybrid_retriever.py

Combines semantic search (ChromaDB) and keyword search (BM25) using
Reciprocal Rank Fusion (RRF).

--- Why hybrid search ---

Vector search alone misses:
  - Exact acronym queries: "what is KDP" — "KDP" may not embed distinctly
  - Specific section references: "what does 6.3.2 say"
  - Rare technical terms that appear in only 1-2 chunks

BM25 alone misses:
  - Paraphrase queries: "how do we validate requirements" vs "verification process"
  - Conceptual questions: "what's the purpose of the Vee model"

Hybrid catches both.

--- Reciprocal Rank Fusion (RRF) ---

Given two ranked lists (semantic ranks, BM25 ranks), RRF combines them:

  score(chunk) = 1/(K + rank_semantic) + 1/(K + rank_bm25)

Where K=60 is a constant that dampens the effect of very high ranks.
A chunk appearing at rank 1 in both lists scores highest.
A chunk appearing in only one list still gets a partial score.

This is simple, parameter-free, and consistently outperforms weighted combinations.
"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

import chromadb

from app.knowledge_base.bm25_index import load_bm25_index, query_bm25
from app.knowledge_base.vector_store import load_vector_store, query_vector_store
from app.knowledge_base.acronym_store import load_acronym_dict, expand_query

load_dotenv()

RRF_K = 60           # RRF constant — standard value from the original paper
TOP_SEMANTIC = 10    # candidates from vector search before fusion
TOP_BM25 = 10        # candidates from BM25 before fusion
TOP_FINAL = 8        # final results returned after fusion

CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"


def _embed_query(query: str, client: OpenAI) -> list[float]:
    """Embed the user's question using the same model used for chunks."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding


def _reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge two ranked lists using RRF.

    Each result dict must have a "chunk_id" key.
    Returns merged list sorted by RRF score descending.
    Also carries forward the metadata and text from whichever list had it.
    """
    # Build a combined registry of all seen chunks
    registry: dict[str, dict] = {}

    for rank, result in enumerate(semantic_results):
        cid = result["chunk_id"]
        if cid not in registry:
            registry[cid] = {**result, "rrf_score": 0.0, "in_semantic": False, "in_bm25": False}
        registry[cid]["rrf_score"] += 1.0 / (k + rank + 1)
        registry[cid]["in_semantic"] = True
        registry[cid]["semantic_rank"] = rank + 1

    for rank, result in enumerate(bm25_results):
        cid = result["chunk_id"]
        if cid not in registry:
            registry[cid] = {**result, "rrf_score": 0.0, "in_semantic": False, "in_bm25": False}
        registry[cid]["rrf_score"] += 1.0 / (k + rank + 1)
        registry[cid]["in_bm25"] = True
        registry[cid]["bm25_rank"] = rank + 1

    # Sort by RRF score descending
    merged = sorted(registry.values(), key=lambda x: x["rrf_score"], reverse=True)
    return merged


class HybridRetriever:
    """
    Main retriever. Initialised once, then call .retrieve(query) per question.
    Loads ChromaDB and BM25 from disk on first use.
    """

    def __init__(self) -> None:
        self._collection: chromadb.Collection | None = None
        self._bm25 = None
        self._chunk_ids: list[str] | None = None
        self._chunk_lookup: dict | None = None
        self._openai_client: OpenAI | None = None
        self._acronyms: dict[str, str] = {}

    def _ensure_loaded(self) -> None:
        if self._collection is None:
            self._collection = load_vector_store(CHROMA_DIR)
        if self._bm25 is None:
            self._bm25, self._chunk_ids, self._chunk_lookup = load_bm25_index()
        if self._openai_client is None:
            self._openai_client = OpenAI()
        if not self._acronyms:
            self._acronyms = load_acronym_dict()

    def retrieve(
        self,
        query: str,
        n_results: int = TOP_FINAL,
    ) -> list[dict]:
        """
        Run hybrid retrieval for a query.

        Returns top n_results chunks as a list of dicts:
          {
            chunk_id, text, metadata,
            rrf_score,
            in_semantic, semantic_rank (if applicable),
            in_bm25, bm25_rank (if applicable)
          }
        """
        self._ensure_loaded()

        # 0. Expand acronyms in query
        expanded_query = expand_query(query, self._acronyms)

        # 1. Semantic search (on expanded query)
        query_emb = _embed_query(expanded_query, self._openai_client)
        semantic = query_vector_store(
            self._collection, query_emb, n_results=TOP_SEMANTIC
        )

        # 2. BM25 keyword search (on expanded query)
        bm25_results = query_bm25(
            self._bm25, self._chunk_ids, self._chunk_lookup,
            expanded_query, n_results=TOP_BM25
        )

        # 3. Fuse
        fused = _reciprocal_rank_fusion(semantic, bm25_results)

        # Attach expanded query so UI can display it if changed
        for r in fused:
            r["expanded_query"] = expanded_query

        return fused[:n_results]
