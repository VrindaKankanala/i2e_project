"""
vector_store.py

Stores chunks + embeddings in ChromaDB for semantic (vector) search.

Why ChromaDB:
  - Runs fully locally — no server, no Docker, data saved to disk
  - Supports metadata filtering (filter by section level, page range, etc.)
  - Persistent: once built, survives restarts

Collection structure:
  - One collection: "nasa_handbook"
  - Each document = one chunk
  - Each document has:
      id        : chunk_id (e.g. "4.1.1_0")
      embedding : 1536-dim vector from OpenAI
      document  : the chunk text (what gets returned to the LLM)
      metadata  : all chunk fields except text — used for filtering + citations

Metadata stored per chunk:
  section_id, section_number, section_title, section_path,
  level, page_start, page_end, parent_id, parent_title,
  chunk_index, total_chunks, cross_refs (as comma-separated string),
  token_count
"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from app.ingestion.chunker import Chunk

COLLECTION_NAME = "nasa_handbook"


def _get_client(persist_dir: Path) -> chromadb.PersistentClient:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _chunk_to_metadata(chunk: Chunk) -> dict[str, Any]:
    """Convert a Chunk to a flat metadata dict for ChromaDB."""
    return {
        "section_id":     chunk.section_id,
        "section_number": chunk.section_number,
        "section_title":  chunk.section_title,
        "section_path":   chunk.section_path,
        "level":          chunk.level,
        "page_start":     chunk.page_start,
        "page_end":       chunk.page_end,
        "parent_id":      chunk.parent_id,
        "parent_title":   chunk.parent_title,
        "chunk_index":    chunk.chunk_index,
        "total_chunks":   chunk.total_chunks,
        "cross_refs":     ",".join(chunk.cross_refs),  # ChromaDB needs scalar values
        "token_count":    chunk.token_count,
    }


def build_vector_store(
    chunk_embeddings: list[tuple[Chunk, list[float]]],
    persist_dir: Path,
    reset: bool = False,
) -> chromadb.Collection:
    """
    Insert all chunk embeddings into ChromaDB.

    Args:
        chunk_embeddings : list of (Chunk, embedding_vector) from embedder.py
        persist_dir      : where to save the ChromaDB files
        reset            : if True, drop and recreate the collection

    Returns the ChromaDB collection.
    """
    client = _get_client(persist_dir)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for embeddings
    )

    # Check if already populated
    existing = collection.count()
    if existing > 0 and not reset:
        print(f"Collection already has {existing} documents. Skipping insert.")
        print("Pass reset=True to rebuild from scratch.")
        return collection

    # Prepare batch insert
    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    for chunk, embedding in chunk_embeddings:
        ids.append(chunk.chunk_id)
        embeddings.append(embedding)
        documents.append(chunk.text)
        metadatas.append(_chunk_to_metadata(chunk))

    # ChromaDB recommends batches of up to 5000
    BATCH = 500
    total = len(ids)
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Inserted {end}/{total} chunks...")

    print(f"Vector store built: {collection.count()} documents in '{COLLECTION_NAME}'")
    return collection


def load_vector_store(persist_dir: Path) -> chromadb.Collection:
    """Load an existing ChromaDB collection from disk."""
    client = _get_client(persist_dir)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Loaded vector store: {collection.count()} documents")
    return collection


def query_vector_store(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Semantic search: find the n_results most similar chunks.

    Args:
        query_embedding : embedding of the user's question
        n_results       : how many chunks to return
        where           : optional ChromaDB metadata filter
                          e.g. {"level": {"$lte": 3}} to restrict to top 3 levels

    Returns list of dicts, each with:
        chunk_id, text, metadata, distance (0=identical, 2=opposite)
    """
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # Flatten ChromaDB's nested list structure into a clean list of dicts
    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "chunk_id": results["ids"][0][i],
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "score":    round(1 - results["distances"][0][i], 4),  # 1=best, 0=worst
        })
    return output
