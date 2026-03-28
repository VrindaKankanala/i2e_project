"""
embedder.py

Generates embeddings for all chunks using OpenAI text-embedding-3-small.

Why text-embedding-3-small:
  - 1536 dimensions, strong on technical/domain text
  - $0.02 per 1M tokens — entire handbook costs ~$0.004
  - Supports batching up to 2048 inputs per API call

Batching strategy:
  - Send chunks in batches of BATCH_SIZE (100) to avoid rate limits
  - Small sleep between batches to stay under TPM (tokens-per-minute) limits
  - Free tier: 1M TPM — our 538 chunks at ~383 tokens each = ~206K tokens total,
    well within a single minute. No throttling needed in practice.

Output:
  - Returns list of (chunk, embedding_vector) pairs
  - Embedding vector is a list of 1536 floats
"""

import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from app.ingestion.chunker import Chunk

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100          # chunks per API call
SLEEP_BETWEEN_BATCHES = 0.5  # seconds — polite rate limiting


def _chunk_to_embed_text(chunk: Chunk) -> str:
    """
    What we actually send to the embedding model.

    We prepend the section path so the embedding captures the hierarchical
    context, not just the raw paragraph text.

    Example:
      "Section: 4 System Design > 4.1 Stakeholder Expectations > 4.1.1 Process Description
       4.1.1 Process Description

       FIGURE 4.1-1 provides a typical flow diagram..."
    """
    header = f"Section: {chunk.section_path}\n{chunk.section_number} {chunk.section_title}\n\n"
    return header + chunk.text


def generate_embeddings(
    chunks: list[Chunk],
    client: OpenAI | None = None,
) -> list[tuple[Chunk, list[float]]]:
    """
    Generate embeddings for all chunks.

    Returns list of (chunk, embedding) pairs in the same order as input.
    """
    if client is None:
        client = OpenAI()  # reads OPENAI_API_KEY from environment

    results: list[tuple[Chunk, list[float]]] = []

    # Split into batches
    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    print(f"Embedding {len(chunks)} chunks in {len(batches)} batches "
          f"(model: {EMBEDDING_MODEL})")

    for batch_idx, batch in enumerate(tqdm(batches, desc="Embedding")):
        texts = [_chunk_to_embed_text(c) for c in batch]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        for chunk, embedding_obj in zip(batch, response.data):
            results.append((chunk, embedding_obj.embedding))

        if batch_idx < len(batches) - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"Done. Generated {len(results)} embeddings "
          f"(dimension: {len(results[0][1])})")
    return results
