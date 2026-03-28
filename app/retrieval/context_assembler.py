"""
context_assembler.py

Takes the retrieved chunks and builds a clean context block to send to the LLM.

Two responsibilities:
  1. Cross-reference following: if a retrieved chunk mentions "see Section 4.3",
     fetch that section's chunk from ChromaDB and add it to context.
  2. Context formatting: structure the chunks into a readable prompt section
     with clear labels so the LLM knows where each piece of info came from.

--- Why cross-ref following matters ---

Example: User asks "What are the inputs to the Stakeholder Expectations process?"
  - Chunk 4.1.1 retrieved, contains: "...see Section 4.1.1.1 for typical inputs..."
  - Without cross-ref following: LLM only sees partial info
  - With cross-ref following: chunk 4.1.1.1 is also fetched and added to context
  - LLM can now give a complete answer with the actual input list

--- Context format sent to LLM ---

[SOURCE 1] Section 4.1.1 | Process Description | Pages 55-63
Path: 4 System Design Processes > 4.1 Stakeholder Expectations > 4.1.1 Process Description
---
<chunk text>

[SOURCE 2] Section 4.1.1.1 | Inputs | Pages 55-58
Path: ... > 4.1.1 Process Description > 4.1.1.1 Inputs
(cross-reference from Section 4.1.1)
---
<chunk text>
"""

from pathlib import Path

import chromadb
from openai import OpenAI

from app.knowledge_base.vector_store import load_vector_store, query_vector_store
from app.knowledge_base.embedder import EMBEDDING_MODEL

CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "chroma_db"
MAX_CROSS_REF_CHUNKS = 3    # max extra chunks fetched via cross-references
MAX_CONTEXT_CHUNKS = 8      # absolute cap on total chunks sent to LLM


def _fetch_chunk_by_section_id(
    section_id: str,
    collection: chromadb.Collection,
) -> list[dict]:
    """
    Fetch all chunks belonging to a specific section_id from ChromaDB.
    Uses metadata filtering — no embedding needed.
    """
    results = collection.get(
        where={"section_id": section_id},
        include=["documents", "metadatas"],
    )
    if not results["ids"]:
        return []

    return [
        {
            "chunk_id": results["ids"][i],
            "text":     results["documents"][i],
            "metadata": results["metadatas"][i],
            "rrf_score": 0.0,
            "via_crossref": True,
        }
        for i in range(len(results["ids"]))
    ]


def _parse_cross_refs(metadata: dict) -> list[str]:
    """Extract cross_refs list from a chunk's metadata dict."""
    raw = metadata.get("cross_refs", "")
    if not raw:
        return []
    return [r.strip() for r in raw.split(",") if r.strip()]


def assemble_context(
    retrieved_chunks: list[dict],
    collection: chromadb.Collection,
) -> tuple[str, list[dict]]:
    """
    Build the full context string to send to the LLM.

    Steps:
      1. Start with the retrieved chunks
      2. For each chunk's cross_refs, fetch those sections too
      3. Deduplicate by chunk_id
      4. Format into a numbered SOURCE block

    Returns:
      context_str  : formatted string ready to insert into LLM prompt
      all_chunks   : the full list of chunks used (for citation display in UI)
    """
    # --- Step 1: Start with retrieved chunks ---
    seen_ids: set[str] = set()
    final_chunks: list[dict] = []

    for chunk in retrieved_chunks:
        cid = chunk["chunk_id"]
        if cid not in seen_ids:
            chunk["via_crossref"] = False
            final_chunks.append(chunk)
            seen_ids.add(cid)

    # --- Step 2: Follow cross-references ---
    crossref_chunks: list[dict] = []
    for chunk in list(final_chunks):  # iterate over original retrieved only
        refs = _parse_cross_refs(chunk.get("metadata", {}))
        for ref_id in refs:
            if len(crossref_chunks) >= MAX_CROSS_REF_CHUNKS:
                break
            fetched = _fetch_chunk_by_section_id(ref_id, collection)
            for fc in fetched:
                if fc["chunk_id"] not in seen_ids:
                    fc["via_crossref"] = True
                    fc["crossref_source"] = chunk["metadata"].get("section_id", "")
                    crossref_chunks.append(fc)
                    seen_ids.add(fc["chunk_id"])

    final_chunks.extend(crossref_chunks)

    # Cap total chunks
    final_chunks = final_chunks[:MAX_CONTEXT_CHUNKS]

    # --- Step 3: Format context string ---
    parts = []
    for i, chunk in enumerate(final_chunks, start=1):
        meta = chunk.get("metadata", {})
        section_id    = meta.get("section_id", "?")
        section_title = meta.get("section_title", "")
        section_path  = meta.get("section_path", "")
        page_start    = meta.get("page_start", "?")
        page_end      = meta.get("page_end", "?")

        header = f"[SOURCE {i}] Section {section_id} | {section_title} | Pages {page_start}-{page_end}"
        path_line = f"Path: {section_path}"

        extra = ""
        if chunk.get("via_crossref"):
            src = chunk.get("crossref_source", "")
            extra = f"(cross-reference from Section {src})\n" if src else "(cross-reference)\n"

        block = f"{header}\n{path_line}\n{extra}---\n{chunk['text']}"
        parts.append(block)

    context_str = "\n\n".join(parts)
    return context_str, final_chunks
