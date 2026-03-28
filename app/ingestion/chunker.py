"""
chunker.py

Takes the section tree (from structure_extractor) + raw blocks (from pdf_parser)
and produces a flat list of Chunk objects ready for embedding.

--- Design decisions ---

Chunk size: MAX_CHUNK_TOKENS = 600
  - Large enough to contain a complete thought / process description
  - Small enough to embed accurately and retrieve precisely
  - 600 tokens ~ 450 words ~ 2-3 paragraphs

Overlap: OVERLAP_TOKENS = 100
  - When a section is split into multiple chunks, each chunk re-includes
    the last 100 tokens of the previous chunk
  - This prevents answers from being cut off at chunk boundaries

Token estimation: len(text) // 4
  - Rough but fast. Accurate enough for sizing (no API call needed).

Section path: "Chapter 4 > 4.1 Stakeholder Expectations > 4.1.1 Process Description"
  - Stored on every chunk so the LLM always has full context hierarchy
  - Also used in citations

Parent context: every chunk includes a one-line summary of its parent section
  - e.g. a chunk from 4.1.1 includes "Parent: 4.1 Stakeholder Expectations"
  - This helps the embedding model understand context without inflating chunk size

--- Output format ---

Each Chunk is saved to data/chunks/ as JSON for inspection.
The same Chunk objects are passed to the vector store in the next step.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from app.ingestion.pdf_parser import TextBlock, parse_pdf
from app.ingestion.structure_extractor import Section, extract_structure

try:
    import fitz
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False


MAX_CHUNK_TOKENS = 600
OVERLAP_TOKENS = 100


@dataclass
class Chunk:
    chunk_id: str          # unique id: "4.1.1_0", "6.3_2"
    section_id: str        # "4.1.1"
    section_number: str    # "4.1.1"
    section_title: str     # "Process Description"
    section_path: str      # "4 System Design > 4.1 Stakeholder Expectations > 4.1.1 Process Description"
    level: int             # 1-5
    page_start: int
    page_end: int
    parent_id: str         # "4.1"
    parent_title: str      # "Stakeholder Expectations"
    text: str              # the actual content to embed and retrieve
    chunk_index: int       # 0-based index if section was split
    total_chunks: int      # total number of chunks for this section
    cross_refs: list[str]  # section ids mentioned in this section's text
    token_count: int       # approximate token count


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _block_in_box(block_bbox: tuple, box_rect) -> bool:
    """
    Return True if the text block's center point falls inside the box rect.
    Used to exclude box content from section text chunks.
    """
    x0, y0, x1, y1 = block_bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return (box_rect.x0 <= cx <= box_rect.x1 and
            box_rect.y0 <= cy <= box_rect.y1)


def _clean_block_text(text: str) -> str:
    """
    Clean a raw block's text for inclusion in a chunk.
    - Remove soft hyphens (shy hyphens used for line-break hints in PDF)
    - Normalize whitespace
    - Strip control characters except newlines
    """
    # Remove soft hyphens (\xad) — these break words mid-line in PDFs
    text = text.replace("\xad", "")
    # Remove other control characters except \n and \t
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    # Normalize multiple spaces (but preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize multiple newlines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_section_path(
    section: Section,
    sections_by_id: dict[str, Section],
) -> str:
    """
    Build a breadcrumb path like:
    "4 System Design Processes > 4.1 Stakeholder Expectations > 4.1.1 Process Description"
    """
    parts = []
    current = section
    while current:
        label = f"{current.number} {current.title}".strip()
        parts.append(label)
        parent_id = current.parent_id
        current = sections_by_id.get(parent_id) if parent_id else None
    parts.reverse()
    return " > ".join(parts)


def _split_into_chunks(
    text: str,
    section: Section,
    section_path: str,
    parent_title: str,
    sections_by_id: dict[str, Section],
) -> list[Chunk]:
    """
    Split a long text into overlapping chunks of MAX_CHUNK_TOKENS each.

    Strategy:
    1. Split text into paragraphs (on double newlines, or single newlines
       between blocks).
    2. Greedily accumulate paragraphs until we hit MAX_CHUNK_TOKENS.
    3. Start next chunk with the last OVERLAP_TOKENS worth of text.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[Chunk] = []
    current_paras: list[str] = []
    current_tokens = 0
    overlap_text = ""

    def _make_chunk(paras: list[str], index: int) -> str:
        body = "\n\n".join(paras)
        if overlap_text:
            return overlap_text + "\n\n" + body
        return body

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        if current_tokens + para_tokens > MAX_CHUNK_TOKENS and current_paras:
            # Flush current chunk
            chunk_text = _make_chunk(current_paras, len(chunks))
            chunks.append(_build_chunk(
                text=chunk_text,
                section=section,
                section_path=section_path,
                parent_title=parent_title,
                chunk_index=len(chunks),
            ))

            # Build overlap: take text from the end of current chunk
            full_text = "\n\n".join(current_paras)
            overlap_text = _tail_tokens(full_text, OVERLAP_TOKENS)

            current_paras = [para]
            current_tokens = para_tokens
        else:
            current_paras.append(para)
            current_tokens += para_tokens

    # Flush remaining
    if current_paras:
        chunk_text = _make_chunk(current_paras, len(chunks))
        chunks.append(_build_chunk(
            text=chunk_text,
            section=section,
            section_path=section_path,
            parent_title=parent_title,
            chunk_index=len(chunks),
        ))

    # Back-fill total_chunks
    total = len(chunks)
    for c in chunks:
        c.total_chunks = total

    return chunks


def _tail_tokens(text: str, n_tokens: int) -> str:
    """Return approximately the last n_tokens worth of characters from text."""
    chars = n_tokens * 4
    return text[-chars:].strip() if len(text) > chars else text.strip()


def _build_chunk(
    text: str,
    section: Section,
    section_path: str,
    parent_title: str,
    chunk_index: int,
) -> Chunk:
    chunk_id = f"{section.id}_{chunk_index}"
    return Chunk(
        chunk_id=chunk_id,
        section_id=section.id,
        section_number=section.number,
        section_title=section.title,
        section_path=section_path,
        level=section.level,
        page_start=section.page_start,
        page_end=section.page_end,
        parent_id=section.parent_id,
        parent_title=parent_title,
        text=text,
        chunk_index=chunk_index,
        total_chunks=1,  # filled in later
        cross_refs=section.cross_refs,
        token_count=_estimate_tokens(text),
    )


def build_chunks(
    blocks: list[TextBlock],
    sections_ordered: list[Section],
    sections_by_id: dict[str, Section],
    box_rects: dict | None = None,
    diagram_rects: dict | None = None,
) -> list[Chunk]:
    """
    Main entry point.

    For each section:
      1. Gather its content blocks and join into clean text
      2. If short enough: one chunk
      3. If too long: split with overlap

    Sections with zero content blocks (headings only, or empty appendix entries)
    are skipped unless they have children — in that case we create a minimal
    chunk from the heading title alone so the section is still retrievable.

    box_rects: optional dict {pdf_page_1indexed: fitz.Rect} — blocks whose
    center falls inside a box rect are skipped (they are stored separately
    as BoxChunk objects via box_extractor.py).

    Returns a flat list of all Chunk objects in document order.
    """
    all_chunks: list[Chunk] = []
    _box_rects     = box_rects     or {}
    _diagram_rects = diagram_rects or {}

    for section in sections_ordered:
        # Gather text from content blocks
        raw_texts = []
        for block_idx in section.content_blocks:
            if block_idx >= len(blocks):
                continue
            b = blocks[block_idx]
            if b.block_type == "image":
                continue
            # Skip blocks that belong to a callout box
            if _box_rects and b.page in _box_rects:
                if _block_in_box(b.bbox, _box_rects[b.page]):
                    continue
            # Skip blocks that are diagram labels inside a figure
            if _diagram_rects and b.page in _diagram_rects:
                if _block_in_box(b.bbox, _diagram_rects[b.page]):
                    continue
            cleaned = _clean_block_text(b.text)
            if cleaned:
                raw_texts.append(cleaned)

        # Join blocks into section text
        # Use double newline between blocks to mark paragraph boundaries
        section_text = "\n\n".join(raw_texts).strip()

        # If no content at all, create a stub chunk from the title
        # (so the section is still findable by its title)
        if not section_text:
            if section.title:
                section_text = f"{section.number} {section.title}"
            else:
                continue  # nothing to chunk

        # Build path and parent title for context
        section_path = _build_section_path(section, sections_by_id)
        parent = sections_by_id.get(section.parent_id)
        parent_title = parent.title if parent else ""

        # Decide: single chunk or split
        token_count = _estimate_tokens(section_text)
        if token_count <= MAX_CHUNK_TOKENS:
            chunk = _build_chunk(
                text=section_text,
                section=section,
                section_path=section_path,
                parent_title=parent_title,
                chunk_index=0,
            )
            chunk.total_chunks = 1
            all_chunks.append(chunk)
        else:
            split = _split_into_chunks(
                text=section_text,
                section=section,
                section_path=section_path,
                parent_title=parent_title,
                sections_by_id=sections_by_id,
            )
            all_chunks.extend(split)

    return all_chunks


def save_chunks(chunks: list[Chunk], output_dir: Path) -> None:
    """Save all chunks to JSON files for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # One big file with all chunks
    all_path = output_dir / "all_chunks.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks to {all_path}")


if __name__ == "__main__":
    import sys
    from collections import defaultdict

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    PDF_PATH = Path(__file__).parent.parent.parent / "nasa_systems_engineering_handbook_0.pdf"
    CHUNKS_DIR = Path(__file__).parent.parent.parent / "data" / "chunks"

    print("Parsing PDF...")
    blocks = parse_pdf(PDF_PATH)

    print("Extracting structure...")
    sections, by_id = extract_structure(blocks)

    print("Building chunks...")
    chunks = build_chunks(blocks, sections, by_id)

    # --- Stats ---
    token_counts = [c.token_count for c in chunks]
    multi_chunk_sections = [c for c in chunks if c.total_chunks > 1]
    level_dist = defaultdict(int)
    for c in chunks:
        level_dist[c.level] += 1

    print(f"\n--- Chunk Summary ---")
    print(f"Total chunks         : {len(chunks)}")
    print(f"Single-chunk sections: {len(chunks) - len(multi_chunk_sections)}")
    print(f"Split chunks         : {len(multi_chunk_sections)}")
    print(f"Avg tokens per chunk : {sum(token_counts) // len(token_counts)}")
    print(f"Max tokens           : {max(token_counts)}")
    print(f"Min tokens           : {min(token_counts)}")

    print(f"\n--- Chunks by section level ---")
    for lvl in sorted(level_dist):
        label = {1: "Chapter", 2: "Section", 3: "Subsection",
                 4: "Sub-subsection", 5: "Sub-sub"}.get(lvl, f"L{lvl}")
        print(f"  Level {lvl} ({label:16s}): {level_dist[lvl]} chunks")

    print(f"\n--- Sample chunk: section 4.1.1 ---")
    sample = next((c for c in chunks if c.section_id == "4.1.1"), None)
    if sample:
        print(f"  chunk_id     : {sample.chunk_id}")
        print(f"  section_path : {sample.section_path}")
        print(f"  pages        : {sample.page_start}-{sample.page_end}")
        print(f"  tokens       : {sample.token_count}")
        print(f"  cross_refs   : {sample.cross_refs}")
        print(f"  text preview : {repr(sample.text[:300])}")

    print(f"\n--- Largest chunks (may be split) ---")
    top5 = sorted(chunks, key=lambda c: c.token_count, reverse=True)[:5]
    for c in top5:
        print(f"  {c.section_id:15} {c.chunk_id:20} tokens={c.token_count:4}  "
              f"'{c.section_title[:40]}'")

    # Save to disk
    save_chunks(chunks, CHUNKS_DIR)
