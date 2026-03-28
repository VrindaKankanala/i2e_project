"""
box_extractor.py

Extracts callout boxes from the NASA handbook PDF.

The handbook has a "Table of Boxes" index on page 7 listing ~30 named boxes,
each on a specific document page (e.g. "Methods of Verification ... 93").

Each box is rendered as a large bordered rectangle below the page header.
PyMuPDF's page.get_drawings() detects these rectangles; we then clip text
extraction to that rectangle to get exactly the box content.

--- Why boxes need separate chunks ---

Boxes like "Differences between Verification and Validation Testing" (p89)
contain precise, structured information that is NOT repeated in surrounding
body text. When merged into the section chunk they become diluted and may
not rank high enough for targeted queries like:
  "What are the methods of verification?"
  "What is the difference between verification and validation?"

As separate chunks with chunk_type="box" they surface directly.

--- Page offset ---

The PDF has 10 pages of front matter (roman-numeral pages i–x) before
document page 1. So: PDF_index = doc_page + PAGE_OFFSET - 1  (0-indexed)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from tqdm import tqdm

PAGE_OFFSET = 10   # doc page 1 = PDF page index 10
BOX_Y0_MIN  = 40   # ignore rects in the top header area (y0 < 40)
BOX_MIN_W   = 100  # minimum box width in points
BOX_MIN_H   = 40   # minimum box height in points
TOC_PAGE    = 7    # PDF page number of the Table of Boxes (1-indexed)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class BoxChunk:
    chunk_id:      str
    section_id:    str
    section_title: str
    section_path:  str
    level:         int
    page:          int           # document page number (matches Table of Boxes)
    box_title:     str           # title from Table of Boxes index
    text:          str           # full box text for embedding/BM25
    token_count:   int
    chunk_type:    str = "box"
    cross_refs:    list[str] = field(default_factory=list)
    parent_id:     str = ""
    parent_title:  str = ""
    page_start:    int = 0
    page_end:      int = 0


# ---------------------------------------------------------------------------
# Parse Table of Boxes index
# ---------------------------------------------------------------------------

def _parse_table_of_boxes(pdf_path: Path) -> list[dict]:
    """
    Extract the list of boxes from the 'Table of Boxes' index page.
    Returns list of {title, doc_page}.

    The index lines look like:
      'Methods of Verification .  .   .   .   .   .   . 93'
      'Space Flight Phase A:\nConcept and Technology Development .  .  24'
    """
    doc = fitz.open(str(pdf_path))
    page = doc[TOC_PAGE - 1]  # 0-indexed
    text = page.get_text("text")
    doc.close()

    # Each entry ends with a page number after dots/spaces
    # Pattern: any text followed by spaces/dots then a number at end of line
    entry_re = re.compile(r"^(.+?)\s*[.\s]{3,}\s*(\d+)\s*$")

    entries = []
    current_title_parts = []

    for line in text.split("\n"):
        line = line.strip()
        if not line or line in ("Table of Boxes", "NASA SYSTEMS ENGINEERING HANDBOOK", "vii"):
            current_title_parts = []
            continue

        m = entry_re.match(line)
        if m:
            # Line ends with a page number
            title_part = m.group(1).strip()
            page_num = int(m.group(2))
            # Combine with any continuation lines from before
            if current_title_parts:
                current_title_parts.append(title_part)
                full_title = " ".join(current_title_parts)
                current_title_parts = []
            else:
                full_title = title_part
            entries.append({"title": full_title, "doc_page": page_num})
        else:
            # Continuation line (e.g. "Space Flight Phase A:" followed by next line)
            if line and not line.startswith("Table"):
                current_title_parts.append(line)

    return entries


# ---------------------------------------------------------------------------
# Box rectangle detection on a page
# ---------------------------------------------------------------------------

def _find_box_rect(page: fitz.Page) -> Optional[fitz.Rect]:
    """
    Find the primary callout box rectangle on a PDF page.

    Strategy:
    - Use get_drawings() to find stroke rectangles
    - Filter out the header (y0 < BOX_Y0_MIN) and tiny decorative elements
    - Return the largest qualifying rectangle (the box border)
    """
    drawings = page.get_drawings()

    candidates = []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        if r.y0 < BOX_Y0_MIN:
            continue  # header area
        if r.width < BOX_MIN_W or r.height < BOX_MIN_H:
            continue  # too small
        # Accept both stroke (border only) and fill types
        if d.get("type") in ("s", "f", "fs"):
            candidates.append(r)

    if not candidates:
        return None

    # Return the largest rectangle by area
    return max(candidates, key=lambda r: r.width * r.height)


# ---------------------------------------------------------------------------
# Section lookup
# ---------------------------------------------------------------------------

def _find_section_for_page(page: int, sections_by_id: dict) -> Optional[object]:
    """Find the deepest section that contains this page."""
    best = None
    best_range = float("inf")
    for sec in sections_by_id.values():
        start = sec.page_start
        end = sec.page_end if sec.page_end > 0 else 9999
        if start <= page <= end:
            page_range = end - start
            if page_range < best_range:
                best_range = page_range
                best = sec
    return best


# ---------------------------------------------------------------------------
# Public helper: box rect map for chunker
# ---------------------------------------------------------------------------

def get_box_rects(pdf_path: Path) -> dict[int, fitz.Rect]:
    """
    Return a dict mapping PDF page number (1-indexed) to the box rect on that page.
    Used by chunker.py to skip text blocks that fall inside a box.

    Example: {99: Rect(54, 81, 522, 332), 103: Rect(54, 81, 522, 550), ...}
    """
    box_index = _parse_table_of_boxes(pdf_path)
    doc = fitz.open(str(pdf_path))
    result: dict[int, fitz.Rect] = {}

    for entry in box_index:
        doc_page = entry["doc_page"]
        pdf_page_1indexed = doc_page + PAGE_OFFSET   # TextBlock.page is 1-indexed
        pdf_idx = pdf_page_1indexed - 1              # fitz is 0-indexed

        if pdf_idx >= len(doc):
            continue

        rect = _find_box_rect(doc[pdf_idx])
        if rect is not None:
            result[pdf_page_1indexed] = rect

    doc.close()
    return result


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_boxes(
    pdf_path: Path,
    sections_by_id: dict,
) -> list[BoxChunk]:
    """
    Extract all callout boxes from the PDF using the Table of Boxes index.

    Args:
        pdf_path       : path to the NASA handbook PDF
        sections_by_id : dict from structure_extractor (section mapping)

    Returns list of BoxChunk objects.
    """
    # Step 1: Parse the Table of Boxes index
    box_index = _parse_table_of_boxes(pdf_path)
    print(f"  Found {len(box_index)} entries in Table of Boxes index")

    doc = fitz.open(str(pdf_path))
    chunks: list[BoxChunk] = []

    for entry in tqdm(box_index, desc="Extracting boxes"):
        doc_page  = entry["doc_page"]
        box_title = entry["title"]
        pdf_idx   = doc_page + PAGE_OFFSET - 1  # convert to 0-indexed

        if pdf_idx >= len(doc):
            print(f"  [WARN] doc page {doc_page} out of range (PDF has {len(doc)} pages)")
            continue

        page = doc[pdf_idx]

        # Step 2: Detect the box rectangle
        box_rect = _find_box_rect(page)
        if box_rect is None:
            print(f"  [WARN] No box rect found on doc page {doc_page} ({box_title})")
            continue

        # Step 3: Extract text within the box
        raw_text = page.get_text("text", clip=box_rect).strip()
        if not raw_text:
            print(f"  [WARN] Empty text for box on doc page {doc_page} ({box_title})")
            continue

        # Clean up extracted text
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        # Remove the title line if it's duplicated at the top (all-caps)
        if lines and lines[0].upper() == box_title.upper():
            lines = lines[1:]
        elif lines and lines[0].upper().replace(" ", "") == box_title.upper().replace(" ", ""):
            lines = lines[1:]

        body_text = "\n".join(lines)

        # Build the full chunk text with [BOX] prefix
        text = f"[BOX] {box_title}\n\n{body_text}"

        # Step 4: Find section context
        section = _find_section_for_page(doc_page, sections_by_id)
        if section is None:
            section_id    = "unknown"
            section_title = ""
            section_path  = f"Page {doc_page}"
            level         = 0
            parent_id     = ""
            parent_title  = ""
        else:
            section_id    = section.id
            section_title = section.title
            section_path  = f"{section.number} {section.title}"
            level         = section.level
            parent_id     = section.parent_id
            parent_title  = (
                sections_by_id.get(section.parent_id, section).title
                if section.parent_id else ""
            )

        chunk_id = f"box_p{doc_page}_{re.sub(r'[^a-z0-9]', '_', box_title.lower())[:30]}"
        token_count = max(1, len(text) // 4)

        chunks.append(BoxChunk(
            chunk_id=chunk_id,
            section_id=section_id,
            section_title=section_title,
            section_path=section_path,
            level=level,
            page=doc_page,
            box_title=box_title,
            text=text,
            token_count=token_count,
            parent_id=parent_id,
            parent_title=parent_title,
            page_start=doc_page,
            page_end=doc_page,
        ))

    doc.close()
    return chunks


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from app.ingestion.pdf_parser import parse_pdf
    from app.ingestion.structure_extractor import extract_structure

    PDF_PATH = Path(__file__).parent.parent.parent / "nasa_systems_engineering_handbook_0.pdf"

    print("Parsing Table of Boxes index...")
    box_index = _parse_table_of_boxes(PDF_PATH)
    print(f"  {len(box_index)} boxes found:")
    for b in box_index:
        print(f"    p{b['doc_page']:3d}  {b['title']}")

    print("\nParsing structure...")
    blocks = parse_pdf(PDF_PATH)
    _, by_id = extract_structure(blocks)

    print("\nExtracting box chunks...")
    chunks = extract_boxes(PDF_PATH, by_id)

    print(f"\n--- Box Extraction Summary ---")
    print(f"Total box chunks: {len(chunks)}")
    print(f"\n--- Sample boxes ---")
    for bc in chunks[:5]:
        print(f"\n  chunk_id : {bc.chunk_id}")
        print(f"  title    : {bc.box_title}")
        print(f"  section  : {bc.section_id} | {bc.section_title}")
        print(f"  page     : {bc.page}")
        print(f"  preview  :")
        for line in bc.text.split("\n")[:6]:
            print(f"    {line[:90]}")
