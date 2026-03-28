"""
pdf_parser.py

Extracts raw text blocks from the NASA PDF using PyMuPDF.
Each block carries:
  - page       : 1-indexed page number
  - text       : the actual string
  - font_size  : size of the dominant font in the block
  - is_bold    : True if the dominant font is bold
  - block_type : "text" or "image" (image blocks noted but not processed here)
  - bbox       : (x0, y0, x1, y1) bounding box on the page

Why we care about font_size and is_bold:
  PyMuPDF gives us rich typography metadata. Chapter headings (e.g. "Chapter 4")
  are rendered at ~16-20pt bold. Section headings (e.g. "4.3 Logical Decomposition")
  are ~12-14pt bold. Body text is ~10pt normal. We use these signals in
  structure_extractor.py to detect the section hierarchy without needing the TOC.
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextBlock:
    page: int               # 1-indexed
    text: str               # cleaned text content
    font_size: float        # dominant font size in this block
    is_bold: bool           # True if dominant font contains "Bold" or "bold"
    block_type: str         # "text" | "image"
    bbox: tuple             # (x0, y0, x1, y1)
    fonts: list = field(default_factory=list)  # all font names seen in block


def parse_pdf(pdf_path: str | Path) -> list[TextBlock]:
    """
    Open the PDF and extract all text blocks from every page.

    PyMuPDF represents each page as a tree of:
      Page → Blocks → Lines → Spans
    A Span is the smallest unit — a run of text with uniform font/size.
    We aggregate spans up to the block level and pick the dominant font.

    Returns a flat list of TextBlock objects, in page order.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    blocks: list[TextBlock] = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # get_text("rawdict") gives us the full span-level detail:
        # blocks → lines → spans, each span has: text, font, size, flags, bbox
        page_dict = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in page_dict["blocks"]:
            # block["type"] == 0 means text; 1 means image
            if block["type"] == 1:
                # Image block — record its presence and bounding box for later
                blocks.append(TextBlock(
                    page=page_num + 1,
                    text="[IMAGE]",
                    font_size=0.0,
                    is_bold=False,
                    block_type="image",
                    bbox=tuple(block["bbox"]),
                ))
                continue

            # --- Collect all spans in this text block ---
            spans = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    raw_chars = span.get("chars", [])
                    span_text = "".join(c.get("c", "") for c in raw_chars)
                    if span_text.strip():
                        spans.append({
                            "text": span_text,
                            "size": span.get("size", 0.0),
                            "font": span.get("font", ""),
                            "flags": span.get("flags", 0),
                        })

            if not spans:
                continue

            # --- Determine dominant font size (by total character count) ---
            full_text = "".join(s["text"] for s in spans)
            if not full_text.strip():
                continue

            # Weight each span's font size by number of characters
            total_chars = sum(len(s["text"]) for s in spans)
            weighted_size = sum(
                s["size"] * len(s["text"]) for s in spans
            ) / total_chars if total_chars > 0 else 0.0

            # Check boldness: PyMuPDF flag bit 4 (16) = bold, or font name contains "Bold"
            bold_chars = sum(
                len(s["text"])
                for s in spans
                if (s["flags"] & 16) or "bold" in s["font"].lower()
            )
            is_bold = bold_chars > (total_chars * 0.5)  # majority bold = bold block

            all_fonts = list({s["font"] for s in spans})

            blocks.append(TextBlock(
                page=page_num + 1,
                text=full_text,
                font_size=round(weighted_size, 2),
                is_bold=is_bold,
                block_type="text",
                bbox=tuple(block["bbox"]),
                fonts=all_fonts,
            ))

    doc.close()
    return blocks


def get_font_size_distribution(blocks: list[TextBlock]) -> dict:
    """
    Analyse the distribution of font sizes across all text blocks.
    This helps us understand what sizes correspond to headings vs body text
    BEFORE we build the heading detector.

    Returns a dict: {font_size: count_of_blocks}
    """
    from collections import Counter
    sizes = [round(b.font_size) for b in blocks if b.block_type == "text" and b.font_size > 0]
    return dict(sorted(Counter(sizes).items(), reverse=True))


if __name__ == "__main__":
    # Quick inspection run — prints stats and sample blocks
    import json

    PDF_PATH = Path(__file__).parent.parent.parent / "nasa_systems_engineering_handbook_0.pdf"

    print(f"Parsing: {PDF_PATH.name}")
    blocks = parse_pdf(PDF_PATH)

    text_blocks = [b for b in blocks if b.block_type == "text"]
    image_blocks = [b for b in blocks if b.block_type == "image"]

    print(f"\n--- Summary ---")
    print(f"Total blocks  : {len(blocks)}")
    print(f"Text blocks   : {len(text_blocks)}")
    print(f"Image blocks  : {len(image_blocks)}")

    print(f"\n--- Font Size Distribution (size: block_count) ---")
    dist = get_font_size_distribution(blocks)
    for size, count in list(dist.items())[:15]:
        print(f"  {size:5} pt  ->  {count} blocks")

    print(f"\n--- Bold blocks by font size (likely headings) ---")
    from collections import defaultdict
    bold_by_size = defaultdict(list)
    for b in text_blocks:
        if b.is_bold:
            bold_by_size[round(b.font_size)].append(b.text.strip()[:80])
    for size in sorted(bold_by_size.keys(), reverse=True)[:8]:
        samples = bold_by_size[size][:3]
        print(f"\n  {size}pt bold ({len(bold_by_size[size])} blocks):")
        for s in samples:
            print(f"    >> {repr(s)}")

    print(f"\n--- Sample: Pages 1-3, first 5 text blocks ---")
    early = [b for b in text_blocks if b.page <= 3][:5]
    for b in early:
        print(f"  [p{b.page}] size={b.font_size} bold={b.is_bold} | {repr(b.text[:100])}")
