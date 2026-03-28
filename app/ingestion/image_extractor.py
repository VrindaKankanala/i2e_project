"""
image_extractor.py

Extracts diagrams/figures from the NASA handbook PDF using:
  1. Caption detection  — find "Figure X-Y. Title" lines in text blocks
  2. Page rendering     — PyMuPDF renders the page as a PNG image
  3. Vision API         — gpt-4o-mini describes what the diagram shows
  4. FigureChunk        — description stored as a searchable chunk

--- Why figures need special handling ---

A diagram like the Vee Model on page 6 shows the full lifecycle from
stakeholder expectations → system design → product realisation → operations,
with review gates (MDR, SDR, PDR, CDR, SIR, SAR, ORR) at each step.

This information is NOT in the surrounding text — it's only in the diagram.
Without Vision, queries like "what reviews happen during CDR?" miss this.

--- Filtering strategy ---

Only pages with detected figure captions are sent to Vision API.
This keeps costs low: ~30-40 API calls instead of 270.

A caption is detected if a text line matches:
  "Figure N-M" or "Figure N.M" (NASA uses both dash and dot separators)

--- Output format ---

Each figure becomes a FigureChunk with:
  text = "[FIGURE] Page N | Figure X-Y Title\\n<vision description>"

The [FIGURE] prefix lets BM25 treat it as a figure-type chunk.
"""

import base64
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SKIP_PAGES = 5          # skip cover pages
DPI = 150               # rendering resolution (150 = good quality, ~150KB PNG)
MAX_IMAGE_BYTES = 4_000_000   # 4 MB — OpenAI Vision limit per image
MIN_DIAGRAM_DRAWINGS = 10     # pages with fewer content drawings are text-only (not real diagrams)
DIAGRAM_Y0_MIN = 50           # ignore drawing elements in page header area
DIAGRAM_Y1_MAX = 720          # ignore drawing elements in page footer area

# Regex to detect figure captions, e.g.:
#   "Figure 2-3. Vee Model"
#   "Figure A-1. Lifecycle Phases"
#   "Figure 6.7.1-1. Technical Process Relationships"
CAPTION_RE = re.compile(
    r"Figure\s+([A-Z0-9]+[-\.][A-Z0-9][-\.A-Z0-9]*)[.:]?\s+(.+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FigureChunk:
    chunk_id:       str
    section_id:     str
    section_title:  str
    section_path:   str
    level:          int
    page:           int
    figure_id:      str        # e.g. "2-3"
    figure_title:   str        # caption text
    description:    str        # Vision API output
    text:           str        # combined [FIGURE] text for embedding/BM25
    token_count:    int
    chunk_type:     str = "figure"
    cross_refs:     list[str] = field(default_factory=list)
    parent_id:      str = ""
    parent_title:   str = ""
    page_start:     int = 0
    page_end:       int = 0


# ---------------------------------------------------------------------------
# Caption detection
# ---------------------------------------------------------------------------

def _find_figure_captions(pdf_path: Path) -> list[dict]:
    """
    Scan all text blocks for figure caption lines.
    Returns list of {page, figure_id, figure_title}.

    Uses PyMuPDF text extraction (same as pdf_parser) so we get
    exact page numbers without re-running the full parser.
    """
    captions = []
    doc = fitz.open(str(pdf_path))

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_1indexed = page_num + 1

        if page_1indexed <= SKIP_PAGES:
            continue

        text = page.get_text("text")
        for line in text.split("\n"):
            line = line.strip()
            m = CAPTION_RE.search(line)
            if m:
                captions.append({
                    "page":        page_1indexed,
                    "figure_id":   m.group(1).strip(),
                    "figure_title": m.group(2).strip()[:120],  # cap title length
                })

    doc.close()
    return captions


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def _render_page_as_base64(pdf_path: Path, page_num: int, dpi: int = DPI) -> str:
    """
    Render a PDF page (1-indexed) as a base64-encoded PNG.
    Returns the base64 string (no data URI prefix).
    """
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]  # 0-indexed

    # Scale matrix: DPI/72 (PDF default is 72 DPI)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

    # Convert to PNG bytes
    png_bytes = pix.tobytes("png")
    doc.close()

    # Truncate if too large (keep quality, reduce DPI if needed)
    if len(png_bytes) > MAX_IMAGE_BYTES:
        # Re-render at lower DPI
        doc2 = fitz.open(str(pdf_path))
        page2 = doc2[page_num - 1]
        mat2 = fitz.Matrix(100 / 72, 100 / 72)  # 100 DPI fallback
        pix2 = page2.get_pixmap(matrix=mat2, colorspace=fitz.csRGB)
        png_bytes = pix2.tobytes("png")
        doc2.close()

    return base64.b64encode(png_bytes).decode("utf-8")


# ---------------------------------------------------------------------------
# Vision API
# ---------------------------------------------------------------------------

VISION_SYSTEM_PROMPT = """You are a technical documentation analyst for NASA systems engineering.
You are looking at a page from the NASA Systems Engineering Handbook (SP-2016-6105 Rev2).

Describe the diagram or figure on this page in detail:
- What type of diagram is it? (flow chart, lifecycle diagram, process map, hierarchy, table, etc.)
- What are the main components, phases, or steps shown?
- What relationships or flows are depicted (arrows, connections)?
- What labels, acronyms, or key terms appear in the diagram?
- What systems engineering concept does it illustrate?

Be specific and thorough — your description will be used to answer questions about this diagram.
If there is no diagram (only text), say "No diagram found on this page."
Keep your response under 400 words."""


def _describe_figure(
    pdf_path: Path,
    page_num: int,
    figure_id: str,
    figure_title: str,
    client: OpenAI,
) -> str:
    """
    Send a rendered PDF page to OpenAI Vision API and get a description.
    Returns the description text.
    """
    image_b64 = _render_page_as_base64(pdf_path, page_num)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"This is page {page_num} of the NASA SE Handbook. "
                            f"It contains Figure {figure_id}: {figure_title}. "
                            f"Please describe the diagram in detail."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Section lookup (same approach as table_extractor)
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
# Public helper: diagram rect map for chunker
# ---------------------------------------------------------------------------

def get_diagram_rects(pdf_path: Path) -> dict[int, fitz.Rect]:
    """
    Scan every page of the PDF and return a dict mapping
    PDF page number (1-indexed) → bounding rect of the diagram,
    for pages that actually contain vector diagrams.

    Pages with fewer than MIN_DIAGRAM_DRAWINGS content drawing elements
    are text-only pages that just reference figures — they are skipped.

    Used by chunker.py to exclude diagram label text blocks from section chunks.
    """
    result: dict[int, fitz.Rect] = {}
    doc = fitz.open(str(pdf_path))

    for page_num in range(len(doc)):
        pdf_page_1indexed = page_num + 1
        page = doc[page_num]
        drawings = page.get_drawings()

        # Only count drawing elements in the content area
        content = [
            d for d in drawings
            if d.get("rect")
            and d["rect"].y0 > DIAGRAM_Y0_MIN
            and d["rect"].y1 < DIAGRAM_Y1_MAX
        ]

        if len(content) < MIN_DIAGRAM_DRAWINGS:
            continue  # not a real diagram page

        # Compute bounding box of all diagram elements
        x0 = min(d["rect"].x0 for d in content)
        y0 = min(d["rect"].y0 for d in content)
        x1 = max(d["rect"].x1 for d in content)
        y1 = max(d["rect"].y1 for d in content)

        pad = 6
        result[pdf_page_1indexed] = fitz.Rect(
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(page.rect.width,  x1 + pad),
            min(page.rect.height, y1 + pad),
        )

    doc.close()
    return result


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_figures(
    pdf_path: Path,
    sections_by_id: dict,
    client: OpenAI,
    rate_limit_delay: float = 0.5,
) -> list[FigureChunk]:
    """
    Find all figure captions, render each page, describe via Vision API,
    and return as FigureChunk objects.

    Args:
        pdf_path       : path to the NASA handbook PDF
        sections_by_id : dict from structure_extractor (section mapping)
        client         : OpenAI client
        rate_limit_delay: seconds to wait between Vision API calls
    """
    # Step 1: Find all figure captions
    print("Scanning for figure captions...")
    captions = _find_figure_captions(pdf_path)
    print(f"  Found {len(captions)} figure captions")

    # Deduplicate: if same page appears multiple times (caption + label),
    # keep only first occurrence per page
    seen_pages: set[int] = set()
    unique_captions = []
    for cap in captions:
        if cap["page"] not in seen_pages:
            seen_pages.add(cap["page"])
            unique_captions.append(cap)
    print(f"  {len(unique_captions)} unique pages with figures")

    # Step 2: Process each figure
    chunks: list[FigureChunk] = []

    for cap in tqdm(unique_captions, desc="Describing figures"):
        page_num    = cap["page"]
        figure_id   = cap["figure_id"]
        figure_title = cap["figure_title"]

        # Get section context
        section = _find_section_for_page(page_num, sections_by_id)
        if section is None:
            section_id    = "unknown"
            section_title = ""
            section_path  = f"Page {page_num}"
            level         = 0
            parent_id     = ""
            parent_title  = ""
        else:
            section_id    = section.id
            section_title = section.title
            section_path  = f"{section.number} {section.title}"
            level         = section.level
            parent_id     = section.parent_id
            parent_title  = sections_by_id.get(section.parent_id, section).title if section.parent_id else ""

        # Call Vision API
        try:
            description = _describe_figure(
                pdf_path, page_num, figure_id, figure_title, client
            )
        except Exception as e:
            print(f"  [WARN] Vision API failed for Figure {figure_id} (page {page_num}): {e}")
            description = f"Figure {figure_id}: {figure_title}"

        # Build combined text
        header = f"[FIGURE] Page {page_num} | Figure {figure_id}: {figure_title}"
        if section_id != "unknown":
            header += f" | Section {section_id} {section_title}"
        text = f"{header}\n\n{description}"

        token_count = max(1, len(text) // 4)
        chunk_id = f"figure_{figure_id.replace('.', '_').replace('-', '_')}_p{page_num}"

        chunks.append(FigureChunk(
            chunk_id=chunk_id,
            section_id=section_id,
            section_title=section_title,
            section_path=section_path,
            level=level,
            page=page_num,
            figure_id=figure_id,
            figure_title=figure_title,
            description=description,
            text=text,
            token_count=token_count,
            parent_id=parent_id,
            parent_title=parent_title,
            page_start=page_num,
            page_end=page_num,
        ))

        # Respect rate limits
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    return chunks


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from dotenv import load_dotenv
    load_dotenv()

    from app.ingestion.pdf_parser import parse_pdf
    from app.ingestion.structure_extractor import extract_structure

    PDF_PATH = Path(__file__).parent.parent.parent / "nasa_systems_engineering_handbook_0.pdf"

    print("Step 1: Parsing structure...")
    blocks = parse_pdf(PDF_PATH)
    _, by_id = extract_structure(blocks)

    print("\nStep 2: Finding figure captions...")
    captions = _find_figure_captions(PDF_PATH)
    print(f"  {len(captions)} captions found")
    print("\nFirst 10 figures detected:")
    for cap in captions[:10]:
        print(f"  Page {cap['page']:3d}  Figure {cap['figure_id']:<12}  {cap['figure_title'][:60]}")

    print("\nStep 3: Test — describe first figure only (Vision API)")
    client = OpenAI()
    if captions:
        cap = captions[0]
        print(f"\nDescribing Figure {cap['figure_id']} on page {cap['page']}...")
        desc = _describe_figure(PDF_PATH, cap["page"], cap["figure_id"], cap["figure_title"], client)
        print(f"\nDescription:\n{desc}")
