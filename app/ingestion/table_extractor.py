"""
table_extractor.py

Extracts tables from the NASA handbook PDF using pdfplumber and converts
them into searchable text chunks.

--- Why tables need special handling ---

A table like this on page 110:
  | Review | Entry Criteria          | Success Criteria       |
  | PDR    | Preliminary design done | Board approval         |
  | CDR    | Final design complete   | All issues resolved    |

...contains information that keyword search CAN find ("PDR entry criteria")
but the text extractor in pdf_parser.py misses entirely because pdfplumber's
table detector is more accurate than PyMuPDF for structured grid content.

--- Filtering strategy (from PDF inspection) ---

Keep a table if:
  - Page > 17 (skip front matter / TOC pages)
  - Rows >= MIN_ROWS (3) — filters out header/footer fragments
  - Cols >= MIN_COLS (3) — filters out 2-col text that isn't really a table
  - Non-empty cell ratio >= MIN_FILL (0.35) — filters sparse diagram labels

--- Output format ---

Each table becomes a string like:
  [TABLE] Page 28 | Section 2.7 SE Competency Model
  Competency Area | Competency | Description
  SE 1.0 System Design | SE 1.1 Stakeholder Expectations | Eliciting and defining use cases...
  SE 1.0 System Design | SE 1.2 Technical Requirements | Transforming the baseline...

This format lets BM25 and semantic search both find it by:
  - Column headers ("Entry Criteria", "Success Criteria")
  - Cell values ("PDR", "CDR", "preliminary design")
  - Section context ("Section 2.7")
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber
from tqdm import tqdm

MIN_ROWS = 3
MIN_COLS = 3
MIN_FILL = 0.35    # fraction of cells that must be non-empty
SKIP_PAGES = 17    # ignore front matter


@dataclass
class TableChunk:
    chunk_id: str
    section_id: str
    section_title: str
    section_path: str
    level: int
    page: int
    table_index: int       # which table on the page (0-indexed)
    n_rows: int
    n_cols: int
    text: str              # formatted table text for embedding
    raw_headers: list[str] # first row of the table
    token_count: int
    chunk_type: str = "table"
    cross_refs: list[str] = field(default_factory=list)
    parent_id: str = ""
    parent_title: str = ""
    page_start: int = 0
    page_end: int = 0


def _clean_cell(cell) -> str:
    """Clean a single table cell value."""
    if cell is None:
        return ""
    text = str(cell)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _table_to_text(table: list[list], page: int, section_id: str, section_title: str) -> str:
    """
    Convert a 2D table (list of rows) into a readable text block.

    Format:
      [TABLE] Page N | Section X.Y Title
      Header1 | Header2 | Header3
      val1 | val2 | val3
      ...
    """
    lines = [f"[TABLE] Page {page} | Section {section_id} {section_title}"]

    for row in table:
        cells = [_clean_cell(c) for c in row]
        # Skip completely empty rows
        if not any(cells):
            continue
        lines.append(" | ".join(cells))

    return "\n".join(lines)


def _is_real_table(table: list[list], page: int) -> bool:
    """Filter out noise: front matter, sparse tables, header/footer fragments."""
    if page <= SKIP_PAGES:
        return False
    if len(table) < MIN_ROWS:
        return False
    if not table[0] or len(table[0]) < MIN_COLS:
        return False

    # Check fill ratio
    total = sum(len(row) for row in table)
    filled = sum(1 for row in table for cell in row if _clean_cell(cell))
    if total == 0 or filled / total < MIN_FILL:
        return False

    return True


def _find_section_for_page(page: int, sections_by_id: dict) -> Optional[object]:
    """
    Find the deepest section that contains this page.
    Walk all sections and return the one with the narrowest page range
    that still contains the given page.
    """
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


def extract_tables(
    pdf_path: Path,
    sections_by_id: dict,
) -> list[TableChunk]:
    """
    Extract all valid tables from the PDF and return as TableChunk objects.

    Args:
        pdf_path      : path to the NASA handbook PDF
        sections_by_id: dict from structure_extractor (used to assign sections)
    """
    chunks: list[TableChunk] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_obj in tqdm(pdf.pages, desc="Extracting tables"):
            page_num = page_obj.page_number  # 1-indexed

            tables = page_obj.extract_tables()
            if not tables:
                continue

            real_tables = [t for t in tables if _is_real_table(t, page_num)]
            if not real_tables:
                continue

            # Find which section this page belongs to
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

            for t_idx, table in enumerate(real_tables):
                text = _table_to_text(table, page_num, section_id, section_title)
                headers = [_clean_cell(c) for c in (table[0] if table else [])]
                n_rows = len([r for r in table if any(_clean_cell(c) for c in r)])
                n_cols = len(table[0]) if table else 0
                token_count = max(1, len(text) // 4)

                chunk_id = f"table_p{page_num}_{t_idx}"

                chunks.append(TableChunk(
                    chunk_id=chunk_id,
                    section_id=section_id,
                    section_title=section_title,
                    section_path=section_path,
                    level=level,
                    page=page_num,
                    table_index=t_idx,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    text=text,
                    raw_headers=headers,
                    token_count=token_count,
                    parent_id=parent_id,
                    parent_title=parent_title,
                    page_start=page_num,
                    page_end=page_num,
                ))

    return chunks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from app.ingestion.pdf_parser import parse_pdf
    from app.ingestion.structure_extractor import extract_structure

    PDF_PATH = Path(__file__).parent.parent.parent / "nasa_systems_engineering_handbook_0.pdf"

    print("Parsing structure...")
    blocks = parse_pdf(PDF_PATH)
    _, by_id = extract_structure(blocks)

    print("Extracting tables...")
    table_chunks = extract_tables(PDF_PATH, by_id)

    print(f"\n--- Table Extraction Summary ---")
    print(f"Total table chunks : {len(table_chunks)}")

    print(f"\n--- Sample tables ---")
    for tc in table_chunks[:5]:
        print(f"\n  chunk_id     : {tc.chunk_id}")
        print(f"  section      : {tc.section_id} | {tc.section_title}")
        print(f"  page         : {tc.page}")
        print(f"  size         : {tc.n_rows} rows x {tc.n_cols} cols")
        print(f"  headers      : {tc.raw_headers}")
        print(f"  text preview :")
        for line in tc.text.split("\n")[:5]:
            print(f"    {line[:90]}")
