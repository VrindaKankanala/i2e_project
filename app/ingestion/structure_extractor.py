"""
structure_extractor.py

Reads the flat list of TextBlocks from pdf_parser and produces a section
hierarchy tree for the NASA handbook.

--- Discovered PDF structure (from inspection) ---

Level 1 (chapter)        : "4.0  System Design Processes"    9.5pt, NOT bold
                           Pattern: X.0 suffix marks chapters
Level 2 (section)        : "4.1  Stakeholder Expectations"   16pt, bold
Level 3 (subsection)     : "4.1.1  Process Description"      12pt, NOT bold
Level 4 (sub-subsection) : "4.1.1.1  Process Activities"     10.9pt, NOT bold
Level 5 (sub-sub)        : "4.1.1.2.1  Identify Stakeholders" 10.9pt, NOT bold

Separator characters between number and title:
  - tab  (\\t)
  - en-space (\\u2002)
  - regular space
  - bell char (\\x07) — appears in some 16pt bold blocks, stripped

Footer noise to reject:
  - 13pt bold single or double digit numbers at y0 > 90% of page = PAGE NUMBERS
  - 9pt bold small text at top/bottom margin = running headers/footers

Primary detection signal: block TEXT starts with a section number pattern.
Bold / font-size are secondary confirmations, not requirements.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.ingestion.pdf_parser import TextBlock, parse_pdf


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches the section number at the START of a block's text, e.g.:
#   "4.1.2.1.3  Some Title"  -> group(1)="4.1.2.1.3"  group(2)="Some Title"
#   "A.3  Appendix Section"  -> group(1)="A.3"         group(2)="Appendix Section"
# Separator can be tab, en-space, bell, or regular spaces.
_SEC_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2}){0,4}|[A-Z](?:\.\d{1,2}){0,3})"   # section number
    r"[\t \u2002\x07]+"                                           # separator
    r"(.*)$",                                                     # title text
    re.DOTALL,
)

# A chapter block ends with ".0" — e.g. "4.0", "A.0" or just "1.0"
_CHAPTER_NUM_RE = re.compile(r"^\d{1,2}\.0$|^[A-Z]\.0$")

# Valid chapter-level numbers: 1-17 with .0, or A-J with .0
# We derive these dynamically from the regex above.
_VALID_CHAPTER_LETTERS = set("ABCDEFGHIJ")

# Footer page numbers: short bold numbers at the bottom of the page
# bbox y0 > 85% of page height = footer zone
FOOTER_Y_FRACTION = 0.85
HEADER_Y_FRACTION = 0.08

# Font size thresholds — used only to REJECT obvious non-headings
# (body text is 11pt; we allow any heading >= 9pt to catch chapter titles)
MIN_HEADING_FONT = 9.0


@dataclass
class Section:
    id: str                          # e.g. "4.1.2"
    number: str                      # same as id
    title: str                       # e.g. "Functional Analysis"
    level: int                       # 1-5
    page_start: int
    page_end: int = 0
    parent_id: str = ""
    children: list[str] = field(default_factory=list)
    content_blocks: list[int] = field(default_factory=list)
    cross_refs: list[str] = field(default_factory=list)


def _section_level(number: str) -> int:
    """
    Count dots to get level.
    "4.1"       -> 2  (section under a chapter)
    "4.1.2"     -> 3
    "4.1.2.1"   -> 4
    "4.1.2.1.3" -> 5

    Note: we never detect "X.0" chapter blocks directly — chapters are inferred
    from section number prefixes after all sections are collected.
    """
    return len(number.split("."))


def _parent_id(number: str) -> str:
    """
    "4.1.2" -> "4.1"
    "4.1"   -> "4"    (the inferred chapter node)
    "4"     -> ""
    """
    parts = number.split(".")
    if len(parts) == 1:
        return ""
    if len(parts) == 2:
        # Level-2 section: parent is the inferred chapter "4"
        return parts[0]
    return ".".join(parts[:-1])


def _is_footer_or_header(bbox: tuple, page_height: float) -> bool:
    """True if the block is in the top or bottom margin."""
    _, y0, _, y1 = bbox
    return y0 > page_height * FOOTER_Y_FRACTION or y1 < page_height * HEADER_Y_FRACTION


def _get_page_height(blocks: list[TextBlock], page: int) -> float:
    ys = [b.bbox[3] for b in blocks if b.page == page]
    return max(ys) if ys else 792.0


def _clean_title(raw: str) -> str:
    """Strip control chars and normalise whitespace in a title string."""
    cleaned = re.sub(r"[\x00-\x1f]", " ", raw)  # remove control chars
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_cross_refs(text: str) -> list[str]:
    """Extract all section/chapter references from body text."""
    refs = []
    for m in re.finditer(r"[Ss]ection\s+(\d{1,2}(?:\.\d{1,2}){0,4})", text):
        refs.append(m.group(1))
    for m in re.finditer(r"[Cc]hapter\s+(\d{1,2})", text):
        refs.append(f"{m.group(1)}.0")
    for m in re.finditer(r"[Aa]ppendix\s+([A-J](?:\.\d{1,2})?)", text):
        val = m.group(1)
        refs.append(val if "." in val else f"{val}.0")
    return list(set(refs))


def _try_parse_heading(block: TextBlock, page_height: float) -> tuple[bool, str, str]:
    """
    Returns (is_heading, section_number, title).
    """
    # Reject footer/header noise
    if _is_footer_or_header(block.bbox, page_height):
        return False, "", ""

    # Reject too-small font
    if block.font_size < MIN_HEADING_FONT:
        return False, "", ""

    text = block.text.strip()
    if not text:
        return False, "", ""

    m = _SEC_RE.match(text)
    if not m:
        return False, "", ""

    number = m.group(1).strip()
    raw_title = m.group(2)

    # Reject X.0 blocks — these are either TOC entries, running headers, or
    # appendix templates. Chapters are inferred from section prefixes instead.
    if _CHAPTER_NUM_RE.match(number):
        return False, "", ""

    # Reject single letters with no dot (e.g. bare "A", "C") — too noisy
    if len(number) == 1 and number.isalpha():
        return False, "", ""

    # Strip trailing page-reference numbers from TOC entries: "Title .  . 43" -> "Title"
    clean = re.sub(r"[\s.]+\d{1,3}\s*$", "", raw_title)
    title = _clean_title(clean)

    level = _section_level(number)

    # Level-2 sections (e.g. "4.1") must be at larger font — this guards against
    # TOC entries at 9pt slipping through.
    if level == 2 and block.font_size < 11.0:
        return False, "", ""

    return True, number, title


def extract_structure(blocks: list[TextBlock]) -> tuple[list[Section], dict[str, Section]]:
    """
    Walk all blocks in document order.
    Detect heading blocks, build Section objects, assign content blocks.

    Returns:
        sections_ordered : list[Section] in document order
        sections_by_id   : dict[str, Section]
    """
    page_heights: dict[int, float] = {}
    for b in blocks:
        if b.page not in page_heights:
            page_heights[b.page] = _get_page_height(blocks, b.page)

    sections_ordered: list[Section] = []
    sections_by_id: dict[str, Section] = {}
    current_section: Optional[Section] = None

    i = 0
    while i < len(blocks):
        block = blocks[i]
        ph = page_heights.get(block.page, 792.0)

        is_heading, number, title = _try_parse_heading(block, ph)

        if is_heading:
            # If title is empty, look ahead for it on the next block
            # (happens when section number and title are in separate spans)
            if not title and i + 1 < len(blocks):
                next_b = blocks[i + 1]
                next_ph = page_heights.get(next_b.page, 792.0)
                # Accept as continuation if: not in margin, no new section number
                if (
                    not _is_footer_or_header(next_b.bbox, next_ph)
                    and not _SEC_RE.match(next_b.text.strip())
                ):
                    title = _clean_title(next_b.text.strip())
                    i += 1  # consume the title block

            level = _section_level(number)
            parent = _parent_id(number)

            # Keep the FIRST occurrence of each section number.
            # Later duplicates come from appendix templates that reuse numbering.
            if number in sections_by_id:
                current_section = sections_by_id[number]
                i += 1
                continue

            section = Section(
                id=number,
                number=number,
                title=title,
                level=level,
                page_start=block.page,
                parent_id=parent,
            )

            if parent and parent in sections_by_id:
                parent_sec = sections_by_id[parent]
                if number not in parent_sec.children:
                    parent_sec.children.append(number)

            sections_ordered.append(section)
            sections_by_id[number] = section
            current_section = section

        else:
            if current_section is not None:
                current_section.content_blocks.append(i)
                refs = _extract_cross_refs(block.text)
                if refs:
                    current_section.cross_refs.extend(refs)

        i += 1

    # Deduplicate cross refs
    for sec in sections_ordered:
        sec.cross_refs = list(set(sec.cross_refs))

    # Fill page_end
    for idx, sec in enumerate(sections_ordered):
        for future in sections_ordered[idx + 1:]:
            if future.level <= sec.level:
                sec.page_end = future.page_start
                break
        if sec.page_end == 0:
            sec.page_end = blocks[-1].page if blocks else sec.page_start

    return sections_ordered, sections_by_id


if __name__ == "__main__":
    import sys
    from collections import defaultdict

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    PDF_PATH = Path(__file__).parent.parent.parent / "nasa_systems_engineering_handbook_0.pdf"

    print("Parsing PDF...")
    blocks = parse_pdf(PDF_PATH)

    print("Extracting structure...")
    sections, by_id = extract_structure(blocks)

    # Summary
    level_counts = defaultdict(int)
    for s in sections:
        level_counts[s.level] += 1

    print(f"\n--- Structure Summary ---")
    print(f"Total sections : {len(sections)}")
    for lvl in sorted(level_counts):
        label = {1: "Chapters", 2: "Sections", 3: "Subsections",
                 4: "Sub-subsections", 5: "Sub-sub-subsections"}.get(lvl, f"Level {lvl}")
        print(f"  Level {lvl} ({label:22s}): {level_counts[lvl]}")

    print(f"\n--- Chapter List ---")
    for s in sections:
        if s.level == 1:
            print(f"  {s.number:>5}  p{s.page_start:>3}-{s.page_end:<3}  "
                  f"children={len(s.children):3}  '{s.title[:55]}'")

    print(f"\n--- Sections under Chapter 4 ---")
    for s in sections:
        if s.number.startswith("4.") and s.level in (2, 3):
            indent = "  " * (s.level - 1)
            print(f"  {indent}{s.number:12}  p{s.page_start:>3}  "
                  f"xrefs={len(s.cross_refs)}  '{s.title[:50]}'")

    print(f"\n--- Cross-refs sample ---")
    shown = 0
    for s in sections:
        if s.cross_refs and shown < 8:
            print(f"  {s.number:12} -> {s.cross_refs[:4]}")
            shown += 1
