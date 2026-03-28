"""
acronym_store.py

Builds an acronym dictionary from the handbook text itself, then uses it
to expand queries before retrieval.

--- How acronyms are defined in the handbook ---

Pattern 1 (most common):
  "Key Decision Point (KDP)"       -> KDP = Key Decision Point
  "Technology Readiness Level (TRL)" -> TRL = Technology Readiness Level

Pattern 2 (reversed):
  "(KDP) Key Decision Point"       -> KDP = Key Decision Point

Pattern 3 (acronym first, defined nearby):
  "TRL — Technology Readiness Level"

We scan all chunk texts for these patterns and build a dict:
  { "KDP": "Key Decision Point", "TRL": "Technology Readiness Level", ... }

--- Query expansion ---

Before sending to the retriever:
  "What is KDP?"
  -> finds KDP in dict
  -> "What is KDP (Key Decision Point)?"

This improves both BM25 (keyword match on full form) and semantic
(embedding now contains full meaning of the acronym).
"""

import json
import re
from pathlib import Path

ACRONYM_FILE = Path(__file__).parent.parent.parent / "data" / "chunks" / "acronyms.json"

# Matches: "Full Form (ACRONYM)" — acronym is 2-8 uppercase letters/digits
_PATTERN_FWD = re.compile(
    r"([A-Z][a-zA-Z\s\-]{2,60}?)\s*\(([A-Z][A-Z0-9\-]{1,7})\)"
)

# Matches: "(ACRONYM) Full Form"
_PATTERN_REV = re.compile(
    r"\(([A-Z][A-Z0-9\-]{1,7})\)\s+([A-Z][a-zA-Z\s\-]{2,60})"
)

# Matches: "ACRONYM — Full Form" or "ACRONYM - Full Form"
_PATTERN_DASH = re.compile(
    r"\b([A-Z][A-Z0-9\-]{1,7})\s+[—\-]\s+([A-Z][a-zA-Z\s\-]{2,60})"
)

# Noise acronyms to exclude (common English words, Roman numerals, etc.)
_EXCLUDE = {
    "A", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN",
    "IS", "IT", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US",
    "WE", "I", "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI",
    "NASA", "US", "USA",  # too broad — NASA is self-evident
}


def build_acronym_dict(chunks: list) -> dict[str, str]:
    """
    Scan all chunk texts and extract acronym -> full form mappings.

    chunks: list of Chunk objects or dicts (works with either).
    Returns dict sorted by acronym length descending (longer first for expansion).
    """
    found: dict[str, str] = {}

    for chunk in chunks:
        text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
        if not text:
            continue

        # Pattern 1: Full Form (ACRONYM)
        for m in _PATTERN_FWD.finditer(text):
            full_form = m.group(1).strip().rstrip("(, ")
            acronym   = m.group(2).strip()
            if _is_valid(acronym, full_form):
                found[acronym] = full_form

        # Pattern 2: (ACRONYM) Full Form
        for m in _PATTERN_REV.finditer(text):
            acronym   = m.group(1).strip()
            full_form = m.group(2).strip()
            if _is_valid(acronym, full_form):
                found[acronym] = full_form

        # Pattern 3: ACRONYM — Full Form
        for m in _PATTERN_DASH.finditer(text):
            acronym   = m.group(1).strip()
            full_form = m.group(2).strip()
            if _is_valid(acronym, full_form):
                found[acronym] = full_form

    # Sort by acronym length descending so longer acronyms are replaced first
    # (prevents "SE" from being replaced before "SEMP")
    sorted_dict = dict(
        sorted(found.items(), key=lambda x: len(x[0]), reverse=True)
    )

    print(f"Extracted {len(sorted_dict)} acronyms from handbook text")
    return sorted_dict


def _is_valid(acronym: str, full_form: str) -> bool:
    """Filter out noise matches."""
    if acronym in _EXCLUDE:
        return False
    if len(acronym) < 2 or len(acronym) > 8:
        return False
    if len(full_form) < 4:
        return False
    # Full form should start with a capital letter that matches acronym start
    if not full_form[0].upper() == acronym[0]:
        return False
    return True


def save_acronym_dict(acronyms: dict[str, str]) -> None:
    ACRONYM_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACRONYM_FILE, "w", encoding="utf-8") as f:
        json.dump(acronyms, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(acronyms)} acronyms to {ACRONYM_FILE}")


def load_acronym_dict() -> dict[str, str]:
    if not ACRONYM_FILE.exists():
        return {}
    with open(ACRONYM_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_query(query: str, acronyms: dict[str, str]) -> str:
    """
    Expand acronyms in a query string.

    Example:
      query    = "What is KDP and how does it relate to TRL?"
      expanded = "What is KDP (Key Decision Point) and how does it relate to TRL (Technology Readiness Level)?"

    Only expands standalone acronym tokens (word boundaries) to avoid
    partial matches inside other words.
    """
    if not acronyms:
        return query

    expanded = query
    for acronym, full_form in acronyms.items():
        # Only match whole-word occurrences not already followed by "("
        pattern = rf"\b{re.escape(acronym)}\b(?!\s*\()"
        replacement = f"{acronym} ({full_form})"
        expanded = re.sub(pattern, replacement, expanded)

    return expanded


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    import json as _json
    from pathlib import Path as _Path

    chunks_file = _Path(__file__).parent.parent.parent / "data" / "chunks" / "all_chunks.json"
    print("Loading chunks...")
    with open(chunks_file, encoding="utf-8") as f:
        chunks = _json.load(f)

    print("Building acronym dictionary...")
    acronyms = build_acronym_dict(chunks)
    save_acronym_dict(acronyms)

    print(f"\n--- Sample acronyms (first 30) ---")
    for i, (k, v) in enumerate(acronyms.items()):
        if i >= 30:
            break
        print(f"  {k:10} = {v}")

    print(f"\n--- Query expansion examples ---")
    test_queries = [
        "What is KDP?",
        "How does TRL relate to PDR and CDR?",
        "What is the SE engine?",
        "Explain the WBS and its role in project planning",
    ]
    for q in test_queries:
        expanded = expand_query(q, acronyms)
        if expanded != q:
            print(f"  IN : {q}")
            print(f"  OUT: {expanded}")
        else:
            print(f"  (no expansion) {q}")
        print()
