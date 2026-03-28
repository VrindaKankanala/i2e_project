"""
streamlit_app.py

Chat UI for the NASA Systems Engineering Handbook QA system.

Layout:
  Sidebar  — system info, stats, sample questions
  Main     — chat history + input box at bottom

Each answer shows:
  - The answer text with inline [SOURCE N] citations
  - Confidence badge (High / Medium / Low)
  - Expandable "Sources" section showing each chunk used
  - Expandable "Retrieval details" showing hybrid search scores

Run with:
  uv run streamlit run app/ui/streamlit_app.py
"""

import base64
import io
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.context_assembler import assemble_context
from app.knowledge_base.vector_store import load_vector_store
from app.generation.llm_client import ask, QAResponse

load_dotenv()

PDF_PATH   = Path(__file__).parent.parent.parent / "nasa_systems_engineering_handbook_0.pdf"
PAGE_OFFSET = 10   # doc page 1 = PDF page index 10 (0-indexed)

CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "chroma_db"

CHUNK_ICONS = {
    "text":   "📄",
    "table":  "📊",
    "figure": "🖼️",
    "box":    "📦",
}

CHUNK_LABELS = {
    "text":   "Text",
    "table":  "Table",
    "figure": "Figure",
    "box":    "Callout Box",
}

SAMPLE_QUESTIONS = [
    "What is the Systems Engineering Engine?",
    "What is the difference between verification and validation?",
    "What are the entry criteria for PDR?",
    "What does TRL stand for and what are the TRL levels?",
    "What should a verification plan include?",
    "How does risk management feed into technical reviews?",
    "What are the inputs to the Stakeholder Expectations process?",
    "What is a Key Decision Point (KDP)?",
]

CONFIDENCE_COLORS = {
    "High":   "🟢",
    "Medium": "🟡",
    "Low":    "🔴",
}


# ---------------------------------------------------------------------------
# Cached resource initialisation — runs once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    retriever  = HybridRetriever()
    collection = load_vector_store(CHROMA_DIR)
    client     = OpenAI()
    return retriever, collection, client


# ---------------------------------------------------------------------------
# Core QA function
# ---------------------------------------------------------------------------

def run_qa(question: str) -> QAResponse:
    retriever, collection, client = load_resources()
    results = retriever.retrieve(question, n_results=8)
    context_str, chunks = assemble_context(results, collection)
    return ask(question, context_str, chunks, client), results


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

MIN_DIAGRAM_DRAWINGS = 10   # pages with fewer content drawings are text-only


@st.cache_data(show_spinner=False)
def render_diagram_image(pdf_page_1indexed: int, dpi: int = 150) -> bytes:
    """
    Render only the diagram area of a PDF page (vector drawings), not the whole page.

    - Finds all vector drawing elements on the page (content_drawings)
    - Computes their bounding box (union of all rects)
    - Renders only that clipped region at high DPI
    - Returns empty bytes if the page has no real diagram (< MIN_DIAGRAM_DRAWINGS)

    pdf_page_1indexed: 1-indexed PDF page number (as stored in figure chunk page_start)
    """
    pdf_idx = pdf_page_1indexed - 1
    try:
        doc = fitz.open(str(PDF_PATH))
        if pdf_idx < 0 or pdf_idx >= len(doc):
            doc.close()
            return b""

        page = doc[pdf_idx]
        drawings = page.get_drawings()

        # Only count drawings in the content area (skip header/footer)
        content_drawings = [
            d for d in drawings
            if d.get("rect") and d["rect"].y0 > 50 and d["rect"].y1 < 720
        ]

        if len(content_drawings) < MIN_DIAGRAM_DRAWINGS:
            doc.close()
            return b""   # not a diagram page — just a text reference

        # Compute bounding box of all drawing elements
        x0 = min(d["rect"].x0 for d in content_drawings)
        y0 = min(d["rect"].y0 for d in content_drawings)
        x1 = max(d["rect"].x1 for d in content_drawings)
        y1 = max(d["rect"].y1 for d in content_drawings)

        # Add padding around the diagram
        pad = 12
        clip = fitz.Rect(
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(page.rect.width,  x1 + pad),
            min(page.rect.height, y1 + pad),
        )

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes

    except Exception:
        return b""


def _parse_table_text(text: str) -> pd.DataFrame | None:
    """
    Try to parse a [TABLE] chunk's pipe-delimited text into a DataFrame.
    Returns None if parsing fails or too few rows.
    """
    lines = [l for l in text.split("\n") if "|" in l and not l.startswith("[TABLE]")]
    if len(lines) < 2:
        return None
    try:
        rows = [[cell.strip() for cell in l.split("|")] for l in lines]
        # Normalize row widths
        max_cols = max(len(r) for r in rows)
        rows = [r + [""] * (max_cols - len(r)) for r in rows]
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df
    except Exception:
        return None


def _chunk_type_badge(chunk_type: str) -> str:
    icon  = CHUNK_ICONS.get(chunk_type, "📄")
    label = CHUNK_LABELS.get(chunk_type, chunk_type.capitalize())
    return f"{icon} **{label}**"


def render_answer(response: QAResponse, retrieval_results: list) -> None:
    """Render one complete answer card."""

    # Confidence badge + chunk type summary on the same row
    confidence = response.confidence
    badge = CONFIDENCE_COLORS.get(confidence, "⚪")

    type_counts = Counter(
        c.get("metadata", {}).get("chunk_type", "text")
        for c in response.chunks_used
    )
    type_summary = "  ·  ".join(
        f"{CHUNK_ICONS.get(t,'📄')} {n} {CHUNK_LABELS.get(t,t)}"
        for t, n in sorted(type_counts.items())
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Confidence:** {badge} {confidence}")
    with col2:
        if type_summary:
            st.caption(f"Sources used: {type_summary}")

    st.markdown("---")

    # Answer text — strip CITATIONS and CONFIDENCE footer before display
    answer_display = response.answer
    # Remove everything from "CITATIONS:" onwards (LLM footer block)
    for marker in ("CITATIONS:", "CONFIDENCE:"):
        idx = answer_display.find(marker)
        if idx != -1:
            answer_display = answer_display[:idx].strip()
    st.markdown(answer_display)

    # Figures inline — render images and only show heading if at least one renders
    figure_chunks = [
        c for c in response.chunks_used
        if c.get("metadata", {}).get("chunk_type") == "figure"
    ]
    if figure_chunks:
        # Pre-render all images; skip chunks that return empty bytes (text-reference pages)
        rendered_figures = []
        for fc in figure_chunks:
            page_start = fc.get("metadata", {}).get("page_start", 0)
            try:
                img_bytes = render_diagram_image(int(page_start))
            except Exception:
                img_bytes = b""
            if img_bytes:
                rendered_figures.append((fc, img_bytes))

        if rendered_figures:
            st.markdown("**Referenced diagrams:**")
            cols = st.columns(min(len(rendered_figures), 2))
            for i, (fc, img_bytes) in enumerate(rendered_figures):
                fig_text = fc.get("text", "")
                first_line = fig_text.split("\n")[0]
                fig_label = first_line.replace("[FIGURE]", "").strip()
                cols[i % 2].image(img_bytes, caption=fig_label, width=500)

    # Sources expander
    if response.chunks_used:
        with st.expander(f"Sources used ({len(response.chunks_used)} chunks)", expanded=False):
            for i, chunk in enumerate(response.chunks_used, start=1):
                meta         = chunk.get("metadata", {})
                chunk_type   = meta.get("chunk_type", "text")
                section_id   = meta.get("section_id", "?")
                section_title= meta.get("section_title", "")
                section_path = meta.get("section_path", "")
                page_start   = meta.get("page_start", "?")
                page_end     = meta.get("page_end", "?")
                via_crossref = chunk.get("via_crossref", False)

                icon  = CHUNK_ICONS.get(chunk_type, "📄")
                label = CHUNK_LABELS.get(chunk_type, chunk_type)

                st.markdown(f"---\n**{icon} SOURCE {i} — {label}** &nbsp;·&nbsp; "
                            f"Section {section_id} {section_title} &nbsp;·&nbsp; p. {page_start}–{page_end}"
                            + (" &nbsp;*(cross-ref)*" if via_crossref else ""))
                st.caption(f"Path: {section_path}")

                if chunk_type == "figure":
                    text = chunk.get("text", "")
                    lines = [l for l in text.split("\n") if not l.startswith("[FIGURE]")]
                    # Description is the Vision API output
                    st.markdown("\n".join(lines).strip())

                elif chunk_type == "table":
                    text = chunk.get("text", "")
                    header_line = next((l for l in text.split("\n") if l.startswith("[TABLE]")), "")
                    if header_line:
                        st.caption(header_line)
                    df = _parse_table_text(text)
                    if df is not None and not df.empty:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.text(text[:800] + ("..." if len(text) > 800 else ""))

                elif chunk_type == "box":
                    text = chunk.get("text", "")
                    lines = [l for l in text.split("\n") if not l.startswith("[BOX]")]
                    st.info("\n".join(lines).strip())

                else:
                    text = chunk.get("text", "")
                    st.text(text[:800] + ("..." if len(text) > 800 else ""))

    # Retrieval details expander
    with st.expander("Retrieval details", expanded=False):
        expanded_q = retrieval_results[0].get("expanded_query", "") if retrieval_results else ""
        if expanded_q and expanded_q != retrieval_results[0].get("original_query", expanded_q):
            st.caption(f"Query expanded to: _{expanded_q}_")
        st.caption("How each chunk was found (S = semantic, B = BM25)")
        for r in retrieval_results:
            meta       = r.get("metadata", {})
            chunk_type = meta.get("chunk_type", "text")
            icon       = CHUNK_ICONS.get(chunk_type, "📄")
            s = "✓" if r.get("in_semantic") else "✗"
            b = "✓" if r.get("in_bm25") else "✗"
            st.markdown(
                f"{icon} **{r['chunk_id']}** &nbsp; RRF={r['rrf_score']:.4f} &nbsp; "
                f"Semantic {s} &nbsp; BM25 {b}  \n"
                f"_{meta.get('section_path','')[:70]}_"
            )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NASA SE Handbook QA",
    page_icon="🚀",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.title("🚀 NASA SE Handbook")
    st.caption("SP-2016-6105 Rev2 — 270 pages, 17 chapters")
    st.markdown("---")

    st.markdown("**Knowledge base**")
    st.markdown("- 483 text + 23 tables + 30 figures + 31 boxes")
    st.markdown("- Hybrid search (semantic + BM25)")
    st.markdown("- Cross-reference following")
    st.markdown("- Citations with page numbers")

    st.markdown("---")
    st.markdown("**Try a sample question:**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main area
st.title("NASA Systems Engineering Handbook — QA")
st.caption("Ask any question about the handbook. Answers are grounded to specific sections with page citations.")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_answer(msg["response"], msg["retrieval"])
        else:
            st.markdown(msg["content"])

# Handle sample question button click
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
else:
    question = None

# Chat input
user_input = st.chat_input("Ask a question about NASA systems engineering...")
if user_input:
    question = user_input

# Process question
if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate and show answer
    with st.chat_message("assistant"):
        with st.spinner("Searching handbook and generating answer..."):
            try:
                response, retrieval_results = run_qa(question)
                render_answer(response, retrieval_results)
                st.session_state.messages.append({
                    "role":      "assistant",
                    "response":  response,
                    "retrieval": retrieval_results,
                })
            except Exception as e:
                st.error(f"Error: {e}")
