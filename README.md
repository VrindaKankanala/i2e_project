# NASA Systems Engineering Handbook QA System

A retrieval-augmented generation (RAG) system for querying the NASA Systems Engineering Handbook (SP-2016-6105 Rev2). Ask natural language questions and get answers backed by text, tables, diagrams, and callout boxes from the handbook.

---

## Features

- Hybrid search — semantic (ChromaDB) + keyword (BM25) with Reciprocal Rank Fusion
- 4 chunk types — text sections, tables, callout boxes, and diagram descriptions via Vision API
- Diagram rendering — figures displayed inline with answers, cropped from the PDF
- Cross-reference following — automatically fetches linked sections
- Streamlit UI — confidence badge, chunk type summary, collapsible sources, inline diagrams

---

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key
- NASA SE Handbook PDF (`nasa_systems_engineering_handbook_0.pdf`) placed in the project root

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/VrindaKankanala/i2e_project.git
cd i2e_project
```

**2. Install dependencies**

```bash
uv sync
```

**3. Create a `.env` file with your API key**

```
OPENAI_API_KEY=sk-...
```

**4. Add the PDF to the project root**

Download the handbook and rename/place it as:
```
nasa_systems_engineering_handbook_0.pdf
```

---

## Building the Knowledge Base

Run these four scripts in order. Only needed once (or after a `--reset`).

```bash
# Step 1 — Parse PDF, chunk text, build ChromaDB + BM25
uv run python scripts/ingest.py

# Step 2 — Extract tables
uv run python scripts/add_tables.py

# Step 3 — Extract callout boxes
uv run python scripts/add_boxes.py

# Step 4 — Describe diagrams via Vision API (~$0.01, ~10 min first run)
uv run python scripts/add_figures.py
```

After all four steps: **567 chunks** (483 text + 23 tables + 31 boxes + 30 figures)

> To start fresh: `uv run python scripts/ingest.py --reset` then re-run steps 2–4.

---

## Running the App

```bash
uv run streamlit run app/ui/streamlit_app.py
```

Open http://localhost:8501

---

## Project Structure

```
i2e_project/
├── app/
│   ├── ingestion/
│   │   ├── pdf_parser.py           # Text block extraction (PyMuPDF)
│   │   ├── structure_extractor.py  # Section hierarchy detection
│   │   ├── chunker.py              # Text chunking
│   │   ├── table_extractor.py      # Table extraction (pdfplumber)
│   │   ├── box_extractor.py        # Callout box extraction
│   │   └── image_extractor.py      # Figure detection + Vision API descriptions
│   ├── knowledge_base/
│   │   ├── vector_store.py         # ChromaDB wrapper
│   │   ├── bm25_index.py           # BM25 keyword index
│   │   ├── embedder.py             # OpenAI text-embedding-3-small
│   │   └── acronym_store.py        # NASA acronym expansion
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # RRF fusion of semantic + BM25 results
│   │   └── context_assembler.py    # Cross-reference following + context formatting
│   ├── generation/
│   │   └── llm_client.py           # GPT-4o-mini prompt and response parsing
│   └── ui/
│       └── streamlit_app.py        # Streamlit chat UI
├── scripts/
│   ├── ingest.py                   # Step 1
│   ├── add_tables.py               # Step 2
│   ├── add_boxes.py                # Step 3
│   └── add_figures.py              # Step 4
├── pyproject.toml
└── .env                            # Not committed — add your own
```

---

## Sample Questions

| Type | Question |
|------|----------|
| Text | What are the 17 technical processes in the SE Engine? |
| Text | What is the purpose of the Concept of Operations? |
| Table | What are the inputs and outputs of the Stakeholder Expectations Definition process? |
| Figure | Describe the SE Engine diagram and how processes are organized |
| Figure | What does the Vee Model show about design and verification? |
| Box | What are the key principles of systems thinking? |
| Box | What is the difference between verification and validation? |
