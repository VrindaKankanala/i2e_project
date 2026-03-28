"""
llm_client.py

Sends the assembled context + user question to OpenAI GPT-4o-mini
and returns a structured answer with citations.

Model choice — GPT-4o-mini:
  - $0.15/1M input tokens, $0.60/1M output tokens
  - Each QA call ~ 2000 input tokens + 500 output tokens = ~$0.0006 per question
  - Fast: typically 2-4 seconds
  - Strong instruction-following for citation formatting

System prompt design:
  - Tells the model it is a technical handbook assistant
  - Instructs it to cite sources using [SOURCE N] markers
  - Tells it to say "not found" if the answer isn't in the sources
    (prevents hallucination by grounding strictly to provided context)
  - Asks for a confidence level so the UI can display it
"""

from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GENERATION_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a technical assistant for the NASA Systems Engineering Handbook (SP-2016-6105 Rev2).

You will be given a set of numbered SOURCE blocks extracted from the handbook, followed by a user question.

Your job:
1. Answer the question using ONLY information from the provided sources.
2. Write a clean, readable answer with NO inline citation markers like [SOURCE N].
   The sources are already shown to the user separately — do not clutter the answer with them.
3. If the sources do not contain enough information to answer, say:
   "The provided sections do not contain sufficient information to answer this question."
4. At the end of your answer, output a CITATIONS block listing each source you used:

CITATIONS:
- [SOURCE 1] Section X.Y | Title | Pages A-B
- [SOURCE 2] Section X.Z | Title | Pages C-D

5. On the last line, output your confidence:
CONFIDENCE: High / Medium / Low

Rules:
- Do not invent information not present in the sources.
- Keep answers concise but complete.
- Use technical terminology as it appears in the handbook.
- If a source contains a cross-reference to another section, note it but do not speculate about that section's content unless it was also provided."""


@dataclass
class QAResponse:
    answer: str              # full answer text with inline citations
    citations: list[dict]    # list of {source_num, section_id, title, pages}
    confidence: str          # "High", "Medium", or "Low"
    model: str               # which model was used
    chunks_used: list[dict]  # the raw chunks that were passed as context


def _parse_citations(answer_text: str) -> list[dict]:
    """Extract the CITATIONS block from the model's response."""
    import re
    citations = []
    # Find lines like: - [SOURCE 1] Section 4.1 | Process Description | Pages 55-63
    pattern = re.compile(
        r"\[SOURCE\s+(\d+)\]\s+Section\s+([\d.A-Za-z]+)\s*\|\s*([^|]+?)\s*\|\s*Pages\s*([\d\-]+)"
    )
    for m in pattern.finditer(answer_text):
        citations.append({
            "source_num":  int(m.group(1)),
            "section_id":  m.group(2).strip(),
            "title":       m.group(3).strip(),
            "pages":       m.group(4).strip(),
        })
    return citations


def _parse_confidence(answer_text: str) -> str:
    """Extract confidence level from the model's response."""
    import re
    m = re.search(r"CONFIDENCE:\s*(High|Medium|Low)", answer_text, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Medium"


def ask(
    question: str,
    context_str: str,
    chunks_used: list[dict],
    client: OpenAI | None = None,
) -> QAResponse:
    """
    Send question + context to GPT-4o-mini and return a structured response.

    Args:
        question    : the user's natural language question
        context_str : formatted SOURCE blocks from context_assembler.py
        chunks_used : the raw chunk dicts (for UI display)
        client      : optional pre-initialised OpenAI client
    """
    if client is None:
        client = OpenAI()

    user_message = f"""SOURCES:
{context_str}

QUESTION: {question}"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.1,   # low temperature = factual, consistent answers
    )

    answer_text = response.choices[0].message.content or ""

    return QAResponse(
        answer=answer_text,
        citations=_parse_citations(answer_text),
        confidence=_parse_confidence(answer_text),
        model=GENERATION_MODEL,
        chunks_used=chunks_used,
    )
