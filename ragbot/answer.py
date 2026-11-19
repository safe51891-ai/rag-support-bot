"""Answer questions from retrieved chunks, with citations and a low-confidence fallback."""
import os
import re
from dataclasses import dataclass

from ragbot.index import BM25Index

# Calibrated on the sample docs: supported questions scored 1.2 to 10.5 (BM25), unsupported scored 0.
# Set just above 0 so that real questions are answered, and unsupported ones are escalated.
MIN_SCORE = 1.0
MIN_COVERAGE = 0.3  # chosen on the tuning set: 1 wrong answer, 5 unnecessary escalations (0.5 gave 9)
FALLBACK = "I could not find that in our help articles. I have forwarded your question to a support agent."


@dataclass
class Answer:
    text: str
    citations: list[str]
    escalated: bool


def _best_sentence(text: str, query: str) -> str:
    qwords = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return max(sentences, key=lambda s: len(qwords & set(re.findall(r"[a-z0-9]+", s.lower()))))


def decide(question: str, index: BM25Index) -> tuple[bool, object]:
    """Return (supported, best_chunk). Shared by answer() and the evaluation, so both use the same rule."""
    hits = index.search(question, k=1)
    if not hits:
        return False, None
    chunk, score = hits[0]
    supported = score >= MIN_SCORE and index.coverage(question, chunk) >= MIN_COVERAGE
    return supported, chunk


def answer(question: str, index: BM25Index, k: int = 3) -> Answer:
    supported, _ = decide(question, index)
    if not supported:
        return Answer(FALLBACK, [], escalated=True)
    hits = index.search(question, k=k)

    sources = [f"{c.source} > {c.section}" for c, _ in hits]
    if os.getenv("ANTHROPIC_API_KEY"):
        return Answer(_llm_answer(question, hits), sources, escalated=False)

    # Extractive answer: quote the most relevant sentence from the top chunk, with its citation.
    top_chunk = hits[0][0]
    sentence = _best_sentence(top_chunk.text, question)
    return Answer(f"{sentence} [{sources[0]}]", sources, escalated=False)


def _llm_answer(question, hits) -> str:
    import anthropic

    context = "\n\n".join(f"[{i + 1}] {c.source} > {c.section}\n{c.text}" for i, (c, _) in enumerate(hits))
    prompt = (
        "Answer the question using ONLY the numbered sources. Cite like [1]. "
        "If the sources do not contain the answer, reply exactly: NOT_FOUND.\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}"
    )
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.getenv("AGENT_MODEL", "claude-haiku-5-5"),
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    return FALLBACK if text.startswith("NOT_FOUND") else text
