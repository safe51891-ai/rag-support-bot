from pathlib import Path

import pytest

from ragbot.answer import FALLBACK, answer
from ragbot.evaluate import EvalCase, evaluate
from ragbot.index import BM25Index, chunk_markdown, load_chunks

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


@pytest.fixture(scope="module")
def index():
    return BM25Index(load_chunks(DOCS))


def test_chunking_keeps_section_names(tmp_path):
    f = tmp_path / "x.md"
    f.write_text("# Title\n## Alpha\nFirst text.\n## Beta\nSecond text.\n", encoding="utf-8")
    chunks = chunk_markdown(f)
    assert [(c.section, c.text) for c in chunks] == [("Alpha", "First text."), ("Beta", "Second text.")]


def test_refund_question_cites_refund_doc(index):
    result = answer("How long do refunds take to reach my card?", index)
    assert not result.escalated
    assert result.citations[0].startswith("refunds >")
    assert "[refunds" in result.text


def test_unsupported_question_falls_back_and_escalates(index):
    result = answer("What is the warranty on the moon?", index)
    assert result.escalated
    assert result.text == FALLBACK
    assert result.citations == []


def test_retrieval_quality_meets_threshold(index):
    cases = [
        EvalCase("How many days do I have to return an item?", "refunds"),
        EvalCase("Where is my tracking number?", "shipping"),
        EvalCase("My order is late, what do I do?", "shipping"),
        EvalCase("How do I reset my password?", "accounts"),
        EvalCase("How do I turn on two factor authentication?", "accounts"),
        EvalCase("How do I close my account?", "accounts"),
    ]
    metrics = evaluate(index, cases)
    assert metrics["hit_at_k"] >= 0.83, metrics
    assert metrics["mrr"] >= 0.7, metrics
