"""Retrieval evaluation: hit@k and MRR over a labeled question set. Run this on every change to chunking or ranking."""
from dataclasses import dataclass

from ragbot.index import BM25Index


@dataclass
class EvalCase:
    question: str
    expected_source: str  # the document stem that should be retrieved


def evaluate(index: BM25Index, cases: list[EvalCase], k: int = 3) -> dict[str, float]:
    hits, reciprocal_ranks = 0, 0.0
    for case in cases:
        results = [c.source for c, _ in index.search(case.question, k=k)]
        if case.expected_source in results:
            hits += 1
            reciprocal_ranks += 1 / (results.index(case.expected_source) + 1)
    n = len(cases)
    return {"hit_at_k": round(hits / n, 3), "mrr": round(reciprocal_ranks / n, 3), "n": n}
