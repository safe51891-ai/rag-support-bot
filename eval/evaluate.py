"""Evaluate retrieval and escalation on labeled questions. Run: python -m eval.evaluate

Compares a baseline index with the improved one (stemming and synonyms), so each change is measured.
"""
import json
from pathlib import Path

from ragbot.answer import decide
from ragbot.index import BM25Index, load_chunks

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
CHUNKS = load_chunks(ROOT / "data" / "docs")


HOLDOUT = json.loads((ROOT / "eval" / "holdout.json").read_text(encoding="utf-8"))


def run(index: BM25Index, questions=None) -> dict:
    global QUESTIONS
    questions = questions or QUESTIONS
    answerable = [q for q in questions if q["source"]]
    unanswerable = [q for q in questions if not q["source"]]
    hit1 = hit3 = 0
    rr = 0.0
    for q in answerable:
        ranked = [c.source for c, _ in index.search(q["q"], k=3)]
        if ranked[:1] == [q["source"]]:
            hit1 += 1
        if q["source"] in ranked:
            hit3 += 1
            rr += 1 / (ranked.index(q["source"]) + 1)
    wrongly_answered = [q["q"] for q in unanswerable if decide(q["q"], index)[0]]
    missed = [q["q"] for q in answerable if not decide(q["q"], index)[0]]
    n_a, n_u = len(answerable), len(unanswerable)
    return {
        "hit_at_1": round(hit1 / n_a, 3), "hit_at_3": round(hit3 / n_a, 3), "mrr": round(rr / n_a, 3),
        "unanswerable_wrongly_answered": f"{len(wrongly_answered)}/{n_u}",
        "answerable_wrongly_escalated": f"{len(missed)}/{n_a}",
        "wrongly_answered_examples": wrongly_answered,
    }


if __name__ == "__main__":
    baseline = run(BM25Index(CHUNKS, stem=False, synonyms=False))
    improved = run(BM25Index(CHUNKS, stem=True, synonyms=True))
    print("TUNING SET (used while building the gate)")
    print(json.dumps({"baseline": baseline, "improved": improved}, indent=2))
    print("HELD-OUT SET (never used for tuning)")
    print(json.dumps({"improved": run(BM25Index(CHUNKS, stem=True, synonyms=True), HOLDOUT)}, indent=2))
