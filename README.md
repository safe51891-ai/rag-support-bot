# Support RAG Chatbot with Citations

Answers customer questions from help-center documents. Every answer cites its source section. When the documents do not support an answer, the bot says so and escalates to a human.

## Pipeline
1. **Chunk** (`ragbot/index.py`): each `##` section becomes one chunk, keeping its title as a citation.
2. **Retrieve**: BM25 ranking, pure Python, runs offline. Replace `BM25Index` with ChromaDB, FAISS, or Pinecone and keep the same `search` signature.
3. **Gate**: if the top score is below `MIN_SCORE`, return the fallback and escalate. This stops the bot from guessing.
4. **Answer** (`ragbot/answer.py`): offline mode quotes the most relevant sentence with its citation. With `ANTHROPIC_API_KEY`, Claude answers from numbered sources only and replies `NOT_FOUND` when they lack the answer.
5. **Evaluate** (`ragbot/evaluate.py`): hit@k and MRR over labeled questions. A test enforces a minimum score, so retrieval regressions fail CI.

## Measured results (see eval/REPORT.md)
- Hit@3 on 19 answerable tuning questions: **0.895 to 0.947** after stemming and synonyms.
- Wrong answers to unanswerable questions: **1 of 11** (a known gap: "delivery address" matches the shipping doc).
- Held-out sanity check (10 questions): 0 wrong answers, 0 unneeded escalations. Ten questions is too few to rely on.

## Run
```bash
pip install -r requirements.txt
pytest -q                               # chunking, citations, fallback, retrieval quality
uvicorn app:app --reload                # POST http://127.0.0.1:8000/ask  {"question": "..."}
```

## Next steps for production
- Replace BM25 with embeddings in a vector database, and compare hit@k against BM25 on the same labeled set.
- Add a reranker, and log every question with its retrieved chunks to find gaps in the docs.
- Add an answer-faithfulness check (does each claim appear in a cited source?).
