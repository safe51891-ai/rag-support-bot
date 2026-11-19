"""FastAPI service. Run: uvicorn app:app --reload   then POST /ask {"question": "..."}"""
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from ragbot.answer import answer
from ragbot.index import BM25Index, load_chunks

INDEX = BM25Index(load_chunks(Path(__file__).parent / "data" / "docs"))
app = FastAPI(title="Support RAG Bot")


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Question):
    result = answer(body.question, INDEX)
    return {"answer": result.text, "citations": result.citations, "escalated": result.escalated}
