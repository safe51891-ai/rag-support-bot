"""Chunk documents and retrieve with BM25 (pure Python, no model download needed).

Swap BM25Index for a vector store (ChromaDB, FAISS, Pinecone) by keeping the same `search` signature.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = set("a an the is are was to of in on for and or it i my be by with at this that can how".split())


@dataclass
class Chunk:
    source: str
    section: str
    text: str


# Customers say "money back" where the docs say "refund", and "parcel" where the docs say "shipping".
SYNONYMS = {
    "money back": "refund",
    "get my money": "refund",
    "parcel": "shipping delivery",
    "package": "shipping delivery order",
    "locked out": "password reset",
    "log in": "password sign in",
    "charged twice": "duplicate charge payment",
    "stop my": "cancel",
    "monthly": "subscription",
    "cash": "cash payment",
    "mastercard": "payment method card",
    "invoice": "invoice download",
    "how long": "processing time days",
    "reach my card": "payment method received",
}


# Words that appear in almost every question and say nothing about the topic.
GENERIC_WORDS = set("you your do does how can i my what is it if after before there any offer where who this "
                    "on about we our will be have get use want need me a an the to of".split())


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _tokens(text: str, stem: bool = False) -> list[str]:
    tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]
    return [_stem(t) for t in tokens] if stem else tokens


def expand_query(query: str) -> str:
    lowered = query.lower()
    extra = [replacement for phrase, replacement in SYNONYMS.items() if phrase in lowered]
    return query + " " + " ".join(extra)


def chunk_markdown(path: Path) -> list[Chunk]:
    """Split on '## ' headings so each chunk is one topic, with its section name kept as a citation."""
    source = path.stem
    chunks, section, buf = [], "Overview", []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if buf:
                chunks.append(Chunk(source, section, "\n".join(buf).strip()))
            section, buf = line[3:].strip(), []
        elif line.startswith("# "):
            continue
        else:
            buf.append(line)
    if buf and "\n".join(buf).strip():
        chunks.append(Chunk(source, section, "\n".join(buf).strip()))
    return chunks


def load_chunks(folder: Path) -> list[Chunk]:
    out = []
    for path in sorted(folder.glob("*.md")):
        out.extend(chunk_markdown(path))
    return out


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75,
                 stem: bool = True, synonyms: bool = True):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.stem = stem
        self.synonyms = synonyms
        self.docs = [Counter(_tokens(c.section + " " + c.text, stem)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def coverage(self, query: str, chunk: Chunk) -> float:
        """Share of the question's content words that appear in this chunk. Generic words do not count."""
        if self.synonyms:
            query = expand_query(query)  # same expansion as search, so synonyms count toward coverage
        q_terms = {t for t in _tokens(query, stem=True) if t not in GENERIC_WORDS}
        if not q_terms:
            return 0.0
        doc_terms = set(_tokens(chunk.section + " " + chunk.text, stem=True))
        return len(q_terms & doc_terms) / len(q_terms)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if self.synonyms:
            query = expand_query(query)
        q = _tokens(query, self.stem)
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for t in q:
                if t not in doc:
                    continue
                tf = doc[t]
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                score += self.idf[t] * tf * (self.k1 + 1) / denom
            if score > 0:
                scored.append((self.chunks[i], round(score, 3)))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
