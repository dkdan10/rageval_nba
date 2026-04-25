"""Deterministic article retrieval for the NBA demo system."""

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from rageval.types import Document

_DB_PATH = Path(__file__).parents[3] / "data" / "nba.db"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "why",
    "with",
}


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.casefold()) if token not in _STOPWORDS}


def _score(query_tokens: set[str], content: str, metadata_text: str) -> float:
    if not query_tokens:
        return 0.0
    content_tokens = _tokens(content)
    metadata_tokens = _tokens(metadata_text)
    content_hits = len(query_tokens & content_tokens)
    metadata_hits = len(query_tokens & metadata_tokens)
    if content_hits == 0 and metadata_hits == 0:
        return 0.0
    length_penalty = math.log2(max(len(content_tokens), 2))
    return (content_hits + 0.5 * metadata_hits) / length_penalty


class RAGAgent:
    """Retrieve article chunks with a deterministic lexical fallback.

    This is not production vector retrieval. It intentionally avoids paid
    embedding calls so unit tests and local demos can run offline. A future
    corpus build can populate ``chunk_embeddings`` and swap in sqlite-vec search.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DB_PATH

    def retrieve(self, question: str, k: int = 5) -> list[Document]:
        if k <= 0 or not self._db_path.exists():
            return []

        try:
            con = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            try:
                rows = con.execute(
                    """
                    SELECT
                        c.chunk_id,
                        c.article_id,
                        c.chunk_index,
                        c.content,
                        c.token_count,
                        a.title,
                        a.source,
                        a.url,
                        a.author,
                        a.publish_date
                    FROM article_chunks c
                    JOIN articles a ON a.article_id = c.article_id
                    """
                ).fetchall()
            finally:
                con.close()
        except sqlite3.Error:
            return []

        query_tokens = _tokens(question)
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            metadata_text = f"{row['title']} {row['source']} {row['article_id']}"
            score = _score(query_tokens, str(row["content"]), metadata_text)
            if score > 0.0:
                scored.append((score, row))

        scored.sort(key=lambda item: (-item[0], str(item[1]["chunk_id"])))
        return [_row_to_document(row, score) for score, row in scored[:k]]


def _row_to_document(row: sqlite3.Row, score: float) -> Document:
    metadata: dict[str, Any] = {
        "article_id": row["article_id"],
        "chunk_index": row["chunk_index"],
        "token_count": row["token_count"],
        "title": row["title"],
        "source": row["source"],
        "url": row["url"],
        "author": row["author"],
        "publish_date": row["publish_date"],
        "score": score,
    }
    return Document(id=str(row["chunk_id"]), content=str(row["content"]), metadata=metadata)
