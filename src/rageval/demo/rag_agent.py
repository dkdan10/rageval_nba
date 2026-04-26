"""Article retrieval for the NBA demo system.

The default ``offline`` mode uses deterministic lexical scoring. The optional
``vector``/``auto`` modes use sqlite-vec when chunk embeddings and an embedding
client are available, then fall back to lexical retrieval when they are not.
"""

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from rageval.embeddings import EmbeddingClient, OpenAIEmbeddingClient
from rageval.sqlite_vec import load_sqlite_vec, serialize_float32
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
    """Retrieve article chunks with lexical or sqlite-vec search.

    ``offline`` never calls an embedding provider. ``vector`` and ``auto`` try
    sqlite-vec first, then return lexical results if vector search cannot run.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        mode: str = "offline",
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        if mode not in {"offline", "vector", "auto"}:
            raise ValueError("RAGAgent mode must be one of: offline, vector, auto")
        self._db_path = db_path or _DB_PATH
        self.mode = mode
        self._embedding_client = embedding_client

    def retrieve(self, question: str, k: int = 5) -> list[Document]:
        if k <= 0 or not self._db_path.exists():
            return []
        if self.mode in {"vector", "auto"}:
            docs = self._retrieve_vector(question, k)
            if docs:
                return docs
            if self.mode == "vector":
                # Keep the demo resilient, but mark the fallback in returned metadata.
                lexical = self._retrieve_lexical(question, k)
                for doc in lexical:
                    doc.metadata["retrieval_mode"] = "lexical_fallback"
                return lexical
        return self._retrieve_lexical(question, k)

    def _retrieve_lexical(self, question: str, k: int) -> list[Document]:
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
        docs = [_row_to_document(row, score) for score, row in scored[:k]]
        for doc in docs:
            doc.metadata["retrieval_mode"] = "lexical"
        return docs

    def _retrieve_vector(self, question: str, k: int) -> list[Document]:
        client = self._embedding_client
        if client is None:
            try:
                client = OpenAIEmbeddingClient()
            except Exception:
                return []

        try:
            query_embedding = client.embed_query(question)
        except Exception:
            return []

        try:
            con = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            loaded, _reason = load_sqlite_vec(con)
            if not loaded:
                return []
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
                        a.publish_date,
                        e.distance
                    FROM chunk_embeddings e
                    JOIN article_chunks c ON c.chunk_id = e.chunk_id
                    JOIN articles a ON a.article_id = c.article_id
                    WHERE e.embedding MATCH ? AND k = ?
                    ORDER BY e.distance
                    """,
                    (serialize_float32(query_embedding), k),
                ).fetchall()
            finally:
                con.close()
        except sqlite3.Error:
            return []

        docs = [_row_to_document(row, 1.0 / (1.0 + float(row["distance"]))) for row in rows]
        for doc, row in zip(docs, rows, strict=True):
            doc.metadata["retrieval_mode"] = "vector"
            doc.metadata["distance"] = float(row["distance"])
            doc.metadata["embedding_model"] = getattr(client, "model", None)
        return docs


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
