import sqlite3
from pathlib import Path

import pytest

from rageval.demo.rag_agent import RAGAgent
from rageval.sqlite_vec import load_sqlite_vec, serialize_float32
from rageval.types import Document
from scripts.build_stats_db import _init_schema


def _corpus_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "corpus.db"
    con = sqlite3.connect(db_path)
    try:
        _init_schema(con)
        con.execute(
            """
            INSERT INTO articles(
                article_id, title, source, author, url, publish_date,
                full_text, word_count, ingested_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ctg-four-factors",
                "Four Factors",
                "fixture",
                None,
                "https://example.test/four-factors",
                None,
                (
                    "Effective field goal percentage, turnover rate, "
                    "offensive rebounding rate, and free throw rate."
                ),
                12,
                "2026-04-25T00:00:00Z",
            ),
        )
        con.execute(
            """
            INSERT INTO articles(
                article_id, title, source, author, url, publish_date,
                full_text, word_count, ingested_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rim-protection",
                "Rim Protection",
                "fixture",
                None,
                "https://example.test/rim",
                None,
                (
                    "Rim protection includes deterrence and opponent field "
                    "goal percentage near the basket."
                ),
                11,
                "2026-04-25T00:00:00Z",
            ),
        )
        con.executemany(
            """
            INSERT INTO article_chunks(chunk_id, article_id, chunk_index, content, token_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    "ctg-four-factors#0",
                    "ctg-four-factors",
                    0,
                    (
                        "The four factors are effective field goal percentage, "
                        "turnover rate, offensive rebounding rate, and free throw rate."
                    ),
                    16,
                ),
                (
                    "rim-protection#0",
                    "rim-protection",
                    0,
                    (
                        "Rim protection metrics capture deterrence and opponent "
                        "shooting at the basket."
                    ),
                    11,
                ),
            ],
        )
        con.commit()
    finally:
        con.close()
    return db_path


def test_retrieve_relevant_chunks(tmp_path: Path) -> None:
    agent = RAGAgent(db_path=_corpus_db(tmp_path))

    docs = agent.retrieve("What are the four factors?", k=5)

    assert docs
    assert docs[0].id == "ctg-four-factors#0"
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["article_id"] == "ctg-four-factors"


def test_retrieve_respects_top_k(tmp_path: Path) -> None:
    agent = RAGAgent(db_path=_corpus_db(tmp_path))

    docs = agent.retrieve("rate protection factors percentage", k=1)

    assert len(docs) == 1


def test_retrieve_empty_corpus(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    con = sqlite3.connect(db_path)
    try:
        _init_schema(con)
        con.commit()
    finally:
        con.close()

    assert RAGAgent(db_path=db_path).retrieve("four factors") == []


def test_retrieve_missing_db_gracefully_returns_empty(tmp_path: Path) -> None:
    assert RAGAgent(db_path=tmp_path / "missing.db").retrieve("four factors") == []


def test_retrieve_missing_tables_gracefully_returns_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "no_tables.db"
    sqlite3.connect(db_path).close()

    assert RAGAgent(db_path=db_path).retrieve("four factors") == []


def test_retrieve_non_positive_k_returns_empty(tmp_path: Path) -> None:
    assert RAGAgent(db_path=_corpus_db(tmp_path)).retrieve("four factors", k=0) == []


def test_vector_mode_retrieves_embeddings_when_sqlite_vec_available(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    db_path = _corpus_db(tmp_path)
    con = sqlite3.connect(db_path)
    try:
        loaded, reason = load_sqlite_vec(con)
        if not loaded:
            pytest.skip(reason or "sqlite-vec unavailable")
        con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings "
            "USING vec0(chunk_id TEXT PRIMARY KEY, embedding FLOAT[1024])"
        )
        four_vector = [1.0] + [0.0] * 1023
        rim_vector = [0.0, 1.0] + [0.0] * 1022
        con.execute(
            "INSERT OR REPLACE INTO chunk_embeddings(chunk_id, embedding) VALUES (?, ?)",
            ("ctg-four-factors#0", serialize_float32(four_vector)),
        )
        con.execute(
            "INSERT OR REPLACE INTO chunk_embeddings(chunk_id, embedding) VALUES (?, ?)",
            ("rim-protection#0", serialize_float32(rim_vector)),
        )
        con.commit()
    finally:
        con.close()

    class FakeEmbeddingClient:
        provider = "fake"
        model = "fake-embedding"
        dimensions = 1024

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            assert texts
            return [[1.0] + [0.0] * 1023 for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            assert "four" in text
            return [1.0] + [0.0] * 1023

    docs = RAGAgent(
        db_path=db_path,
        mode="vector",
        embedding_client=FakeEmbeddingClient(),
    ).retrieve("four factors", k=1)

    assert [doc.id for doc in docs] == ["ctg-four-factors#0"]
    assert docs[0].metadata["retrieval_mode"] == "vector"


def test_vector_mode_falls_back_to_lexical_when_vector_unavailable(tmp_path: Path) -> None:
    docs = RAGAgent(db_path=_corpus_db(tmp_path), mode="vector").retrieve(
        "four factors",
        k=1,
    )

    assert docs
    assert docs[0].id == "ctg-four-factors#0"
    assert docs[0].metadata["retrieval_mode"] == "lexical_fallback"


def test_retrieve_rejects_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="RAGAgent mode"):
        RAGAgent(db_path=_corpus_db(tmp_path), mode="strange")
