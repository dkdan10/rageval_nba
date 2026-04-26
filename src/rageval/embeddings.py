"""Embedding helpers used by the optional live/vector demo path."""

from __future__ import annotations

import os
from typing import Protocol

import httpx

DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
_OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
_ESTIMATED_TEXT_EMBEDDING_3_SMALL_COST_PER_M_TOKENS = 0.02


class EmbeddingClient(Protocol):
    provider: str
    model: str
    dimensions: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class OpenAIEmbeddingClient:
    """Minimal OpenAI embeddings client using the project's existing httpx dependency."""

    provider = DEFAULT_EMBEDDING_PROVIDER

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")

        response = httpx.post(
            _OPENAI_EMBEDDINGS_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": texts,
                "dimensions": self.dimensions,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("OpenAI embeddings response missing data list")

        vectors: list[list[float]] = []
        for item in sorted(data, key=lambda row: int(row.get("index", 0))):
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("OpenAI embeddings response missing embedding")
            vector = [float(value) for value in embedding]
            if len(vector) != self.dimensions:
                raise RuntimeError(
                    f"Embedding dimension mismatch: got {len(vector)}, "
                    f"expected {self.dimensions}"
                )
            vectors.append(vector)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def estimate_embedding_cost_usd(
    token_count: int,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> float:
    """Return a conservative cost estimate for supported embedding models."""
    if model != DEFAULT_EMBEDDING_MODEL:
        return 0.0
    return (token_count * _ESTIMATED_TEXT_EMBEDDING_3_SMALL_COST_PER_M_TOKENS) / 1_000_000
