"""Answer synthesis for the NBA demo system."""

from __future__ import annotations

import json
from pathlib import Path

from rageval.llm_client import LLMClient
from rageval.types import Document, SQLResult

_PROMPT_PATH = Path(__file__).parents[3] / "prompts" / "synthesizer" / "v1.txt"
_MODEL = "claude-haiku-4-5-20251001"


class Synthesizer:
    """Generate final answers from SQL rows and retrieved documents.

    Without an injected LLM client, this class uses a deterministic evidence
    summary so tests and local demos do not require network calls.
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm
        self._system = _PROMPT_PATH.read_text(encoding="utf-8")

    async def synthesize(
        self,
        question: str,
        sql_result: SQLResult | None = None,
        docs: list[Document] | None = None,
    ) -> str:
        docs = docs or []
        if self._llm is None:
            return _deterministic_answer(sql_result=sql_result, docs=docs)

        result = await self._llm.complete(
            system=self._system,
            user=_build_user_prompt(question, sql_result, docs),
            model=_MODEL,
            temperature=0.0,
        )
        content = str(result.get("content", "")).strip()
        if content:
            return content
        return _deterministic_answer(sql_result=sql_result, docs=docs)


def _build_user_prompt(
    question: str,
    sql_result: SQLResult | None,
    docs: list[Document],
) -> str:
    sql_json = "null" if sql_result is None else json.dumps(sql_result.model_dump())
    chunks = [
        {"id": doc.id, "content": doc.content, "metadata": doc.metadata}
        for doc in docs
    ]
    return (
        f"QUESTION: {question}\n\n"
        f"SQL RESULTS: {sql_json}\n\n"
        f"RETRIEVED ARTICLES: {json.dumps(chunks)}"
    )


def _deterministic_answer(
    sql_result: SQLResult | None,
    docs: list[Document],
) -> str:
    parts: list[str] = []
    if sql_result is not None and sql_result.error is None and sql_result.rows:
        parts.append(f"SQL evidence: {sql_result.rows} [sql]")
    elif sql_result is not None and sql_result.error is not None:
        parts.append(f"SQL evidence unavailable: {sql_result.error} [sql]")

    for doc in docs:
        snippet = " ".join(doc.content.split())
        if len(snippet) > 240:
            snippet = f"{snippet[:237]}..."
        parts.append(f"{snippet} [article:{doc.id}]")

    if not parts:
        return "The provided sources are insufficient to answer this question."
    return " ".join(parts)
