"""Router: classifies questions into QuestionType using an LLM."""

import json
from pathlib import Path

from rageval.llm_client import LLMClient
from rageval.types import QuestionType

_PROMPT_PATH = Path(__file__).parents[3] / "prompts" / "router" / "v1.txt"
_MODEL = "claude-haiku-4-5-20251001"

_VALID_CATEGORIES: frozenset[str] = frozenset(qt.value for qt in QuestionType)


class Router:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()
        self._system: str = _PROMPT_PATH.read_text(encoding="utf-8")

    async def classify(self, question: str) -> QuestionType:
        response = await self._llm.complete(
            system=self._system,
            user=question,
            model=_MODEL,
            temperature=0.0,
        )
        raw: str = str(response["content"])
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return QuestionType.UNANSWERABLE
        if not isinstance(data, dict):
            return QuestionType.UNANSWERABLE
        category = str(data.get("category", "")).lower()
        if category not in _VALID_CATEGORIES:
            return QuestionType.UNANSWERABLE
        return QuestionType(category)
