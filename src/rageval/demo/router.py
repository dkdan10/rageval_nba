"""Router: classifies questions into QuestionType using an LLM via tool-use."""

from pathlib import Path
from typing import Any

from rageval.llm_client import LLMClient
from rageval.types import QuestionType

_PROMPT_PATH = Path(__file__).parents[3] / "prompts" / "router" / "v1.txt"
_MODEL = "claude-haiku-4-5-20251001"
_TOOL_NAME = "classify_question"

_VALID_CATEGORIES: frozenset[str] = frozenset(qt.value for qt in QuestionType)

_ROUTER_TOOL: dict[str, Any] = {
    "name": _TOOL_NAME,
    "description": (
        "Record the routing classification for an NBA question. "
        "Always call this exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "One short sentence explaining the classification.",
            },
            "category": {
                "type": "string",
                "enum": sorted(_VALID_CATEGORIES),
                "description": "The routing category.",
            },
        },
        "required": ["reasoning", "category"],
    },
}


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
            tools=[_ROUTER_TOOL],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
        )
        return _parse_router_result(response)


def _parse_router_result(response: dict[str, Any]) -> QuestionType:
    """Extract the category from a tool-use response. Fall back to UNANSWERABLE."""
    tool_calls: list[dict[str, Any]] = response.get("tool_calls") or []
    for call in tool_calls:
        if call.get("name") != _TOOL_NAME:
            continue
        inp = call.get("input")
        if not isinstance(inp, dict):
            continue
        category = str(inp.get("category", "")).lower()
        if category in _VALID_CATEGORIES:
            return QuestionType(category)
    return QuestionType.UNANSWERABLE
