import json
from pathlib import Path
from typing import Any

from rageval.llm_client import LLMClient
from rageval.types import MetricResult, RAGResponse, TestCase

_MODEL = "claude-haiku-4-5-20251001"
_PROMPT_PATH = Path(__file__).parents[3] / "prompts" / "judges" / "faithfulness" / "v1.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_sources(response: RAGResponse) -> str:
    parts: list[str] = []
    if response.sql_result is not None:
        rows_str = str(response.sql_result.rows)
        parts.append(f"SQL Result:\nQuery: {response.sql_result.query}\nRows: {rows_str}")
    for doc in response.retrieved_docs:
        parts.append(f"Document {doc.id}:\n{doc.content}")
    return "\n\n".join(parts) if parts else "(none)"


def _parse_judge_output(case_id: str, content: str) -> MetricResult:
    try:
        data: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        return MetricResult(
            metric_name="faithfulness",
            case_id=case_id,
            value=0.0,
            error=f"Invalid JSON from judge: {exc}",
        )

    if not isinstance(data, dict):
        return MetricResult(
            metric_name="faithfulness",
            case_id=case_id,
            value=0.0,
            error="Judge output is not a JSON object",
        )

    if "faithful" not in data:
        return MetricResult(
            metric_name="faithfulness",
            case_id=case_id,
            value=0.0,
            error="Judge output missing required field: 'faithful'",
        )

    faithful: bool = bool(data["faithful"])
    reasoning: str = str(data.get("reasoning", ""))

    raw_claims: Any = data.get("unsupported_claims", [])
    if isinstance(raw_claims, list):
        unsupported_claims: list[str] = [str(c) for c in raw_claims]
    else:
        unsupported_claims = [str(raw_claims)]

    return MetricResult(
        metric_name="faithfulness",
        case_id=case_id,
        value=1.0 if faithful else 0.0,
        details={
            "reasoning": reasoning,
            "unsupported_claims": unsupported_claims,
        },
    )


class FaithfulnessJudge:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm: LLMClient = llm or LLMClient()
        self._system: str = _load_prompt()

    async def evaluate(self, case: TestCase, response: RAGResponse) -> MetricResult:
        sources = _build_sources(response)
        user = (
            f"Question: {case.question}\n\n"
            f"Sources:\n{sources}\n\n"
            f"Answer: {response.answer}"
        )
        result = await self._llm.complete(
            system=self._system,
            user=user,
            model=_MODEL,
            temperature=0.0,
        )
        content: str = str(result.get("content", ""))
        return _parse_judge_output(case.id, content)
