"""LLM-as-judge metrics (Milestones 4 & 5).

All LLM-backed judges use Anthropic tool-use for structured output. The raw
JSON-text fallback is preserved only so that existing mocked tests that return
`{"content": "<json>"}` continue to work.

Position-swap mitigation is implemented for CorrectnessJudge.
"""

import json
from pathlib import Path
from typing import Any

from rageval.llm_client import LLMClient
from rageval.types import MetricResult, QuestionType, RAGResponse, TestCase

_MODEL = "claude-haiku-4-5-20251001"
_JUDGES_DIR = Path(__file__).parents[3] / "prompts" / "judges"


# ---------------------------------------------------------------------------
# Tool schemas (one per judge)
# ---------------------------------------------------------------------------

_FAITHFULNESS_TOOL: dict[str, Any] = {
    "name": "record_faithfulness",
    "description": "Record a faithfulness judgment for an answer against its sources.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "faithful": {"type": "boolean"},
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", "faithful", "unsupported_claims"],
    },
}

_RELEVANCE_TOOL: dict[str, Any] = {
    "name": "record_relevance",
    "description": "Record whether the answer directly addresses the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "relevant": {"type": "boolean"},
            "irrelevant_parts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", "relevant", "irrelevant_parts"],
    },
}

_CORRECTNESS_TOOL: dict[str, Any] = {
    "name": "record_correctness",
    "description": "Record a 0-4 correctness score comparing candidate to reference.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "score": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
            "errors": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", "score", "errors"],
    },
}


def _load_prompt(judge: str) -> str:
    return (_JUDGES_DIR / judge / "v1.txt").read_text(encoding="utf-8")


def _build_sources(response: RAGResponse) -> str:
    parts: list[str] = []
    if response.sql_result is not None:
        rows_str = str(response.sql_result.rows)
        parts.append(f"SQL Result:\nQuery: {response.sql_result.query}\nRows: {rows_str}")
    for doc in response.retrieved_docs:
        parts.append(f"Document {doc.id}:\n{doc.content}")
    return "\n\n".join(parts) if parts else "(none)"


def _extract_tool_input(
    response: dict[str, Any], tool_name: str
) -> tuple[dict[str, Any] | None, str]:
    """Return (tool input dict, error string).

    First checks `tool_calls`. If none present, falls back to parsing JSON
    from the `content` text (for backward compatibility with mocked tests).
    """
    tool_calls: list[dict[str, Any]] = response.get("tool_calls") or []
    for call in tool_calls:
        if call.get("name") != tool_name:
            continue
        inp = call.get("input")
        if isinstance(inp, dict):
            return inp, ""

    content = response.get("content")
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError) as exc:
            return None, f"Invalid JSON from judge: {exc}"
        if isinstance(parsed, dict):
            return parsed, ""
        return None, "Judge output is not a JSON object"

    return None, "Judge did not return a tool call or parsable JSON content"


def _require_bool(
    data: dict[str, Any], field: str, metric_name: str, case_id: str
) -> tuple[bool, MetricResult | None]:
    if field not in data:
        return False, MetricResult(
            metric_name=metric_name,
            case_id=case_id,
            value=0.0,
            error=f"Judge output missing required field: '{field}'",
        )
    val = data[field]
    if not isinstance(val, bool):
        return False, MetricResult(
            metric_name=metric_name,
            case_id=case_id,
            value=0.0,
            error=f"Field '{field}' must be a boolean, got {type(val).__name__}: {val!r}",
        )
    return val, None


# ---------------------------------------------------------------------------
# FaithfulnessJudge
# ---------------------------------------------------------------------------


def _parse_faithfulness(case_id: str, response: dict[str, Any]) -> MetricResult:
    data, err = _extract_tool_input(response, _FAITHFULNESS_TOOL["name"])
    if data is None:
        return MetricResult(
            metric_name="faithfulness",
            case_id=case_id,
            value=0.0,
            error=err or "No tool input found",
        )

    faithful, berr = _require_bool(data, "faithful", "faithfulness", case_id)
    if berr is not None:
        return berr

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
        details={"reasoning": reasoning, "unsupported_claims": unsupported_claims},
    )


class FaithfulnessJudge:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm: LLMClient = llm or LLMClient()
        self._system: str = _load_prompt("faithfulness")

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
            tools=[_FAITHFULNESS_TOOL],
            tool_choice={"type": "tool", "name": _FAITHFULNESS_TOOL["name"]},
        )
        return _parse_faithfulness(case.id, result)


# ---------------------------------------------------------------------------
# RelevanceJudge
# ---------------------------------------------------------------------------


def _parse_relevance(case_id: str, response: dict[str, Any]) -> MetricResult:
    data, err = _extract_tool_input(response, _RELEVANCE_TOOL["name"])
    if data is None:
        return MetricResult(
            metric_name="relevance",
            case_id=case_id,
            value=0.0,
            error=err or "No tool input found",
        )

    relevant, berr = _require_bool(data, "relevant", "relevance", case_id)
    if berr is not None:
        return berr

    reasoning: str = str(data.get("reasoning", ""))
    raw_parts: Any = data.get("irrelevant_parts", [])
    if isinstance(raw_parts, list):
        irrelevant_parts: list[str] = [str(p) for p in raw_parts]
    else:
        irrelevant_parts = [str(raw_parts)]

    return MetricResult(
        metric_name="relevance",
        case_id=case_id,
        value=1.0 if relevant else 0.0,
        details={"reasoning": reasoning, "irrelevant_parts": irrelevant_parts},
    )


class RelevanceJudge:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm: LLMClient = llm or LLMClient()
        self._system: str = _load_prompt("relevance")

    async def evaluate(self, case: TestCase, response: RAGResponse) -> MetricResult:
        user = f"Question: {case.question}\n\nAnswer: {response.answer}"
        result = await self._llm.complete(
            system=self._system,
            user=user,
            model=_MODEL,
            temperature=0.0,
            tools=[_RELEVANCE_TOOL],
            tool_choice={"type": "tool", "name": _RELEVANCE_TOOL["name"]},
        )
        return _parse_relevance(case.id, result)


# ---------------------------------------------------------------------------
# CorrectnessJudge
# ---------------------------------------------------------------------------


def _parse_correctness_pass(
    case_id: str, response: dict[str, Any], pass_name: str
) -> tuple[int | None, str, MetricResult | None]:
    data, err = _extract_tool_input(response, _CORRECTNESS_TOOL["name"])
    if data is None:
        return None, "", MetricResult(
            metric_name="correctness",
            case_id=case_id,
            value=0.0,
            error=err or f"No tool input found ({pass_name})",
        )

    if "score" not in data:
        return None, "", MetricResult(
            metric_name="correctness",
            case_id=case_id,
            value=0.0,
            error=f"Judge output missing required field: 'score' ({pass_name})",
        )

    raw_score = data["score"]
    if isinstance(raw_score, bool) or not isinstance(raw_score, int) or not (0 <= raw_score <= 4):
        return None, "", MetricResult(
            metric_name="correctness",
            case_id=case_id,
            value=0.0,
            error=f"Field 'score' must be an integer 0-4, got {raw_score!r} ({pass_name})",
        )

    return raw_score, str(data.get("reasoning", "")), None


_DISAGREEMENT_THRESHOLD = 2


class CorrectnessJudge:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm: LLMClient = llm or LLMClient()
        self._system: str = _load_prompt("correctness")

    async def evaluate(self, case: TestCase, response: RAGResponse) -> MetricResult:
        if case.expected_answer is None:
            return MetricResult(
                metric_name="correctness",
                case_id=case.id,
                value=0.0,
                error="No expected answer available",
            )

        forward_user = (
            f"Question: {case.question}\n\n"
            f"Candidate Answer: {response.answer}\n\n"
            f"Reference Answer: {case.expected_answer}"
        )
        swapped_user = (
            f"Question: {case.question}\n\n"
            f"Candidate Answer: {case.expected_answer}\n\n"
            f"Reference Answer: {response.answer}\n\n"
            "Note: You are checking semantic equivalence. "
            "Judge the candidate against the reference regardless of order."
        )

        tools = [_CORRECTNESS_TOOL]
        tool_choice = {"type": "tool", "name": _CORRECTNESS_TOOL["name"]}

        forward_raw = await self._llm.complete(
            system=self._system,
            user=forward_user,
            model=_MODEL,
            temperature=0.0,
            tools=tools,
            tool_choice=tool_choice,
        )
        swapped_raw = await self._llm.complete(
            system=self._system,
            user=swapped_user,
            model=_MODEL,
            temperature=0.0,
            tools=tools,
            tool_choice=tool_choice,
        )

        fwd_score, fwd_reasoning, fwd_err = _parse_correctness_pass(
            case.id, forward_raw, "forward"
        )
        if fwd_err is not None:
            return fwd_err

        swp_score, swp_reasoning, swp_err = _parse_correctness_pass(
            case.id, swapped_raw, "swapped"
        )
        if swp_err is not None:
            return swp_err

        assert fwd_score is not None and swp_score is not None
        final_value = (fwd_score + swp_score) / 2.0 / 4.0
        disagreement = abs(fwd_score - swp_score)

        return MetricResult(
            metric_name="correctness",
            case_id=case.id,
            value=final_value,
            details={
                "forward_score": fwd_score,
                "swapped_score": swp_score,
                "disagreement": disagreement,
                "disagreement_flag": disagreement >= _DISAGREEMENT_THRESHOLD,
                "reasoning_forward": fwd_reasoning,
                "reasoning_swapped": swp_reasoning,
            },
        )


# ---------------------------------------------------------------------------
# RoutingJudge
#
# Deterministic by design — routing accuracy is a direct comparison between
# the system's `routing_decision` and the test case's `question_type`. No LLM
# call is involved. The file `prompts/judges/routing/v1.txt` is kept as a
# placeholder for a future LLM-assisted variant and is intentionally unused.
# ---------------------------------------------------------------------------


class RoutingJudge:
    async def evaluate(self, case: TestCase, response: RAGResponse) -> MetricResult:
        expected: QuestionType = case.question_type
        actual: QuestionType | None = response.routing_decision

        if actual is None:
            return MetricResult(
                metric_name="routing_accuracy",
                case_id=case.id,
                value=0.0,
                details={"expected_route": expected.value, "actual_route": None},
            )

        return MetricResult(
            metric_name="routing_accuracy",
            case_id=case.id,
            value=1.0 if actual == expected else 0.0,
            details={"expected_route": expected.value, "actual_route": actual.value},
        )
