"""Run judge calibration against hand-labeled fixtures.

Usage:
    uv run python scripts/calibrate_judge.py [all|faithfulness|relevance|correctness|routing]

This script drives each judge against its calibration fixture (10 cases per
judge) and prints agreement rates. By default it uses ``LLMClient()`` which
requires ``ANTHROPIC_API_KEY`` in the environment. Responses are cached in
``.rageval_cache/`` so repeated runs are fast and cheap.

Exits non-zero if any judge falls below the ``--threshold`` (default 0.8).

For offline / CI usage, ``calibrate_*`` functions accept a pre-configured
``LLMClient`` (or any mock) via the ``llm`` kwarg — see ``tests/test_calibrate.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from rageval.llm_client import LLMClient
from rageval.metrics.judge import (
    CorrectnessJudge,
    FaithfulnessJudge,
    RelevanceJudge,
    RoutingJudge,
)
from rageval.types import Document, QuestionType, RAGResponse, TestCase

_FIXTURES_DIR = Path(__file__).parents[1] / "tests" / "fixtures"
_DEFAULT_THRESHOLD = 0.8
_ALL_JUDGES = frozenset({"faithfulness", "relevance", "correctness", "routing"})


def load_fixture(judge: str) -> list[dict[str, Any]]:
    """Load calibration cases for *judge* from its YAML fixture file."""
    path = _FIXTURES_DIR / f"{judge}_calibration.yaml"
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(raw["cases"])


def binary_agreement(verdicts: list[bool], labels: list[bool]) -> float:
    """Fraction of (verdict, label) pairs that agree."""
    if not verdicts:
        return 0.0
    return sum(1 for v, lbl in zip(verdicts, labels, strict=True) if v == lbl) / len(verdicts)


def correctness_agreement(scores: list[float], human_scores: list[int]) -> float:
    """Fraction of cases where the judge's rounded 0-4 score matches the human score."""
    if not scores:
        return 0.0
    return sum(
        1 for s, h in zip(scores, human_scores, strict=True) if round(s * 4.0) == h
    ) / len(scores)


async def calibrate_faithfulness(llm: LLMClient | None = None) -> float:
    """Return faithfulness judge agreement rate against human labels."""
    cases = load_fixture("faithfulness")
    judge = FaithfulnessJudge(llm=llm)
    verdicts: list[bool] = []
    labels: list[bool] = []
    for c in cases:
        tc = TestCase(id=c["id"], question=str(c["question"]), question_type=QuestionType.FACTUAL)
        source_doc = Document(id=f"source-{c['id']}", content=str(c["sources"]))
        resp = RAGResponse(answer=str(c["answer"]), retrieved_docs=[source_doc])
        result = await judge.evaluate(tc, resp)
        verdicts.append(result.value == 1.0)
        labels.append(bool(c["human_label"]))
    return binary_agreement(verdicts, labels)


async def calibrate_relevance(llm: LLMClient | None = None) -> float:
    """Return relevance judge agreement rate against human labels."""
    cases = load_fixture("relevance")
    judge = RelevanceJudge(llm=llm)
    verdicts: list[bool] = []
    labels: list[bool] = []
    for c in cases:
        tc = TestCase(
            id=c["id"], question=str(c["question"]), question_type=QuestionType.ANALYTICAL
        )
        resp = RAGResponse(answer=str(c["answer"]))
        result = await judge.evaluate(tc, resp)
        verdicts.append(result.value == 1.0)
        labels.append(bool(c["human_label"]))
    return binary_agreement(verdicts, labels)


async def calibrate_correctness(llm: LLMClient | None = None) -> float:
    """Return correctness judge agreement rate against human scores."""
    cases = load_fixture("correctness")
    judge = CorrectnessJudge(llm=llm)
    scores: list[float] = []
    human_scores: list[int] = []
    for c in cases:
        tc = TestCase(
            id=c["id"],
            question=str(c["question"]),
            question_type=QuestionType.FACTUAL,
            expected_answer=str(c["expected_answer"]),
        )
        resp = RAGResponse(answer=str(c["answer"]))
        result = await judge.evaluate(tc, resp)
        scores.append(result.value)
        human_scores.append(int(c["human_score"]))
    return correctness_agreement(scores, human_scores)


async def calibrate_routing() -> float:
    """Return routing judge agreement rate (deterministic; should be 100%)."""
    cases = load_fixture("routing")
    judge = RoutingJudge()
    verdicts: list[bool] = []
    labels: list[bool] = []
    for c in cases:
        expected = QuestionType(str(c["question_type"]))
        actual = QuestionType(str(c["routing_decision"]))
        tc = TestCase(id=c["id"], question=str(c["question"]), question_type=expected)
        resp = RAGResponse(answer="", routing_decision=actual)
        result = await judge.evaluate(tc, resp)
        verdicts.append(result.value == 1.0)
        labels.append(bool(c["correct"]))
    return binary_agreement(verdicts, labels)


async def run(judges: list[str], threshold: float = _DEFAULT_THRESHOLD) -> bool:
    """Run calibration for *judges*. Returns True if all meet *threshold*.

    LLM-backed judges share one ``LLMClient`` so the on-disk cache is reused
    across runs. If ``ANTHROPIC_API_KEY`` is not set, warns and runs only the
    deterministic routing judge.
    """
    to_run = _ALL_JUDGES if "all" in judges else _ALL_JUDGES & set(judges)

    need_llm = bool(to_run & {"faithfulness", "relevance", "correctness"})
    llm: LLMClient | None = None
    if need_llm:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "WARNING: ANTHROPIC_API_KEY is not set. "
                "Skipping LLM-backed judges (faithfulness, relevance, correctness).",
                file=sys.stderr,
            )
            to_run = to_run - {"faithfulness", "relevance", "correctness"}
        else:
            llm = LLMClient()

    results: dict[str, float] = {}
    if "faithfulness" in to_run:
        results["faithfulness"] = await calibrate_faithfulness(llm=llm)
    if "relevance" in to_run:
        results["relevance"] = await calibrate_relevance(llm=llm)
    if "correctness" in to_run:
        results["correctness"] = await calibrate_correctness(llm=llm)
    if "routing" in to_run:
        results["routing"] = await calibrate_routing()

    if not results:
        print(f"No valid judge names given. Choose from: {sorted(_ALL_JUDGES)}")
        return False

    for name, rate in sorted(results.items()):
        n = len(load_fixture(name))
        hits = round(rate * n)
        status = "PASS" if rate >= threshold else "FAIL"
        print(f"[{status}] {name}: {rate:.0%} ({hits}/{n})")

    if llm is not None:
        print(
            f"Total LLM cost: ${llm.total_cost_usd:.4f} "
            f"({llm.total_input_tokens} in / {llm.total_output_tokens} out tokens)",
            file=sys.stderr,
        )

    return all(r >= threshold for r in results.values())


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run judge calibration against fixtures.")
    p.add_argument(
        "judges",
        nargs="*",
        default=["all"],
        help="Which judges to calibrate. Default: all.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        help=f"Agreement threshold (default {_DEFAULT_THRESHOLD}).",
    )
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    ok = asyncio.run(run(args.judges, threshold=args.threshold))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
