"""Tests for the minimal CLI stub (src/rageval/cli.py)."""

import io
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from rageval import cli
from rageval.cli import main
from rageval.demo.system import HybridRAGSystem
from rageval.types import (
    Document,
    QuestionType,
    RAGResponse,
    SQLResult,
    TestCase,
    TestSuite,
)
from scripts.build_stats_db import _init_schema


@pytest.fixture(autouse=True)
def _isolate_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *_args, **_kwargs: None)


def _db_with_corpus(tmp_path: Path, *, chunks: bool = True) -> Path:
    db_path = tmp_path / "nba.db"
    con = sqlite3.connect(db_path)
    try:
        _init_schema(con)
        if chunks:
            con.execute(
                """
                INSERT INTO articles(
                    article_id, title, source, author, url, publish_date,
                    full_text, word_count, ingested_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "doc",
                    "Doc",
                    "fixture",
                    None,
                    "https://example.test",
                    None,
                    "four factors",
                    2,
                    "2026-04-25T00:00:00Z",
                ),
            )
            con.execute(
                """
                INSERT INTO article_chunks(chunk_id, article_id, chunk_index, content, token_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("doc#0", "doc", 0, "four factors", 2),
            )
        con.commit()
    finally:
        con.close()
    return db_path


def test_cli_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0


def test_cli_help_mentions_rageval() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert "rageval" in result.output.lower()


def test_cli_version_exits_zero() -> None:
    result = CliRunner().invoke(main, ["version"])
    assert result.exit_code == 0


def test_cli_version_prints_version_string() -> None:
    result = CliRunner().invoke(main, ["version"])
    assert "0.1.0" in result.output


def test_cli_version_help() -> None:
    result = CliRunner().invoke(main, ["version", "--help"])
    assert result.exit_code == 0


def test_cli_run_creates_html_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
name: tiny-suite
cases:
  - id: case-001
    question: What are the four factors?
    question_type: analytical
    relevant_doc_ids: ["doc-1"]
""",
        encoding="utf-8",
    )
    output = tmp_path / "report.html"
    monkeypatch.setattr(cli, "_DB_PATH", _db_with_corpus(tmp_path))
    monkeypatch.setattr(
        HybridRAGSystem,
        "answer",
        lambda self, _question: RAGResponse(answer="fake answer"),
    )

    result = CliRunner().invoke(main, ["run", str(suite_path), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert "tiny-suite" in output.read_text(encoding="utf-8")
    assert "Suite: tiny-suite" in result.output
    assert "Cases: 1" in result.output
    assert f"Output: {output}" in result.output
    assert "[case-001]" not in result.output


def test_cli_run_metrics_filters_default_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
name: tiny-suite
cases:
  - id: case-001
    question: What are the four factors?
    question_type: analytical
    relevant_doc_ids: ["doc#0"]
""",
        encoding="utf-8",
    )
    output = tmp_path / "report.html"
    monkeypatch.setattr(cli, "_DB_PATH", _db_with_corpus(tmp_path))
    async def fake_answer(self: HybridRAGSystem, _question: str) -> RAGResponse:
        return RAGResponse(
            answer="fake answer",
            retrieved_docs=[Document(id="doc#1", content="four factors")],
            routing_decision=QuestionType.ANALYTICAL,
        )

    monkeypatch.setattr(HybridRAGSystem, "answer", fake_answer)

    result = CliRunner().invoke(
        main,
        ["run", str(suite_path), "--output", str(output), "--metrics", "refusal"],
    )

    assert result.exit_code == 0, result.output
    html = output.read_text(encoding="utf-8")
    assert "refusal" in html
    assert "prefix_recall@5" not in html


def test_cli_run_metrics_accepts_comma_and_repeated_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
name: tiny-suite
cases:
  - id: case-001
    question: What are the four factors?
    question_type: analytical
    relevant_doc_ids: ["doc#0"]
""",
        encoding="utf-8",
    )
    output = tmp_path / "report.html"
    monkeypatch.setattr(cli, "_DB_PATH", _db_with_corpus(tmp_path))
    async def fake_answer(self: HybridRAGSystem, _question: str) -> RAGResponse:
        return RAGResponse(
            answer="fake answer",
            retrieved_docs=[Document(id="doc#1", content="four factors")],
            routing_decision=QuestionType.ANALYTICAL,
        )

    monkeypatch.setattr(HybridRAGSystem, "answer", fake_answer)

    result = CliRunner().invoke(
        main,
        [
            "run",
            str(suite_path),
            "--output",
            str(output),
            "--metrics",
            "refusal,prefix_recall@5",
            "--metrics",
            "prefix_precision@5",
        ],
    )

    assert result.exit_code == 0, result.output
    html = output.read_text(encoding="utf-8")
    assert "refusal" in html
    assert "prefix_recall@5" in html
    assert "prefix_precision@5" in html
    assert "prefix_ndcg@5" not in html


def test_cli_run_metrics_rejects_unknown_metric(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("name: tiny-suite\ncases: []\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "run",
            str(suite_path),
            "--output",
            str(tmp_path / "report.html"),
            "--metrics",
            "unknown_metric",
        ],
    )

    assert result.exit_code != 0
    assert "unknown metric" in result.output
    assert "refusal" in result.output


def test_cli_run_help_mentions_metrics_and_no_cache() -> None:
    result = CliRunner().invoke(main, ["run", "--help"])

    assert result.exit_code == 0
    assert "--metrics" in result.output
    assert "--no-cache" in result.output
    assert "--live" in result.output
    assert "--offline" in result.output


def test_cli_run_no_cache_is_accepted_and_verbose_notes_deterministic_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("name: tiny-suite\ncases: []\n", encoding="utf-8")
    output = tmp_path / "report.html"
    monkeypatch.setattr(cli, "_DB_PATH", _db_with_corpus(tmp_path))

    result = CliRunner().invoke(
        main,
        [
            "run",
            str(suite_path),
            "--output",
            str(output),
            "--no-cache",
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "deterministic offline path does not use LLM cache" in result.output
    assert output.exists()


def test_cli_run_live_and_offline_are_mutually_exclusive(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("name: tiny-suite\ncases: []\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "run",
            str(suite_path),
            "--output",
            str(tmp_path / "report.html"),
            "--live",
            "--offline",
        ],
    )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_cli_run_explicit_live_requires_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("name: tiny-suite\ncases: []\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_DB_PATH", _db_with_corpus(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = CliRunner().invoke(
        main,
        ["run", str(suite_path), "--output", str(tmp_path / "report.html"), "--live"],
    )

    assert result.exit_code != 0
    assert "API_KEY" in result.output


def test_cli_run_defaults_to_live_when_required_keys_are_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("name: tiny-suite\ncases: []\n", encoding="utf-8")
    output = tmp_path / "report.html"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "_ensure_demo_data_ready", lambda _db_path: None)
    monkeypatch.setattr(cli, "_ensure_live_data_ready", lambda _db_path: None)

    class FakeSystem:
        name = "fake-live"

        async def answer(self, _question: str) -> RAGResponse:
            return RAGResponse(answer="fake")

    captured: dict[str, object] = {}

    def fake_system_for_mode(
        suite: TestSuite,
        mode: str,
        no_cache: bool,
    ) -> FakeSystem:
        captured["mode"] = mode
        captured["no_cache"] = no_cache
        return FakeSystem()

    monkeypatch.setattr(cli, "_system_for_mode", fake_system_for_mode)

    result = CliRunner().invoke(main, ["run", str(suite_path), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert captured["mode"] == "live"
    assert "Mode: live" in result.output


def test_cli_run_explicit_offline_overrides_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("name: tiny-suite\ncases: []\n", encoding="utf-8")
    output = tmp_path / "report.html"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "_ensure_demo_data_ready", lambda _db_path: None)

    class FakeSystem:
        name = "fake-offline"

        async def answer(self, _question: str) -> RAGResponse:
            return RAGResponse(answer="fake")

    captured: dict[str, object] = {}

    def fake_system_for_mode(
        suite: TestSuite,
        mode: str,
        no_cache: bool,
    ) -> FakeSystem:
        captured["mode"] = mode
        return FakeSystem()

    monkeypatch.setattr(cli, "_system_for_mode", fake_system_for_mode)

    result = CliRunner().invoke(
        main,
        ["run", str(suite_path), "--output", str(output), "--offline"],
    )

    assert result.exit_code == 0, result.output
    assert captured["mode"] == "offline"
    assert "Mode: offline" in result.output


def test_cli_run_fails_when_database_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_DB_PATH", tmp_path / "missing.db")
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("name: tiny-suite\ncases: []\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["run", str(suite_path), "--output", str(tmp_path / "r.html")],
    )

    assert result.exit_code != 0
    assert "Database not found" in result.output
    assert "scripts/build_stats_db.py" in result.output


def test_cli_run_fails_when_corpus_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_DB_PATH", _db_with_corpus(tmp_path, chunks=False))
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("name: tiny-suite\ncases: []\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["run", str(suite_path), "--output", str(tmp_path / "r.html")],
    )

    assert result.exit_code != 0
    assert "article_chunks has no rows" in result.output
    assert "scripts/build_corpus.py --from-cache" in result.output


def test_cli_demo_system_routes_from_suite_without_live_router() -> None:
    suite = TestSuite.model_validate(
        {
            "name": "tiny-suite",
            "cases": [
                {
                    "id": "case-001",
                    "question": "What are the four factors?",
                    "question_type": "analytical",
                }
            ],
        }
    )

    system = cli._demo_system(suite)

    assert isinstance(system.router, cli._SuiteRouter)
    assert isinstance(system.sql_agent, cli._DemoSQLAgent)


def test_factual_001_demo_sql_mapping_is_covered() -> None:
    sql = cli._demo_sql_for_question(
        "Who led the NBA in points per game in the 2023-24 regular season?"
    )

    assert sql is not None
    assert "Luka Dončić" in sql
    assert "points_per_game" in sql


def test_demo_sql_uses_seed_table_query_when_available() -> None:
    sql = cli._demo_sql_for_question(
        "How many wins did the Boston Celtics have in the 2023-24 regular season?"
    )

    assert sql is not None
    assert "FROM team_season_stats" in sql
    assert "JOIN teams" in sql
    assert "UNION ALL" not in sql


def test_default_cli_metrics_are_case_aware() -> None:
    metrics = cli._default_metrics()
    analytical = TestCase(
        id="analytical",
        question="What are the four factors?",
        question_type=QuestionType.ANALYTICAL,
        relevant_doc_ids=["doc#0"],
    )
    factual = TestCase(
        id="factual",
        question="What was the number?",
        question_type=QuestionType.FACTUAL,
        expected_numeric=1.0,
    )
    response = RAGResponse(answer="1.0")

    analytical_results = [metric(analytical, response) for metric in metrics]
    factual_results = [metric(factual, response) for metric in metrics]

    assert [r.metric_name for r in analytical_results if r is not None] == [
        "refusal",
        "prefix_precision@5",
        "prefix_recall@5",
        "prefix_reciprocal_rank",
        "prefix_ndcg@5",
    ]
    assert [r.metric_name for r in factual_results if r is not None] == [
        "numeric_tolerance",
        "refusal",
    ]


def test_offline_cli_sql_metric_uses_expected_sql_rows() -> None:
    metric = next(
        metric for metric in cli._default_metrics(mode="offline")
        if cli._cli_metric_name(metric) == "sql_equivalence"
    )
    case = TestCase(
        id="factual",
        question="Who led in PPG?",
        question_type=QuestionType.FACTUAL,
        expected_sql_rows=[{"player_name": "Luka Dončić"}],
        live_expected_sql_rows=[{"full_name": "Joel Embiid"}],
    )
    response = RAGResponse(
        answer="answer",
        sql_result=SQLResult(
            query="SELECT 'Luka Dončić'",
            rows=[{"player_name": "Luka Dončić"}],
        ),
    )

    result = metric(case, response)

    assert result is not None
    assert result.value == pytest.approx(1.0)
    assert result.details.get("expected_source") is None


def test_live_cli_sql_metric_uses_live_expected_sql_rows() -> None:
    metric = next(
        metric for metric in cli._default_metrics(mode="live")
        if cli._cli_metric_name(metric) == "sql_equivalence"
    )
    case = TestCase(
        id="factual",
        question="Who led in PPG?",
        question_type=QuestionType.FACTUAL,
        expected_sql_rows=[{"player_name": "Luka Dončić"}],
        live_expected_sql_rows=[{"full_name": "Joel Embiid"}],
    )
    response = RAGResponse(
        answer="answer",
        sql_result=SQLResult(
            query="SELECT 'Joel Embiid'",
            rows=[{"full_name": "Joel Embiid"}],
        ),
    )

    result = metric(case, response)

    assert result is not None
    assert result.value == pytest.approx(1.0)
    assert result.details["expected_source"] == "live_expected_sql_rows"


def test_live_cli_sql_metric_emits_skip_when_live_expected_rows_missing() -> None:
    metric = next(
        metric for metric in cli._default_metrics(mode="live")
        if cli._cli_metric_name(metric) == "sql_equivalence"
    )
    case = TestCase(
        id="factual",
        question="Who led in PPG?",
        question_type=QuestionType.FACTUAL,
        expected_sql_rows=[{"player_name": "Luka Dončić"}],
    )
    response = RAGResponse(
        answer="answer",
        sql_result=SQLResult(
            query="SELECT 'Luka Dončić'",
            rows=[{"player_name": "Luka Dončić"}],
        ),
    )

    result = metric(case, response)

    assert result is not None
    assert result.metric_name == "sql_equivalence"
    assert result.value is None
    assert result.details["skipped"] is True
    assert "live_expected_sql_rows" in result.details["reason"]


def test_live_cli_sql_metric_not_applicable_without_any_expected_rows() -> None:
    metric = next(
        metric for metric in cli._default_metrics(mode="live")
        if cli._cli_metric_name(metric) == "sql_equivalence"
    )
    case = TestCase(
        id="analytical",
        question="What are the four factors?",
        question_type=QuestionType.ANALYTICAL,
    )

    assert metric(case, RAGResponse(answer="answer")) is None


def test_live_metric_filter_preserves_sql_skip_semantics() -> None:
    metrics = cli._default_metrics({"sql_equivalence"}, mode="live")
    case = TestCase(
        id="factual",
        question="Who led in PPG?",
        question_type=QuestionType.FACTUAL,
        expected_sql_rows=[{"player_name": "Luka Dončić"}],
    )

    results = [
        result
        for metric in metrics
        if (result := metric(case, RAGResponse(answer="answer"))) is not None
    ]

    assert [result.metric_name for result in results] == ["sql_equivalence"]
    assert results[0].details["skipped"] is True


def test_default_cli_retrieval_metrics_match_article_prefixes() -> None:
    metrics = cli._default_metrics()
    case = TestCase(
        id="case-1",
        question="Jokic offense",
        question_type=QuestionType.ANALYTICAL,
        relevant_doc_ids=["thinking-basketball-jokic-offense#2"],
    )
    response = RAGResponse(
        answer="answer",
        retrieved_docs=[
            Document(id="thinking-basketball-jokic-offense#0", content="Jokic offense"),
            Document(id="other#0", content="Other"),
        ],
    )

    results = {
        result.metric_name: result
        for metric in metrics
        if (result := metric(case, response)) is not None
    }

    assert results["prefix_precision@5"].value == pytest.approx(0.2)
    assert results["prefix_recall@5"].value == pytest.approx(1.0)
    assert results["prefix_reciprocal_rank"].value == pytest.approx(1.0)
    assert results["prefix_ndcg@5"].value == pytest.approx(1.0)
    assert results["prefix_recall@5"].details["matching"] == "article_id_prefix"


def test_prefix_retrieval_metrics_deduplicate_prefixes() -> None:
    metric = cli.PrefixRecallAtK(5)
    case = TestCase(
        id="case-1",
        question="Four factors",
        question_type=QuestionType.ANALYTICAL,
        relevant_doc_ids=["doc#0", "doc#1"],
    )
    response = RAGResponse(
        answer="answer",
        retrieved_docs=[
            Document(id="doc#3", content="first"),
            Document(id="doc#4", content="second"),
        ],
    )

    result = metric(case, response)

    assert result.value == pytest.approx(1.0)
    assert result.details["expected_prefixes"] == ["doc"]
    assert result.details["retrieved_prefixes"] == ["doc"]


def test_cli_run_verbose_prints_case_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
name: tiny-suite
cases:
  - id: case-001
    question: What are the four factors?
    question_type: analytical
    relevant_doc_ids: ["doc-1"]
  - id: case-002
    question: Who led the NBA in points per game in the 2023-24 regular season?
    question_type: factual
""",
        encoding="utf-8",
    )
    output = tmp_path / "report.html"
    monkeypatch.setattr(cli, "_DB_PATH", _db_with_corpus(tmp_path))

    async def fake_answer(self: HybridRAGSystem, _question: str) -> RAGResponse:
        return RAGResponse(answer="fake answer", routing_decision=QuestionType.FACTUAL)

    monkeypatch.setattr(HybridRAGSystem, "answer", fake_answer)

    result = CliRunner().invoke(
        main, ["run", str(suite_path), "--output", str(output), "--verbose"]
    )

    assert result.exit_code == 0, result.output
    assert "[case-001]" in result.output
    assert "[case-002]" in result.output
    assert "route=factual" in result.output
    assert "refused=false" in result.output
    assert output.exists()


def test_cli_run_verbose_non_tty_uses_plain_case_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
name: tiny-suite
cases:
  - id: case-001
    question: What are the four factors?
    question_type: analytical
    relevant_doc_ids: ["doc-1"]
""",
        encoding="utf-8",
    )
    output = tmp_path / "report.html"
    monkeypatch.setattr(cli, "_DB_PATH", _db_with_corpus(tmp_path))

    async def fake_answer(self: HybridRAGSystem, _question: str) -> RAGResponse:
        return RAGResponse(answer="fake answer", routing_decision=QuestionType.ANALYTICAL)

    monkeypatch.setattr(HybridRAGSystem, "answer", fake_answer)

    result = CliRunner().invoke(
        main, ["run", str(suite_path), "--output", str(output), "--verbose"]
    )

    assert result.exit_code == 0, result.output
    assert "[case-001] route=analytical" in result.output
    assert "\r" not in result.output
    assert "Evaluating cases" not in result.output


def test_cli_progress_renders_only_for_tty_live_or_verbose() -> None:
    tty_console = Console(file=io.StringIO(), force_terminal=True)
    plain_console = Console(file=io.StringIO(), force_terminal=False)

    assert cli._should_render_progress(tty_console, mode="live", verbose=False)
    assert cli._should_render_progress(tty_console, mode="offline", verbose=True)
    assert not cli._should_render_progress(tty_console, mode="offline", verbose=False)
    assert not cli._should_render_progress(plain_console, mode="live", verbose=False)


def test_cli_run_live_non_verbose_non_tty_suppresses_case_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
name: tiny-suite
cases:
  - id: case-001
    question: What are the four factors?
    question_type: analytical
    relevant_doc_ids: ["doc-1"]
""",
        encoding="utf-8",
    )
    output = tmp_path / "report.html"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "_ensure_demo_data_ready", lambda _db_path: None)
    monkeypatch.setattr(cli, "_ensure_live_data_ready", lambda _db_path: None)

    class FakeSystem:
        name = "fake-live"

        async def answer(self, _question: str) -> RAGResponse:
            return RAGResponse(answer="fake", routing_decision=QuestionType.ANALYTICAL)

    monkeypatch.setattr(cli, "_system_for_mode", lambda *_args: FakeSystem())

    result = CliRunner().invoke(
        main, ["run", str(suite_path), "--output", str(output), "--live"]
    )

    assert result.exit_code == 0, result.output
    assert "Mode: live" in result.output
    assert "[case-001]" not in result.output
    assert "\r" not in result.output
    assert "Evaluating cases" not in result.output


def test_cli_demo_runs_representative_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = tmp_path / "demo_suite.yaml"
    bundled.write_text(
        """
name: demo-suite
cases:
  - id: factual-001
    question: Who led the NBA in points per game in the 2023-24 regular season?
    question_type: factual
  - id: factual-002
    question: A second factual case that should be skipped during selection.
    question_type: factual
  - id: analytical-001
    question: What are the four factors?
    question_type: analytical
    relevant_doc_ids: ["doc-1"]
  - id: hybrid-001
    question: Some hybrid analysis question.
    question_type: hybrid
  - id: adversarial-001
    question: Will the Knicks win in 2030?
    question_type: unanswerable
    should_refuse: true
""",
        encoding="utf-8",
    )
    output = tmp_path / "demo-report.html"
    monkeypatch.setattr(cli, "_demo_suite_path", lambda: bundled)
    monkeypatch.setattr(cli, "_DB_PATH", _db_with_corpus(tmp_path))
    monkeypatch.setattr(
        HybridRAGSystem,
        "answer",
        lambda self, _question: RAGResponse(answer="fake answer"),
    )

    result = CliRunner().invoke(
        main,
        ["demo", "--output", str(output), "--metrics", "refusal", "--no-cache", "--verbose"],
    )

    assert result.exit_code == 0, result.output
    assert "Suite: demo-suite" in result.output
    assert "deterministic offline path does not use LLM cache" in result.output
    # 4 unique categories present, target=5 → adds factual-002 to fill.
    assert "Cases: 5" in result.output
    text = output.read_text(encoding="utf-8")
    assert "factual-001" in text
    assert "analytical-001" in text
    assert "hybrid-001" in text
    assert "adversarial-001" in text
    assert "factual-002" in text


def test_cli_demo_caps_at_target_when_categories_exceed_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = tmp_path / "demo_suite.yaml"
    bundled.write_text(
        """
name: demo-suite
cases:
  - id: factual-001
    question: Q1.
    question_type: factual
  - id: analytical-001
    question: Q2.
    question_type: analytical
""",
        encoding="utf-8",
    )
    output = tmp_path / "demo-report.html"
    monkeypatch.setattr(cli, "_demo_suite_path", lambda: bundled)
    monkeypatch.setattr(cli, "_DB_PATH", _db_with_corpus(tmp_path))
    monkeypatch.setattr(
        HybridRAGSystem,
        "answer",
        lambda self, _question: RAGResponse(answer="fake answer"),
    )

    result = CliRunner().invoke(main, ["demo", "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert "Cases: 2" in result.output


def test_cli_demo_reports_missing_bundled_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_demo_suite_path", lambda: tmp_path / "missing.yaml")

    result = CliRunner().invoke(main, ["demo", "--output", str(tmp_path / "out.html")])

    assert result.exit_code != 0
    assert "Test suite file not found" in result.output


def test_demo_suite_path_uses_package_resource() -> None:
    path = cli._demo_suite_path()

    assert path.name == "nba_test_suite.yaml"
    assert "src/rageval/examples" in path.as_posix()
    assert path.exists()


def test_cli_calibrate_delegates_to_calibration_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    async def fake_run(
        judges: list[str],
        threshold: float,
        no_cache: bool,
    ) -> bool:
        called["judges"] = judges
        called["threshold"] = threshold
        called["no_cache"] = no_cache
        return True

    from scripts import calibrate_judge

    monkeypatch.setattr(calibrate_judge, "run", fake_run)

    result = CliRunner().invoke(
        main,
        ["calibrate", "routing", "--threshold", "0.9", "--no-cache"],
    )

    assert result.exit_code == 0, result.output
    assert called == {"judges": ["routing"], "threshold": 0.9, "no_cache": True}


def test_cli_calibrate_accepts_multiple_judges(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    async def fake_run(
        judges: list[str],
        threshold: float,
        no_cache: bool,
    ) -> bool:
        called["judges"] = judges
        called["threshold"] = threshold
        called["no_cache"] = no_cache
        return True

    from scripts import calibrate_judge

    monkeypatch.setattr(calibrate_judge, "run", fake_run)

    result = CliRunner().invoke(
        main,
        ["calibrate", "faithfulness", "relevance", "correctness", "routing", "--no-cache"],
    )

    assert result.exit_code == 0, result.output
    assert called == {
        "judges": ["faithfulness", "relevance", "correctness", "routing"],
        "threshold": 0.8,
        "no_cache": True,
    }


def test_cli_calibrate_fails_when_calibration_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(
        judges: list[str],
        threshold: float,
        no_cache: bool,
    ) -> bool:
        return False

    from scripts import calibrate_judge

    monkeypatch.setattr(calibrate_judge, "run", fake_run)

    result = CliRunner().invoke(main, ["calibrate", "routing"])

    assert result.exit_code != 0
    assert "Calibration failed" in result.output


def test_select_demo_cases_picks_one_per_category_first() -> None:
    cases = [
        TestCase(id="f1", question="q", question_type=QuestionType.FACTUAL),
        TestCase(id="f2", question="q", question_type=QuestionType.FACTUAL),
        TestCase(id="a1", question="q", question_type=QuestionType.ANALYTICAL),
        TestCase(id="h1", question="q", question_type=QuestionType.HYBRID),
        TestCase(id="u1", question="q", question_type=QuestionType.UNANSWERABLE),
        TestCase(id="a2", question="q", question_type=QuestionType.ANALYTICAL),
    ]

    selected = cli._select_demo_cases(cases, target=5)

    assert [c.id for c in selected] == ["f1", "a1", "h1", "u1", "f2"]


def test_cli_run_rejects_non_positive_max_cases(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("name: tiny-suite\ncases: []\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["run", str(suite_path), "--output", str(tmp_path / "report.html"), "--max-cases", "0"],
    )

    assert result.exit_code != 0
    assert "--max-cases must be greater than zero" in result.output
