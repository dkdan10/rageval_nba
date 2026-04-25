from collections import Counter
from pathlib import Path

from rageval.demo.rag_agent import RAGAgent
from rageval.types import QuestionType, TestSuite
from scripts.build_corpus import ingest_manifest


def test_nba_suite_loads_and_has_milestone7_category_counts() -> None:
    suite = TestSuite.from_yaml("examples/nba_test_suite.yaml")

    counts = Counter(case.question_type for case in suite.cases)

    assert counts == {
        QuestionType.FACTUAL: 12,
        QuestionType.ANALYTICAL: 15,
        QuestionType.HYBRID: 10,
        QuestionType.UNANSWERABLE: 5,
    }


def test_nba_suite_cases_have_category_specific_fields() -> None:
    suite = TestSuite.from_yaml("examples/nba_test_suite.yaml")

    for case in suite.cases:
        if case.question_type is QuestionType.FACTUAL:
            assert (
                case.expected_sql_rows is not None or case.expected_numeric is not None
            ), case.id
        elif case.question_type is QuestionType.ANALYTICAL:
            assert case.relevant_doc_ids, case.id
        elif case.question_type is QuestionType.HYBRID:
            assert case.expected_sql_rows is not None, case.id
            assert case.relevant_doc_ids, case.id
        elif case.question_type is QuestionType.UNANSWERABLE:
            assert case.should_refuse is True, case.id


def test_repo_authored_manifest_summaries_support_known_fetch_gap_retrieval(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "corpus.db"
    ingest_manifest(
        Path("examples/corpus/articles.json"),
        db_path=db_path,
        cache_dir=tmp_path / "cache",
        max_tokens=120,
    )
    agent = RAGAgent(db_path)

    checks = {
        "What is VORP and how is it calculated?": "bbref-glossary",
        "What does play type analysis reveal about NBA offenses?": "synergy-play-types",
        "What are the analytical arguments for and against zone defense in the NBA?": (
            "thinking-basketball-zone-defense"
        ),
    }
    for question, expected_prefix in checks.items():
        retrieved_prefixes = {
            doc.id.split("#", maxsplit=1)[0] for doc in agent.retrieve(question, k=5)
        }
        assert expected_prefix in retrieved_prefixes
