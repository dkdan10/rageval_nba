import json
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

import httpx
import pytest
import yaml

from scripts import build_corpus
from scripts.build_corpus import CorpusArticle, _cache_path, ingest_manifest


def _article(**overrides: object) -> dict[str, object]:
    article: dict[str, object] = {
        "article_id": "ctg-four-factors",
        "title": "Four Factors",
        "source": "fixture",
        "url": "https://example.test/four-factors",
        "topics": ["four factors", "efficiency"],
        "storage_policy": "repo_fixture",
        "notes": "Original fixture text for tests.",
    }
    article.update(overrides)
    return article


def _write_manifest(path: Path, articles: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(articles), encoding="utf-8")


def test_ingest_manifest_metadata_only_article_has_no_chunks(tmp_path: Path) -> None:
    manifest = tmp_path / "articles.json"
    _write_manifest(manifest, [_article(full_text="")])

    counts = ingest_manifest(manifest, db_path=tmp_path / "nba.db")

    assert counts == {
        "articles": 1,
        "article_chunks": 0,
        "fetched": 0,
        "skipped": 0,
        "failed": 0,
    }
    con = sqlite3.connect(tmp_path / "nba.db")
    try:
        assert con.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM article_chunks").fetchone()[0] == 0
    finally:
        con.close()


def test_ingest_manifest_chunks_permitted_local_full_text(tmp_path: Path) -> None:
    manifest = tmp_path / "articles.json"
    _write_manifest(
        manifest,
        [
            _article(
                full_text=(
                    "Effective field goal percentage matters. "
                    "Turnover rate matters. Offensive rebounding matters."
                )
            )
        ],
    )

    counts = ingest_manifest(manifest, db_path=tmp_path / "nba.db", max_tokens=5)

    assert counts["articles"] == 1
    assert counts["article_chunks"] == 3
    con = sqlite3.connect(tmp_path / "nba.db")
    try:
        chunks = con.execute(
            "SELECT chunk_id, content FROM article_chunks ORDER BY chunk_index"
        ).fetchall()
        assert chunks[0][0] == "ctg-four-factors#0"
        assert "Effective field goal percentage matters." in chunks[0][1]
    finally:
        con.close()


def test_ingest_manifest_from_cache_extracts_readable_html(tmp_path: Path) -> None:
    manifest = tmp_path / "articles.json"
    article = _article(article_id="ctg-shot-quality", url="https://example.test/shot")
    _write_manifest(manifest, [article])
    cache_dir = tmp_path / "cache"
    cache_article = CorpusArticle(
        article_id="ctg-shot-quality",
        title=str(article["title"]),
        source=str(article["source"]),
        url=str(article["url"]),
        topics=["shot quality"],
        storage_policy="metadata_only",
        notes="Fixture.",
    )
    cache_file = _cache_path(cache_dir, cache_article)
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        """
        <html>
          <head><style>.hidden{display:none}</style><script>ignored()</script></head>
          <body>
            <nav>navigation links</nav>
            <main>Shot quality estimates expected points from location and defense.</main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    counts = ingest_manifest(
        manifest,
        db_path=tmp_path / "nba.db",
        from_cache=True,
        cache_dir=cache_dir,
        max_tokens=50,
    )

    assert counts["article_chunks"] == 1
    con = sqlite3.connect(tmp_path / "nba.db")
    try:
        content = con.execute("SELECT content FROM article_chunks").fetchone()[0]
        assert "Shot quality estimates expected points" in content
        assert "navigation links" not in content
        assert "ignored" not in content
    finally:
        con.close()


def test_ingest_manifest_from_cache_preserves_repo_authored_summary(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "articles.json"
    article = _article(
        article_id="thinking-basketball-zone-defense",
        title="Zone Defense",
        url="https://example.test/zone",
        storage_policy="metadata_plus_repo_authored_summary",
        full_text="Repo-authored summary: Zone defense can protect weak defenders.",
    )
    _write_manifest(manifest, [article])
    cache_dir = tmp_path / "cache"
    cache_article = CorpusArticle(
        article_id="thinking-basketball-zone-defense",
        title=str(article["title"]),
        source=str(article["source"]),
        url=str(article["url"]),
        topics=["zone defense"],
        storage_policy="metadata_plus_repo_authored_summary",
        notes="Fixture.",
    )
    cache_file = _cache_path(cache_dir, cache_article)
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        "<html><body><main>Cached page text about rotations.</main></body></html>",
        encoding="utf-8",
    )

    counts = ingest_manifest(
        manifest,
        db_path=tmp_path / "nba.db",
        from_cache=True,
        cache_dir=cache_dir,
        max_tokens=100,
    )

    assert counts["article_chunks"] == 1
    con = sqlite3.connect(tmp_path / "nba.db")
    try:
        content = con.execute("SELECT content FROM article_chunks").fetchone()[0]
        assert "Cached page text about rotations." in content
        assert "Zone defense can protect weak defenders." in content
    finally:
        con.close()


@pytest.mark.parametrize(
    "field",
    ["article_id", "title", "source", "url", "topics", "storage_policy", "notes"],
)
def test_ingest_manifest_requires_curated_metadata_fields(
    tmp_path: Path,
    field: str,
) -> None:
    manifest = tmp_path / "articles.json"
    article = _article()
    article.pop(field)
    _write_manifest(manifest, [article])

    with pytest.raises(ValueError, match=field if field != "storage_policy" else "storage"):
        ingest_manifest(manifest, db_path=tmp_path / "nba.db")


def test_ingest_manifest_rejects_non_list_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "articles.json"
    manifest.write_text(json.dumps({"title": "bad"}), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON list"):
        ingest_manifest(manifest, db_path=tmp_path / "nba.db")


def test_script_can_run_by_direct_path(tmp_path: Path) -> None:
    manifest = tmp_path / "articles.json"
    _write_manifest(manifest, [_article()])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_corpus.py",
            "--manifest",
            str(manifest),
            "--db",
            str(tmp_path / "nba.db"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        check=False,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Corpus built:" in result.stdout


def test_fetch_failure_is_logged_without_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "articles.json"
    _write_manifest(manifest, [_article()])

    monkeypatch.setattr(build_corpus, "_can_fetch", lambda *_args: (True, "allowed"))

    def fail_fetch(_url: str, _user_agent: str) -> str:
        raise httpx.ConnectError("network unavailable")

    monkeypatch.setattr(build_corpus, "_fetch_url", fail_fetch)

    counts = ingest_manifest(
        manifest,
        db_path=tmp_path / "nba.db",
        fetch=True,
        cache_dir=tmp_path / "cache",
        rate_limit_seconds=0.0,
    )

    assert counts["failed"] == 1
    assert counts["articles"] == 1
    con = sqlite3.connect(tmp_path / "nba.db")
    try:
        notes = con.execute(
            "SELECT notes FROM ingestion_log WHERE source='corpus_manifest'"
        ).fetchone()[0]
        assert "failed=1" in notes
    finally:
        con.close()


def test_robots_disallowed_fetch_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "articles.json"
    _write_manifest(manifest, [_article()])

    monkeypatch.setattr(
        build_corpus,
        "_can_fetch",
        lambda *_args: (False, "robots.txt disallows fetch"),
    )

    def should_not_fetch(_url: str, _user_agent: str) -> str:
        raise AssertionError("fetch should not be called when robots disallows it")

    monkeypatch.setattr(build_corpus, "_fetch_url", should_not_fetch)

    counts = ingest_manifest(
        manifest,
        db_path=tmp_path / "nba.db",
        fetch=True,
        cache_dir=tmp_path / "cache",
        rate_limit_seconds=0.0,
    )

    assert counts["skipped"] == 1
    assert counts["failed"] == 0


def test_robots_timeout_is_recorded_as_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*_args: object, **_kwargs: object) -> object:
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", timeout)

    allowed, reason = build_corpus._can_fetch("https://example.test/article", "agent")

    assert allowed is False
    assert "robots check failed" in reason


def test_ingestion_log_includes_useful_summary_notes(tmp_path: Path) -> None:
    manifest = tmp_path / "articles.json"
    _write_manifest(manifest, [_article(full_text="one two three")])
    cache_dir = tmp_path / "cache"

    ingest_manifest(
        manifest,
        db_path=tmp_path / "nba.db",
        cache_dir=cache_dir,
        max_tokens=10,
    )

    con = sqlite3.connect(tmp_path / "nba.db")
    try:
        notes = con.execute(
            "SELECT notes FROM ingestion_log WHERE source='corpus_manifest'"
        ).fetchone()[0]
        assert f"manifest={manifest}" in notes
        assert "articles=1" in notes
        assert "fetched=0" in notes
        assert "skipped=0" in notes
        assert "failed=0" in notes
        assert "chunks=1" in notes
        assert f"cache_dir={cache_dir}" in notes
    finally:
        con.close()


def test_curated_manifest_covers_suite_relevant_doc_id_prefixes() -> None:
    manifest = json.loads(Path("examples/corpus/articles.json").read_text(encoding="utf-8"))
    manifest_ids = {article["article_id"] for article in manifest}

    suite = yaml.safe_load(
        Path("examples/nba_test_suite.yaml").read_text(encoding="utf-8")
    )
    relevant_prefixes = {
        doc_id.split("#", maxsplit=1)[0]
        for case in suite["cases"]
        for doc_id in case.get("relevant_doc_ids", [])
    }

    assert relevant_prefixes <= manifest_ids
    assert len(manifest) >= 40


def test_curated_manifest_has_repo_authored_text_for_known_live_fetch_gaps() -> None:
    manifest = json.loads(Path("examples/corpus/articles.json").read_text(encoding="utf-8"))
    articles = {article["article_id"]: article for article in manifest}

    for article_id in {
        "nba-stats-glossary",
        "nba-stats-playtype",
        "synergy-play-types",
        "thinking-basketball-zone-defense",
    }:
        article = articles[article_id]
        assert "repo_authored" in article["storage_policy"]
        assert article["full_text"].startswith("Repo-authored summary:")
        assert len(article["full_text"].split()) >= 20
