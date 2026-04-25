"""Build the demo article corpus tables from a tracked metadata manifest.

The tracked manifest stores source metadata and URLs. Raw fetched HTML/text is
stored only in the gitignored cache directory (`data/raw/corpus/` by default).
Do not commit full article text unless you have permission to redistribute it.

Default mode ingests metadata and any explicitly provided, permitted `full_text`.
Use `--fetch` to populate the raw cache while respecting robots.txt, and
`--from-cache` to extract readable text from cached pages and chunk it into the
SQLite corpus tables. No embeddings are generated here; the demo RAGAgent uses a
deterministic lexical fallback, not production vector retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stats_db import _init_schema

DB_PATH = Path(__file__).parent.parent / "data" / "nba.db"
DEFAULT_MANIFEST = Path(__file__).parent.parent / "examples" / "corpus" / "articles.json"
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "corpus"
DEFAULT_USER_AGENT = "rageval-nba-corpus-builder/0.1 (+https://github.com/)"
DEFAULT_ROBOTS_TIMEOUT_SECONDS = 10.0
_ARTICLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class CorpusArticle:
    article_id: str
    title: str
    source: str
    url: str
    topics: list[str]
    storage_policy: str
    notes: str
    author: str | None = None
    publish_date: str | None = None
    full_text: str = ""


def _chunk_text(text: str, max_tokens: int = 220) -> list[str]:
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return []
    return [
        " ".join(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def _require_string(item: dict[str, Any], index: int, field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest item {index} missing required string field: {field}")
    return value.strip()


def _load_manifest(path: Path) -> list[CorpusArticle]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Corpus manifest must be a JSON list of article objects")

    articles: list[CorpusArticle] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item {i} must be an object")

        article_id = _require_string(item, i, "article_id")
        if not _ARTICLE_ID_RE.fullmatch(article_id):
            raise ValueError(f"Manifest item {i} has invalid article_id: {article_id}")
        if article_id in seen_ids:
            raise ValueError(f"Duplicate article_id in manifest: {article_id}")
        seen_ids.add(article_id)

        topics_raw = item.get("topics")
        if not isinstance(topics_raw, list) or not topics_raw:
            raise ValueError(f"Manifest item {i} missing required non-empty list field: topics")
        topics = [str(topic).strip() for topic in topics_raw if str(topic).strip()]
        if not topics:
            raise ValueError(f"Manifest item {i} topics must contain at least one string")

        storage_policy = item.get("storage_policy") or item.get("license")
        if not isinstance(storage_policy, str) or not storage_policy.strip():
            raise ValueError(
                f"Manifest item {i} missing required storage_policy or license field"
            )

        notes = _require_string(item, i, "notes")
        full_text = item.get("full_text")
        articles.append(
            CorpusArticle(
                article_id=article_id,
                title=_require_string(item, i, "title"),
                source=_require_string(item, i, "source"),
                url=_require_string(item, i, "url"),
                topics=topics,
                storage_policy=storage_policy.strip(),
                notes=notes,
                author=item.get("author") if isinstance(item.get("author"), str) else None,
                publish_date=(
                    item.get("publish_date")
                    if isinstance(item.get("publish_date"), str)
                    else None
                ),
                full_text=full_text if isinstance(full_text, str) else "",
            )
        )
    return articles


def _cache_path(cache_dir: Path, article: CorpusArticle) -> Path:
    digest = hashlib.sha256(article.url.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{article.article_id}-{digest}.html"


def _robots_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def _can_fetch(
    url: str,
    user_agent: str,
    timeout_seconds: float = DEFAULT_ROBOTS_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    parser = RobotFileParser()
    parser.set_url(_robots_url(url))
    try:
        with urllib.request.urlopen(  # noqa: S310 - URL comes from curated manifest.
            _robots_url(url),
            timeout=timeout_seconds,
        ) as response:
            parser.parse(line.decode("utf-8", errors="ignore") for line in response)
    except (OSError, urllib.error.URLError) as exc:
        return False, f"robots check failed: {exc}"
    if not parser.can_fetch(user_agent, url):
        return False, "robots.txt disallows fetch"
    return True, "allowed"


def _fetch_url(url: str, user_agent: str) -> str:
    response = httpx.get(
        url,
        headers={"User-Agent": user_agent},
        follow_redirects=True,
        timeout=20.0,
    )
    response.raise_for_status()
    return response.text


def _extract_text_from_html(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def _load_cached_text(cache_path: Path) -> str:
    raw = cache_path.read_text(encoding="utf-8", errors="ignore")
    if "<html" in raw[:500].casefold() or "<body" in raw[:1000].casefold():
        return _extract_text_from_html(raw)
    return " ".join(raw.split())


def _merge_texts(*texts: str) -> str:
    return "\n\n".join(text.strip() for text in texts if text.strip())


def ingest_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    db_path: Path = DB_PATH,
    max_tokens: int = 220,
    *,
    fetch: bool = False,
    from_cache: bool = False,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    rate_limit_seconds: float = 3.0,
    user_agent: str = DEFAULT_USER_AGENT,
    respect_robots: bool = True,
) -> dict[str, int]:
    articles = _load_manifest(manifest_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fetched_count = 0
    skipped_count = 0
    failed_count = 0
    article_count = 0
    chunk_count = 0
    last_fetch_by_domain: dict[str, float] = {}

    con = sqlite3.connect(db_path)
    try:
        _init_schema(con)
        run_at = datetime.now(UTC).isoformat()
        with con:
            for article in articles:
                cache_path = _cache_path(cache_dir, article)
                fetch_note = "not requested"

                if fetch:
                    if respect_robots:
                        allowed, reason = _can_fetch(article.url, user_agent)
                        if not allowed:
                            skipped_count += 1
                            fetch_note = reason
                        else:
                            fetched = _fetch_article(
                                article=article,
                                cache_path=cache_path,
                                user_agent=user_agent,
                                rate_limit_seconds=rate_limit_seconds,
                                last_fetch_by_domain=last_fetch_by_domain,
                            )
                            if fetched is None:
                                failed_count += 1
                                fetch_note = "fetch failed"
                            else:
                                fetched_count += 1
                                fetch_note = "fetched"
                    else:
                        fetched = _fetch_article(
                            article=article,
                            cache_path=cache_path,
                            user_agent=user_agent,
                            rate_limit_seconds=rate_limit_seconds,
                            last_fetch_by_domain=last_fetch_by_domain,
                        )
                        if fetched is None:
                            failed_count += 1
                            fetch_note = "fetch failed"
                        else:
                            fetched_count += 1
                            fetch_note = "fetched"

                full_text = article.full_text
                if from_cache and cache_path.exists():
                    full_text = _merge_texts(article.full_text, _load_cached_text(cache_path))

                chunks = _chunk_text(full_text, max_tokens=max_tokens)
                word_count = len(_TOKEN_RE.findall(full_text))
                con.execute(
                    """
                    INSERT OR REPLACE INTO articles(
                        article_id, title, source, author, url, publish_date,
                        full_text, word_count, ingested_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        article.article_id,
                        article.title,
                        article.source,
                        article.author,
                        article.url,
                        article.publish_date,
                        full_text,
                        word_count,
                        run_at,
                    ),
                )
                con.execute(
                    "DELETE FROM article_chunks WHERE article_id = ?",
                    (article.article_id,),
                )
                for chunk_index, chunk in enumerate(chunks):
                    token_count = len(_TOKEN_RE.findall(chunk))
                    con.execute(
                        """
                        INSERT OR REPLACE INTO article_chunks(
                            chunk_id, article_id, chunk_index, content, token_count
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            f"{article.article_id}#{chunk_index}",
                            article.article_id,
                            chunk_index,
                            chunk,
                            token_count,
                        ),
                    )
                    chunk_count += 1
                article_count += 1

                con.execute(
                    "INSERT INTO ingestion_log(run_at, source, records_added, notes)"
                    " VALUES (?, ?, ?, ?)",
                    (
                        run_at,
                        "corpus_article",
                        1 + len(chunks),
                        (
                            f"article_id={article.article_id}; fetch={fetch_note}; "
                            f"chunks={len(chunks)}; cache={cache_path}"
                        ),
                    ),
                )

            summary = (
                f"manifest={manifest_path}; articles={article_count}; "
                f"fetched={fetched_count}; skipped={skipped_count}; "
                f"failed={failed_count}; chunks={chunk_count}; cache_dir={cache_dir}"
            )
            con.execute(
                "INSERT INTO ingestion_log(run_at, source, records_added, notes)"
                " VALUES (?, ?, ?, ?)",
                (
                    run_at,
                    "corpus_manifest",
                    article_count + chunk_count,
                    summary,
                ),
            )
        return {
            "articles": article_count,
            "article_chunks": chunk_count,
            "fetched": fetched_count,
            "skipped": skipped_count,
            "failed": failed_count,
        }
    finally:
        con.close()


def _fetch_article(
    article: CorpusArticle,
    cache_path: Path,
    user_agent: str,
    rate_limit_seconds: float,
    last_fetch_by_domain: dict[str, float],
) -> str | None:
    domain = urlparse(article.url).netloc
    now = time.monotonic()
    previous = last_fetch_by_domain.get(domain)
    if previous is not None:
        delay = rate_limit_seconds - (now - previous)
        if delay > 0:
            time.sleep(delay)
    try:
        raw = _fetch_url(article.url, user_agent)
    except httpx.HTTPError:
        last_fetch_by_domain[domain] = time.monotonic()
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(raw, encoding="utf-8")
    last_fetch_by_domain[domain] = time.monotonic()
    return raw


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build article corpus tables from a JSON manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rate-limit-seconds", type=float, default=3.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    robots = parser.add_mutually_exclusive_group()
    robots.add_argument("--respect-robots", dest="respect_robots", action="store_true")
    robots.add_argument("--no-respect-robots", dest="respect_robots", action="store_false")
    parser.set_defaults(respect_robots=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    counts = ingest_manifest(
        args.manifest,
        db_path=args.db,
        max_tokens=args.max_tokens,
        fetch=args.fetch,
        from_cache=args.from_cache,
        cache_dir=args.cache_dir,
        rate_limit_seconds=args.rate_limit_seconds,
        user_agent=args.user_agent,
        respect_robots=args.respect_robots,
    )
    print(f"Corpus built: {counts}")


if __name__ == "__main__":
    main()
