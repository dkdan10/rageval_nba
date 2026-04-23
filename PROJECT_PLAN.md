# NBA Hybrid RAG Evaluation Harness — Project Plan

## Overview

A Python library and CLI for evaluating hybrid RAG systems — systems that combine unstructured text retrieval with structured database queries. The reference implementation evaluates a hybrid RAG system over NBA analytics writing (unstructured) and NBA statistics (structured).

**Why this project:** Production RAG systems in 2026 are hybrid, not pure vector search. Evaluating them requires metrics across multiple dimensions: routing accuracy, retrieval quality, SQL correctness, and answer quality. No existing open-source tool does this cleanly.

**Stack:** Python 3.11+, uv, Pydantic v2, Anthropic API (Claude), SQLite with sqlite-vec, Jinja2, Click, Rich.

**Deliverables:** Installable Python package on PyPI, HTML report generator, CLI, reference NBA demo system with 40 handcrafted test cases, 90-second demo video.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     EVALUATION HARNESS                        │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Test Suite  │───▶│  Evaluator   │───▶│    Report    │   │
│  │  (YAML)      │    │ (async core) │    │   (HTML)     │   │
│  └──────────────┘    └──────┬───────┘    └──────────────┘   │
│                             │                                 │
│         ┌───────────────────┼───────────────────┐            │
│         ▼                   ▼                   ▼            │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐   │
│  │  Retrieval   │    │  LLM Judge   │   │  Structured  │   │
│  │   Metrics    │    │   Metrics    │   │   Metrics    │   │
│  │              │    │              │   │              │   │
│  │ P@k, R@k,    │    │ Faithfulness │   │ Exact match  │   │
│  │ MRR, nDCG    │    │ Relevance    │   │ Numeric tol. │   │
│  │              │    │ Correctness  │   │ SQL equiv.   │   │
│  └──────────────┘    └──────────────┘   └──────────────┘   │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼ runs against
┌──────────────────────────────────────────────────────────────┐
│            REFERENCE HYBRID RAG SYSTEM (the demo)             │
│                                                               │
│                         Question                              │
│                            │                                  │
│                            ▼                                  │
│                    ┌──────────────┐                          │
│                    │    Router    │ (Claude classifier)      │
│                    └──────┬───────┘                          │
│             ┌─────────────┼─────────────┐                    │
│             ▼             ▼             ▼                    │
│       ┌─────────┐   ┌──────────┐  ┌──────────┐             │
│       │   SQL   │   │  Vector  │  │  Hybrid  │             │
│       │  Path   │   │   Path   │  │   Path   │             │
│       └────┬────┘   └────┬─────┘  └────┬─────┘             │
│            ▼             ▼             ▼                     │
│       ┌─────────┐   ┌──────────┐  ┌──────────┐             │
│       │  Stats  │   │ Articles │  │   Both   │             │
│       │  (SQL)  │   │ (vector) │  │          │             │
│       └────┬────┘   └────┬─────┘  └────┬─────┘             │
│            └─────────────┼─────────────┘                     │
│                          ▼                                   │
│                   ┌─────────────┐                            │
│                   │  Synthesize │ (Claude generator)         │
│                   └──────┬──────┘                            │
│                          ▼                                   │
│                       Answer                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
rageval-nba/
├── .github/workflows/ci.yml
├── .gitignore
├── .env.example
├── README.md
├── PROJECT_PLAN.md           # This file
├── LICENSE                   # MIT
├── pyproject.toml
├── uv.lock
│
├── src/rageval/
│   ├── __init__.py
│   ├── types.py              # Pydantic models (core data types)
│   ├── cache.py              # Disk-based LLM response cache
│   ├── llm_client.py         # Async Anthropic wrapper
│   ├── evaluator.py          # Main orchestrator
│   ├── reporting.py          # HTML report generation
│   ├── cli.py                # Click CLI entry point
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── base.py           # Metric protocol and base class
│   │   ├── retrieval.py      # P@k, R@k, MRR, nDCG
│   │   ├── structured.py     # Exact match, numeric tolerance, SQL equiv
│   │   ├── judge.py          # LLM-as-judge metrics
│   │   └── routing.py        # Routing classification metrics
│   │
│   └── demo/
│       ├── __init__.py
│       ├── system.py         # HybridRAGSystem implementation
│       ├── router.py         # Question classifier
│       ├── sql_agent.py      # Text-to-SQL component
│       ├── rag_agent.py      # Vector retrieval component
│       ├── synthesizer.py    # Final answer generator
│       └── data_loader.py    # NBA data ingestion
│
├── examples/
│   ├── nba_test_suite.yaml   # 40 handcrafted test cases
│   ├── corpus/               # Article URLs + chunked content (or setup script)
│   └── data/
│       └── nba.db            # SQLite DB (built by setup script)
│
├── prompts/                  # Versioned prompt files
│   ├── router/
│   ├── sql_agent/
│   ├── synthesizer/
│   └── judges/
│
├── tests/
│   ├── conftest.py
│   ├── test_types.py
│   ├── test_retrieval_metrics.py
│   ├── test_structured_metrics.py
│   ├── test_judges.py
│   ├── test_evaluator.py
│   └── fixtures/
│
├── docs/
│   ├── judge_calibration.md  # Judge agreement with human labels
│   ├── prompt_evolution.md   # Why prompts changed over iterations
│   └── architecture.md
│
└── scripts/
    ├── build_stats_db.py     # Pull NBA data → SQLite
    ├── build_corpus.py       # Scrape + chunk + embed articles
    └── calibrate_judge.py    # Run judges against human labels
```

---

## Data Schema

### SQLite Schema (stats database)

The SQLite database serves both the structured stats (SQL path) and the vector store (RAG path), using the `sqlite-vec` extension. One file, clean.

```sql
-- ============================================
-- STRUCTURED DATA: NBA stats
-- ============================================

CREATE TABLE teams (
    team_id INTEGER PRIMARY KEY,
    team_abbr TEXT NOT NULL UNIQUE,          -- 'BOS', 'LAL', etc.
    team_name TEXT NOT NULL,                 -- 'Boston Celtics'
    team_city TEXT NOT NULL,
    conference TEXT NOT NULL CHECK(conference IN ('East', 'West')),
    division TEXT NOT NULL
);

CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    position TEXT,                           -- 'G', 'F', 'C', 'G-F', etc.
    height_inches INTEGER,
    weight_lbs INTEGER,
    birth_date TEXT,                         -- ISO 8601
    draft_year INTEGER,
    draft_pick INTEGER                       -- NULL if undrafted
);

CREATE TABLE seasons (
    season_id TEXT PRIMARY KEY,              -- '2023-24'
    start_year INTEGER NOT NULL,
    end_year INTEGER NOT NULL
);

CREATE TABLE games (
    game_id INTEGER PRIMARY KEY,
    game_date TEXT NOT NULL,                 -- ISO 8601
    season_id TEXT NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    game_type TEXT NOT NULL CHECK(game_type IN ('regular', 'playoff')),
    FOREIGN KEY (season_id) REFERENCES seasons(season_id),
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);
CREATE INDEX idx_games_season ON games(season_id);
CREATE INDEX idx_games_date ON games(game_date);

CREATE TABLE player_game_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    minutes REAL,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    fg_made INTEGER,
    fg_attempted INTEGER,
    fg3_made INTEGER,
    fg3_attempted INTEGER,
    ft_made INTEGER,
    ft_attempted INTEGER,
    plus_minus INTEGER,
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
CREATE INDEX idx_pgs_game ON player_game_stats(game_id);
CREATE INDEX idx_pgs_player ON player_game_stats(player_id);

CREATE TABLE player_season_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    season_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    games_played INTEGER,
    games_started INTEGER,
    minutes_per_game REAL,
    points_per_game REAL,
    rebounds_per_game REAL,
    assists_per_game REAL,
    steals_per_game REAL,
    blocks_per_game REAL,
    turnovers_per_game REAL,
    fg_pct REAL,
    fg3_pct REAL,
    ft_pct REAL,
    true_shooting_pct REAL,                  -- Advanced
    effective_fg_pct REAL,                   -- Advanced
    usage_rate REAL,                         -- Advanced
    player_efficiency_rating REAL,           -- Advanced
    win_shares REAL,                         -- Advanced
    box_plus_minus REAL,                     -- Advanced
    vorp REAL,                               -- Advanced
    UNIQUE(player_id, season_id, team_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (season_id) REFERENCES seasons(season_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
CREATE INDEX idx_pss_season ON player_season_stats(season_id);
CREATE INDEX idx_pss_player ON player_season_stats(player_id);

CREATE TABLE team_season_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    season_id TEXT NOT NULL,
    wins INTEGER,
    losses INTEGER,
    points_per_game REAL,
    opp_points_per_game REAL,
    pace REAL,                               -- Advanced
    offensive_rating REAL,                   -- Advanced
    defensive_rating REAL,                   -- Advanced
    net_rating REAL,                         -- Advanced
    UNIQUE(team_id, season_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (season_id) REFERENCES seasons(season_id)
);

-- ============================================
-- UNSTRUCTURED DATA: Articles for RAG
-- ============================================

CREATE TABLE articles (
    article_id TEXT PRIMARY KEY,             -- slug, e.g. 'ctg-four-factors'
    title TEXT NOT NULL,
    source TEXT NOT NULL,                    -- 'cleaning-the-glass', 'thinking-basketball', etc.
    author TEXT,
    url TEXT NOT NULL,
    publish_date TEXT,                       -- ISO 8601, nullable
    full_text TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL                -- ISO 8601 timestamp
);

CREATE TABLE article_chunks (
    chunk_id TEXT PRIMARY KEY,               -- e.g. 'ctg-four-factors#3'
    article_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,            -- 0-based position in article
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    FOREIGN KEY (article_id) REFERENCES articles(article_id)
);
CREATE INDEX idx_chunks_article ON article_chunks(article_id);

-- Vector embeddings (via sqlite-vec extension)
CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[1024]                    -- Voyage-3 dimensions (adjust per model)
);

-- ============================================
-- SYSTEM TABLES
-- ============================================

CREATE TABLE ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    source TEXT NOT NULL,
    records_added INTEGER NOT NULL,
    notes TEXT
);
```

### Pydantic Models (core types)

```python
# src/rageval/types.py

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """The three routing categories."""
    FACTUAL = "factual"          # Answerable via SQL
    ANALYTICAL = "analytical"    # Answerable via RAG
    HYBRID = "hybrid"            # Needs both
    UNANSWERABLE = "unanswerable"  # Out of scope


class Document(BaseModel):
    """A retrievable chunk of text."""
    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SQLResult(BaseModel):
    """Output of a SQL path."""
    query: str
    rows: list[dict[str, Any]]
    error: str | None = None


class TestCase(BaseModel):
    """A single evaluation case."""
    id: str
    question: str
    question_type: QuestionType

    # For factual/hybrid: what the SQL answer should be
    expected_sql_rows: list[dict[str, Any]] | None = None
    expected_numeric: float | None = None
    numeric_tolerance: float = 0.01

    # For analytical/hybrid: which documents should be retrieved
    relevant_doc_ids: list[str] = Field(default_factory=list)

    # For all: the ideal final answer (prose)
    expected_answer: str | None = None

    # For adversarial: should the system refuse?
    should_refuse: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)


class TestSuite(BaseModel):
    """A collection of test cases + metadata."""
    name: str
    description: str = ""
    cases: list[TestCase]

    @classmethod
    def from_yaml(cls, path: str) -> "TestSuite":
        """Load from YAML file."""
        ...


class RAGResponse(BaseModel):
    """What a system-under-test returns."""
    answer: str
    retrieved_docs: list[Document] = Field(default_factory=list)
    sql_result: SQLResult | None = None
    routing_decision: QuestionType | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    refused: bool = False


class RAGSystem(Protocol):
    """Protocol implemented by systems-under-test."""
    async def answer(self, question: str) -> RAGResponse: ...


class MetricResult(BaseModel):
    """Score from running one metric on one case."""
    metric_name: str
    case_id: str
    value: float
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class CaseResult(BaseModel):
    """Aggregated results for one test case."""
    case_id: str
    question: str
    response: RAGResponse
    metric_results: list[MetricResult]


class EvaluationResult(BaseModel):
    """Complete results of an evaluation run."""
    suite_name: str
    system_name: str
    run_at: datetime
    case_results: list[CaseResult]
    aggregate_scores: dict[str, float]       # metric_name -> mean
    total_cost_usd: float
    total_duration_seconds: float
    errors: list[str] = Field(default_factory=list)
```

### Test Suite YAML Format

```yaml
# examples/nba_test_suite.yaml
name: nba-hybrid-demo
description: |
  40 test cases across factual, analytical, hybrid, and adversarial categories.
  Built against 2020-2025 NBA data snapshot and a curated corpus of ~40 analytics articles.

cases:
  # --- FACTUAL (SQL path) ---
  - id: factual-001
    question: Who led the NBA in points per game in the 2023-24 regular season?
    question_type: factual
    expected_sql_rows:
      - player_name: Luka Dončić
        points_per_game: 33.9
    expected_answer: |
      Luka Dončić led the NBA in scoring during the 2023-24 regular season
      with 33.9 points per game.

  - id: factual-002
    question: What was Nikola Jokić's true shooting percentage in 2022-23?
    question_type: factual
    expected_numeric: 0.701
    numeric_tolerance: 0.005
    expected_answer: |
      Nikola Jokić posted a .701 true shooting percentage in 2022-23.

  # --- ANALYTICAL (RAG path) ---
  - id: analytical-001
    question: What are the "four factors" in basketball analytics?
    question_type: analytical
    relevant_doc_ids:
      - ctg-four-factors#0
      - ctg-four-factors#1
      - bbref-glossary#four-factors
    expected_answer: |
      The four factors, popularized by Dean Oliver, are the four statistical
      categories that most strongly predict winning: effective field goal
      percentage, turnover rate, offensive rebounding rate, and free throw
      rate. They can be measured for both offense and defense.

  # --- HYBRID (both paths) ---
  - id: hybrid-001
    question: |
      Jokić is often called a historic offensive talent. What do the
      stats support this and how do analysts describe his game?
    question_type: hybrid
    expected_sql_rows:
      - player_name: Nikola Jokić
        metric: true_shooting_pct
        # (stats subset — evaluator uses "contains" match for hybrid)
    relevant_doc_ids:
      - thinking-basketball-jokic-offense#2
      - ctg-jokic-passing#0
    expected_answer: |
      Statistically, Jokić has posted elite efficiency numbers...
      Analysts describe his game as unique because...

  # --- ADVERSARIAL ---
  - id: adversarial-001
    question: Who will win MVP in the 2027-28 season?
    question_type: unanswerable
    should_refuse: true
    expected_answer: |
      The system should decline to predict future MVPs and note that
      this is outside its knowledge scope.
```

---

## Prompt Structure

All prompts live in `prompts/` as versioned text files. Format: `prompts/{component}/v{N}.txt`. Keep every version for the git history — this becomes your `prompt_evolution.md`.

**Router prompt skeleton:**
```
You classify NBA questions into one of four categories:
- FACTUAL: Needs specific stats (PPG, win totals, shooting percentages)
- ANALYTICAL: Needs conceptual/qualitative analysis
- HYBRID: Needs both stats AND analysis
- UNANSWERABLE: Outside corpus scope (future predictions, opinions, personal info)

Think step by step, then output JSON: {"reasoning": "...", "category": "..."}
```

**SQL agent prompt skeleton:**
```
You are a SQL generator for an NBA stats database.

SCHEMA: {schema_ddl}
QUESTION: {question}

Rules:
- Output only SELECT queries. Never INSERT/UPDATE/DELETE.
- Prefer aggregate queries over row listings when the question asks for leaders/ranks.
- If the question cannot be answered from the schema, return {"error": "reason"}.

Output JSON: {"reasoning": "...", "sql": "...", "expected_row_shape": "..."}
```

**Synthesizer prompt skeleton:**
```
You answer NBA questions using provided sources.

QUESTION: {question}
SQL RESULTS: {sql_json}
RETRIEVED ARTICLES: {chunks}

Rules:
- Use ONLY information from the provided sources.
- If sources are insufficient, say so explicitly.
- Cite claims inline as [sql] or [article:id].
- Be concise.
```

**Faithfulness judge prompt skeleton:**
```
You evaluate whether an ANSWER is faithful to SOURCES.

QUESTION: {question}
SOURCES: {sources}
ANSWER: {answer}

A claim is "unsupported" if it is not directly stated or strongly implied by the sources.
General knowledge and common inferences are acceptable.

Think step by step, then output JSON:
{"reasoning": "...", "faithful": bool, "unsupported_claims": [...]}
```

---

## Milestone Plan (Detailed)

Each milestone has: (a) goal, (b) files to create/modify, (c) ship criterion.

### Milestone 0 — Project Setup

**Goal:** Clean repo with modern Python toolchain.

**Tasks:**
- Install `uv` if not present: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- `uv init --lib rageval-nba`
- Configure `pyproject.toml` with deps (see below)
- Add `ruff`, `mypy`, `pytest`, `pytest-asyncio` dev deps
- Create `.gitignore` (Python + `.env` + `.rageval_cache/` + `*.db`)
- Create `.env.example` with `ANTHROPIC_API_KEY=`
- Create directory scaffold (empty `__init__.py` files)
- Add GitHub Actions `.github/workflows/ci.yml`: runs ruff + mypy + pytest on push
- First commit: "initial scaffold"

**Ship criterion:** `uv run pytest` runs (zero tests OK), `uv run ruff check` passes, CI green.

### Milestone 1 — Core Types

**Goal:** All Pydantic models + YAML test suite loader.

**Tasks:**
- Implement `src/rageval/types.py` (full file spec'd above)
- Implement `TestSuite.from_yaml(path)` using `pyyaml`
- Write `tests/test_types.py`: 
  - Round-trip serialize/deserialize each model
  - Load a fixture YAML file
  - Validate that bad YAML raises clear errors

**Ship criterion:** `TestSuite.from_yaml('examples/nba_test_suite.yaml')` returns a fully-typed object. All type tests pass.

### Milestone 2 — Retrieval Metrics

**Goal:** Deterministic IR metrics, bulletproof.

**Tasks:**
- Implement `src/rageval/metrics/base.py` with `Metric` protocol
- Implement `src/rageval/metrics/retrieval.py`:
  - `precision_at_k(retrieved_ids, relevant_ids, k) -> float`
  - `recall_at_k(retrieved_ids, relevant_ids, k) -> float`
  - `reciprocal_rank(retrieved_ids, relevant_ids) -> float`
  - `ndcg_at_k(retrieved_ids, relevant_ids, k) -> float` (use `2^rel - 1` variant, document choice)
- Wrap each as a `Metric` class that takes (TestCase, RAGResponse) and returns MetricResult
- Write `tests/test_retrieval_metrics.py` with 5+ cases per metric: empty, perfect, zero-match, partial, ordering-sensitive

**Ship criterion:** 20+ assertions pass. Hand-computed expected values verified.

### Milestone 3 — LLM Client + Caching

**Goal:** Robust async Anthropic wrapper with disk cache.

**Tasks:**
- Implement `src/rageval/cache.py`:
  - SHA256 key from (model, system, user, temperature, tool_schema)
  - Store as JSON in `.rageval_cache/{hash[:2]}/{hash}.json`
  - Add `--no-cache` flag support
- Implement `src/rageval/llm_client.py`:
  - `LLMClient` class with async `complete(system, user, model, temperature, tools)` method
  - Exponential backoff for 429s, respect `retry-after`
  - `asyncio.Semaphore` concurrency limiter (default 5)
  - Token + cost tracking per call (Anthropic returns usage in response)
- Write `tests/test_llm_client.py` with mocked transport (httpx MockTransport)

**Ship criterion:** Can call `await client.complete(...)`, get response, call again → instant (cache hit). Cost tracker accurate.

### Milestone 3.5 — Stats Database

**Goal:** Populated SQLite with NBA stats 2020-2025.

**Tasks:**
- `scripts/build_stats_db.py`:
  - Use `nba_api` package
  - Pull: all players, teams, games, player/team season stats for 2020-21 through 2024-25
  - Create schema (see SQL above)
  - Insert with transaction, idempotent
  - Save raw API responses to `data/raw/` as JSON (for reproducibility)
- Add retry logic — nba_api is flaky
- Log to `ingestion_log` table
- Expected: ~1500 players, ~30 teams, ~5000 games, ~7500 player-season rows

**Ship criterion:** `uv run python scripts/build_stats_db.py` runs to completion. Can query `SELECT player_name, points_per_game FROM player_season_stats JOIN players USING(player_id) WHERE season_id='2023-24' ORDER BY points_per_game DESC LIMIT 5` and get sensible results.

### Milestone 3.75 — Router + Text-to-SQL

**Goal:** Classifier and SQL generator components.

**Tasks:**
- Implement `src/rageval/demo/router.py`:
  - `Router.classify(question) -> QuestionType`
  - Uses `LLMClient` with tool-use for structured output
  - Prompt in `prompts/router/v1.txt`
- Implement `src/rageval/demo/sql_agent.py`:
  - `SQLAgent.generate_and_execute(question) -> SQLResult`
  - Schema provided in system prompt
  - Validates SELECT-only before execution (safety)
  - Handles errors gracefully (malformed SQL, no results, too many results)
  - Prompt in `prompts/sql_agent/v1.txt`
- Smoke test: router classifies 10 sample questions correctly; SQL agent answers 5 factual questions

**Ship criterion:** `await router.classify("Who led in points last season?")` returns `FACTUAL`. `await sql_agent.generate_and_execute(...)` returns actual correct stats.

### Milestone 4 — First LLM Judge (Faithfulness)

**Goal:** One working judge, end-to-end, calibrated.

**Tasks:**
- Implement `src/rageval/metrics/judge.py` → `FaithfulnessJudge`
- Prompt in `prompts/judges/faithfulness/v1.txt`
- Use Anthropic tool-use for structured JSON output
- Build `tests/fixtures/faithfulness_calibration.yaml`: 10 (question, sources, answer, human_label) tuples
- `scripts/calibrate_judge.py faithfulness` runs judge, reports agreement rate
- Document in `docs/judge_calibration.md`
- Iterate prompt until agreement ≥ 80%, save each version

**Ship criterion:** Judge agrees with hand labels 8/10 or better. Versioned prompts in `prompts/judges/faithfulness/`.

### Milestone 5 — Complete Judge Suite

**Goal:** Relevance, correctness, and routing-accuracy judges. Bias mitigation.

**Tasks:**
- `RelevanceJudge`, `CorrectnessJudge`, `RoutingJudge` following same pattern
- `CorrectnessJudge` uses 0-4 scale, not binary
- Implement **position-swap** bias mitigation for correctness: run twice with positions swapped, flag disagreements
- Calibrate each against 10 hand labels
- Update `docs/judge_calibration.md`

**Ship criterion:** Four judges, all calibrated ≥ 80% agreement, position-swap mitigation working for correctness.

### Milestone 6 — Evaluator Orchestrator

**Goal:** One function runs a suite against a system, returns structured results.

**Tasks:**
- Implement `src/rageval/evaluator.py`:
  - `Evaluator(metrics, max_concurrent=5)`
  - `async evaluate(system, suite) -> EvaluationResult`
  - Parallelizes across cases with `asyncio.gather` + semaphore
  - Graceful per-case error handling
  - `rich` progress bar (live updating, shows ETA and cost)
- Write `tests/test_evaluator.py` using a fake `RAGSystem` that returns canned responses

**Ship criterion:** `result = await evaluator.evaluate(fake_system, fixture_suite)` completes, returns populated `EvaluationResult`.

### Milestone 6.5 — Structured Metrics

**Goal:** Metrics for the SQL path.

**Tasks:**
- Implement `src/rageval/metrics/structured.py`:
  - `ExactMatchMetric` — answer string matches expected exactly (normalized)
  - `NumericToleranceMetric` — extracts number from answer, compares with tolerance
  - `SQLEquivalenceMetric` — compares `sql_result.rows` to `expected_sql_rows` (set equality, order-insensitive)
  - `RefusalMetric` — for adversarial cases, scores whether system refused appropriately
- Write tests

**Ship criterion:** All structured metrics pass tests. Handle edge cases: missing sql_result, partial matches, number extraction failures.

### Milestone 7 — Reference Hybrid RAG System + Test Suite

**Goal:** The demo system + 40 handcrafted test cases.

**Tasks:**
- `scripts/build_corpus.py`: scrape ~40 analytics articles, chunk, embed (use Voyage-3 or Anthropic embeddings), store in `article_chunks` + `chunk_embeddings`
  - Respect robots.txt, rate-limit scraping, store URLs not full content if licensing unclear
- Implement `src/rageval/demo/rag_agent.py`: vector search, returns top-K chunks
- Implement `src/rageval/demo/synthesizer.py`: takes question + SQL result + retrieved chunks, generates final answer with inline citations
- Implement `src/rageval/demo/system.py` → `HybridRAGSystem` that implements the `RAGSystem` protocol
- Handcraft `examples/nba_test_suite.yaml`: 40 cases (12 factual, 15 analytical, 10 hybrid, 5 adversarial)
- Run full evaluator against the demo system. Debug. Iterate.

**Ship criterion:** `uv run rageval run examples/nba_test_suite.yaml` runs end-to-end, producing a valid `EvaluationResult` with reasonable scores (aim for correctness ≥ 70% on first pass).

### Milestone 8 — HTML Report + CLI

**Goal:** Beautiful demo-able output.

**Tasks:**
- Implement `src/rageval/reporting.py`:
  - Jinja2 template in `src/rageval/templates/report.html.j2`
  - Sections: header (suite, timestamp, cost, duration), summary table (metrics × aggregate scores), per-category breakdown (factual/analytical/hybrid/adversarial), per-case drill-down (collapsible), disagreement panel (interesting failure modes)
  - Use Pico.css (CDN, zero config) for styling
  - Chart.js (CDN) for bar chart of scores and scatter of faithfulness vs correctness
- Implement `src/rageval/cli.py` with Click:
  - `rageval run <suite.yaml> --output report.html [--metrics ...] [--max-cases N] [--no-cache]`
  - `rageval calibrate <judge_name>`
  - `rageval demo` (runs a sample with 5 cases, fast feedback)
  - `--verbose` mode shows live case-by-case output
- Expose via `pyproject.toml` `[project.scripts]`

**Ship criterion:** `rageval run examples/nba_test_suite.yaml --output report.html` produces a demo-able HTML file. Screenshot-worthy.

### Milestone 9 — README, Calibration Docs, Publish

**Goal:** The artifact hiring managers click on.

**Tasks:**
- Write `README.md`:
  - One-paragraph pitch + hero screenshot of report
  - Quickstart (uv install + 10 lines of code + link to generated report)
  - "Why hybrid RAG is different" (2 paragraphs)
  - "Metrics explained" table
  - "LLM-as-judge caveats" with links to Zheng et al. 2023
  - Architecture diagram (Mermaid)
  - Calibration results table
  - Roadmap
- Write `docs/prompt_evolution.md`: why each prompt changed between versions
- Publish to PyPI: `uv build && uv publish` (requires PyPI token)
- Tag `v0.1.0`, write release notes
- Record 90-second Loom: scroll report, explain one finding, show one code snippet
- Pin repo on GitHub profile
- Update resume with project line

**Ship criterion:** Stranger lands on README → understands project in 10s → installs in 30s → demo runs in <2 min.

---

## pyproject.toml (starter)

```toml
[project]
name = "rageval-nba"
version = "0.1.0"
description = "Evaluation harness for hybrid RAG systems, with an NBA analytics reference implementation."
authors = [{name = "Daniel Keinan", email = "dk3425@columbia.edu"}]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "pydantic>=2.9.0",
    "httpx>=0.27.0",
    "rich>=13.9.0",
    "jinja2>=3.1.0",
    "click>=8.1.0",
    "pyyaml>=6.0",
    "sqlite-vec>=0.1.0",
    "nba-api>=1.7.0",
    "beautifulsoup4>=4.12.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.7.0",
    "mypy>=1.11.0",
    "types-pyyaml",
]

[project.scripts]
rageval = "rageval.cli:main"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "RET"]

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Dependency Cheat Sheet

| Package | Why |
|---|---|
| `anthropic` | Claude API client |
| `pydantic` v2 | Data validation, config, schemas |
| `httpx` | Async HTTP (anthropic uses it under the hood) |
| `sqlite-vec` | Vector extension for SQLite |
| `nba-api` | Pull NBA data |
| `rich` | Beautiful terminal output, progress bars |
| `jinja2` | HTML report templating |
| `click` | CLI framework |
| `pyyaml` | Load test suites |
| `beautifulsoup4` | Article scraping |
| `python-dotenv` | Load `.env` |
| `pytest` + `pytest-asyncio` | Testing |
| `ruff` | Linting + formatting (one tool) |
| `mypy` | Type checking |

---

## How to Use This Plan With Claude Code

This document is your working spec. When starting a milestone, feed the relevant section to Claude Code in VS Code. Suggested workflow:

1. **Keep this file in the repo** as `PROJECT_PLAN.md`.
2. **Start each session by pointing Claude Code at the current milestone section.** Example prompt: "Read `PROJECT_PLAN.md`, focus on Milestone 3. Implement the LLM client and cache per the spec. Write tests. Don't implement anything outside this milestone's scope."
3. **After each milestone, commit with a semantic message** like `feat: milestone 3 - LLM client with caching`.
4. **Update the plan if you change scope** — keep it honest. It's a live document, and its evolution is itself interesting for your README.
5. **When stuck, paste the relevant section + the error** into Claude Code. The schema and prompts are detailed enough that Claude Code should have strong grounding.

A useful mental model: this doc is the contract, Claude Code is the implementer, you're the reviewer and integrator. Don't let Claude Code drift from the plan without a reason you understand.

---

## Open Questions to Decide Later

These don't block starting — park them and decide as you go:

- **Embedding model:** Voyage-3 (best quality/$), OpenAI text-embedding-3-small (cheapest), or Anthropic's embedding endpoint? → Decide in Milestone 7.
- **Hosting for demo:** Just GitHub and local? Or deploy the HTML report to GitHub Pages? → GitHub Pages is a nice touch for Milestone 9.
- **License:** MIT vs Apache 2.0. → MIT unless you have opinions; simpler.
- **Article scraping ethics:** Store full text vs store URLs + re-fetch? → Store URLs + keep full text in gitignored local cache for reproducibility. Document in README.

---

## Success Metrics (for you, not the harness)

This project has succeeded if:

1. It runs end-to-end on a fresh clone with `uv sync && uv run rageval demo`.
2. The README is good enough that a hiring manager who scrolls for 30 seconds understands what you built and why it's interesting.
3. You can explain any metric in the harness in 60 seconds.
4. You can walk through one interesting finding from the evaluation (e.g., "my router confuses hybrid questions as analytical 40% of the time, which surfaced a prompt bug").
5. The calibration table shows you took the LLM-as-judge bias problem seriously.
6. The git history shows steady, meaningful commits over 2+ weeks.

If those six are true, this project materially changes how your resume reads.
