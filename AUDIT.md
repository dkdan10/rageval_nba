# Audit — rageval-nba vs PROJECT_PLAN.md

Snapshot date: 2026-04-26. Branch: `main`. Read-only audit; no code/configs/tests modified.

---

## 1. Current file tree (top 3 levels)

Excluded: `.venv/`, `__pycache__/`, `.rageval_cache/`, `.pytest_cache/`, `.mypy_cache/`,
`.ruff_cache/`, `.DS_Store`, `data/raw/` contents (kept folder), `dist/` contents (kept folder).

```
.
├── .env.example
├── .github/workflows/ci.yml
├── .gitignore
├── AUDIT.md                           (this file)
├── LICENSE
├── PROJECT_PLAN.md
├── README.md
├── pyproject.toml
├── uv.lock
├── demo-report.html                   (generated artifact, gitignored)
├── report.html                        (generated artifact, gitignored)
├── data/
│   ├── nba.db                         (gitignored; current copy is seed mode)
│   └── raw/                           (gitignored; nba_api JSON cache + corpus HTML)
├── dist/                              (built wheel + sdist)
├── docs/
│   ├── assets/report-preview.png
│   ├── judge_calibration.md
│   └── prompt_evolution.md
├── examples/
│   ├── corpus/articles.json
│   ├── data/                          (empty placeholder)
│   └── nba_test_suite.yaml
├── prompts/
│   ├── judges/{correctness,faithfulness,relevance,routing}/v1.txt
│   ├── router/v1.txt
│   ├── sql_agent/v1.txt
│   └── synthesizer/v1.txt
├── scripts/
│   ├── build_corpus.py
│   ├── build_stats_db.py
│   └── calibrate_judge.py
├── src/rageval/
│   ├── __init__.py
│   ├── cache.py
│   ├── cli.py
│   ├── evaluator.py
│   ├── llm_client.py
│   ├── reporting.py
│   ├── types.py
│   ├── demo/
│   │   ├── rag_agent.py
│   │   ├── router.py
│   │   ├── sql_agent.py
│   │   ├── synthesizer.py
│   │   └── system.py
│   ├── examples/nba_test_suite.yaml   (bundled copy for `rageval demo`)
│   ├── metrics/
│   │   ├── base.py
│   │   ├── judge.py
│   │   ├── retrieval.py
│   │   └── structured.py
│   └── templates/report.html.j2
└── tests/
    ├── conftest.py                    (empty)
    ├── fixtures/{correctness,faithfulness,relevance,routing}_calibration.yaml
    ├── fixtures/valid_suite.yaml
    └── test_*.py                       (16 test files)
```

Notable: `tests/fixtures/` is the calibration set location, *not* the planned
`tests/fixtures/faithfulness_calibration.yaml` only — there are four fixtures plus a
`valid_suite.yaml`. The plan's `examples/corpus/` exists but contains only `articles.json`
(metadata manifest); raw HTML lives under `data/raw/corpus/`.

---

## 2. Milestone-by-milestone checklist

### Milestone 0 — Project Setup
- ✅ `uv` toolchain, `pyproject.toml` matches spec (deps, dev deps, scripts, ruff/mypy/pytest).
- ✅ `.gitignore` covers `__pycache__`, `.env`, `.rageval_cache/`, `*.db`, `data/raw/`, etc.
- ✅ `.env.example` with `ANTHROPIC_API_KEY=`.
- ✅ Directory scaffold present.
- ✅ `.github/workflows/ci.yml` runs ruff + mypy + pytest on push/PR.
- ✅ Ship criterion met: `uv run pytest`, `ruff check`, `mypy src/` all green locally.

### Milestone 1 — Core Types
- ✅ `src/rageval/types.py` defines `QuestionType`, `Document`, `SQLResult`, `TestCase`,
  `TestSuite`, `RAGResponse`, `RAGSystem`, `MetricResult`, `CaseResult`, `EvaluationResult`.
- ✅ `TestSuite.from_yaml` raises clear `FileNotFoundError`/`ValueError` on bad input.
- 🟡 `CaseResult` adds `question_type: QuestionType | None = None` not in the original
  spec — necessary for category breakdown but a quiet schema deviation. Note in §3.
- ✅ `tests/test_types.py` covers round-trip and YAML loader cases.

### Milestone 2 — Retrieval Metrics
- ✅ `precision_at_k`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k` implemented.
- ✅ Wrapped as `PrecisionAtK`, `RecallAtK`, `ReciprocalRank`, `NDCGAtK` callable classes.
- 🟡 `Metric` protocol exists in [src/rageval/metrics/base.py](src/rageval/metrics/base.py)
  but expects sync `__call__`; the evaluator and judges separately use a duck-typed
  `evaluate()` async protocol. The base.py protocol is effectively unused.
- ✅ `tests/test_retrieval_metrics.py` has 5+ cases per metric.

### Milestone 3 — LLM Client + Caching
- ✅ `cache.py` SHA256-keys on (model, system, user, temperature, tool_schema) and writes to
  `.rageval_cache/{hash[:2]}/{hash}.json`.
- ✅ `llm_client.py` has async `LLMClient.complete`, `asyncio.Semaphore(5)` default, retry on
  `RateLimitError` honoring `retry-after`, token+cost tracking.
- 🟡 `--no-cache` is honored at the LLMClient level, but in the CLI’s default `run`/`demo`
  paths no LLM calls occur (see §3) — the flag is effectively a no-op there. Honest message
  is printed in verbose mode.
- 🟡 No `tests/test_llm_client.py` use of `httpx MockTransport` as the plan describes;
  tests instead patch `anthropic.AsyncAnthropic.messages.create`. Functionally equivalent.

### Milestone 3.5 — Stats Database
- 🟡 Schema matches the plan, including `articles`, `article_chunks`, `ingestion_log`. 
  `chunk_embeddings` virtual table is created best-effort if `sqlite-vec` loads.
- 🟡 `scripts/build_stats_db.py` ships **two** modes: `seed` (default, 9 players / 3 teams /
  4 games / 14 player-season rows / 6 team-season rows / 11 player-game rows) and `real`
  (nba_api). The currently committed/built `data/nba.db` is the **seed** dataset, not the
  ~992 players / 6,417 games / 136K player-game rows the plan promised. `data/raw/` is
  populated with full nba_api JSON, so a `--mode real` rebuild is plausible without
  re-fetching. Counts are well below the plan's "verified 2026-04-24" expectation —
  unverified whether a real-mode build was ever completed against this checkout.
- ✅ `ingestion_log` row written on each build. Real mode logs counts in notes.
- 🟡 Plan says `player_efficiency_rating`, `win_shares`, `box_plus_minus`, `vorp` are left
  NULL — code matches; this is intentionally documented in-script.

### Milestone 3.75 — Router + Text-to-SQL
- ✅ `src/rageval/demo/router.py` — `Router.classify` uses `LLMClient` tool-use, prompt at
  `prompts/router/v1.txt`.
- ✅ `src/rageval/demo/sql_agent.py` — `SQLAgent.generate_and_execute` uses tool-use,
  enforces SELECT-only via regex allowlist + comment stripping, has `_MAX_ROWS=500` cap.
- 🟡 The plan's smoke test ("router classifies 10 sample questions correctly; SQL agent
  answers 5 factual questions") — **unverified**. No live smoke evidence is recorded in
  this repo; tests cover the components only with mocked LLM responses.

### Milestone 4 — Faithfulness Judge
- ✅ `FaithfulnessJudge` implemented with tool-use, prompt versioned.
- ✅ Calibration fixture has 10 hand-labeled cases (5 faithful / 5 unfaithful).
- ✅ `scripts/calibrate_judge.py` and `docs/judge_calibration.md` present.
- ✅ Documented agreement: 100% (10/10) on 2026-04-25 — meets ≥80% bar.
  Authenticity is **unverified** in this audit (live calibration would require API key).

### Milestone 5 — Complete Judge Suite
- ✅ `RelevanceJudge`, `CorrectnessJudge`, `RoutingJudge` implemented.
- ✅ Correctness uses 0–4 scale and runs forward + swapped passes; details surface
  `forward_score`, `swapped_score`, `disagreement`, `disagreement_flag`.
- ⊘ `RoutingJudge` is intentionally deterministic (no LLM); plan was ambiguous, repo
  documents the choice in `prompts/judges/routing/v1.txt`'s placeholder and the docs.
- ✅ All four judges report ≥80% agreement (correctness at exactly 80%, on the line).

### Milestone 6 — Evaluator Orchestrator
- ✅ `Evaluator(metrics, max_concurrent=5)` runs `asyncio.gather` over cases under a
  semaphore, returns `EvaluationResult` with aggregate scores.
- ✅ Per-case errors caught; broken metric calls excluded from aggregate means.
- 🟡 No `rich` progress bar — the plan called for "live updating, shows ETA and cost".
  CLI verbose mode prints one line per completed case after the run, not live.
- ✅ `tests/test_evaluator.py` exercises fake systems and canned responses.

### Milestone 6.5 — Structured Metrics
- ✅ `ExactMatchMetric`, `NumericToleranceMetric`, `SQLEquivalenceMetric`, `RefusalMetric`
  in [src/rageval/metrics/structured.py](src/rageval/metrics/structured.py).
- ✅ Hybrid cases use containment match for SQL.
- 🟡 `ExactMatchMetric` is implemented and tested but **never registered in the CLI
  default metric set** (only NumericTolerance/SQLEquivalence/Refusal/prefix retrieval).
  Plan didn't strictly require wiring, but the metric is dead in practice.

### Milestone 7 — Reference Hybrid RAG System + Test Suite
- 🟡 `src/rageval/demo/system.py` — `HybridRAGSystem` exists and wires Router, SQLAgent,
  RAGAgent, Synthesizer correctly.
- 🟡 `RAGAgent` uses a **deterministic lexical token-overlap retriever** with a
  log-length penalty — not vector search, no embeddings used. `chunk_embeddings` is
  unpopulated. The plan called for Voyage-3 / Anthropic embeddings + vector search.
- 🟡 `Synthesizer` defaults to a **deterministic stitcher** when no `LLMClient` is
  injected; the LLM path is implemented but not used by the CLI's default path.
- ❌ The CLI’s `_demo_system` (cli.py:339) injects a `_SuiteRouter` (routes by suite
  question→label lookup) and `_DemoSQLAgent` (hardcoded `if/elif` SQL literal-row
  responses). The actual `Router` and `SQLAgent` LLM components are bypassed end-to-end
  in `rageval run` and `rageval demo`. This is the single most consequential plan
  deviation. See §3.
- 🟡 `examples/nba_test_suite.yaml` has **42** cases (12 factual / 15 analytical /
  10 hybrid / 5 unanswerable), close to the plan's 40 (12/15/10/5 = 42 in the plan
  too if you add up; the prose said "40" but breakdown sums to 42).
- 🟡 Plan ship criterion is "reasonable scores (correctness ≥ 70% on first pass)" —
  unverified; the harness produces deterministic answers from the demo path so
  correctness is not exercised end-to-end.

### Milestone 8 — HTML Report + CLI
- ✅ `src/rageval/reporting.py` + `templates/report.html.j2` — Pico.css, Chart.js, summary
  cards, per-category breakdown, failure-mode panel, per-case drilldowns.
- ✅ `rageval run`, `rageval demo`, `rageval calibrate`, `rageval version` Click commands.
  `--metrics`, `--max-cases`, `--verbose`, `--no-cache` flags exposed.
- 🟡 `--verbose` does not stream live; it prints a one-line summary per case after the
  full run completes.
- ✅ Console script wired via `[project.scripts] rageval = "rageval.cli:main"`.

### Milestone 9 — README, Calibration Docs, Publish
- ✅ README has pitch, hero screenshot, quickstart, architecture diagram (Mermaid not the
  ASCII variant in plan but acceptable), metrics table, calibration table, caveats with
  Zheng et al. 2023 link, roadmap.
- ✅ `docs/judge_calibration.md` and `docs/prompt_evolution.md` exist.
- ✅ `dist/rageval_nba-0.1.0-py3-none-any.whl` and sdist present, indicating `uv build`
  succeeded.
- ❌ No `v0.1.0` git tag and no release notes file in the repo.
- ❌ No 90-second Loom or video reference in README — explicitly listed in "Roadmap" as
  not done.

---

## 3. Key architectural deviations

### D1. The CLI demo bypasses its own LLM components — **regression**
[src/rageval/cli.py:339-497](src/rageval/cli.py#L339-L497) constructs the demo
`HybridRAGSystem` with hand-written stubs:

- `_SuiteRouter` (cli.py:492) — builds a `{question: question_type}` dict from the test
  suite itself and returns the label as the route. Routing accuracy is therefore 100% by
  construction whenever the question text exactly matches the suite, and `UNANSWERABLE`
  otherwise. The real `Router` (`src/rageval/demo/router.py`) is never called by the CLI.
- `_DemoSQLAgent` + `_demo_sql_for_question` (cli.py:500-585) — a long `if/elif` chain
  matching keywords like `"luka"`, `"joki"`, `"celtics"`, returning hardcoded
  `SELECT 33.9 AS points_per_game ...` literal-row queries. The real `SQLAgent` LLM path
  (`src/rageval/demo/sql_agent.py`) is never called by the CLI.
- `Synthesizer` is constructed with `llm=None`, so it uses
  `_deterministic_answer` (synthesizer.py:65) — a string-stitcher over SQL rows and the
  first three retrieved docs. The real LLM synthesis path is dead in CLI usage.

Net effect: `rageval demo` and `rageval run` evaluate a hand-curated lookup table, not
a hybrid RAG system. The reference HybridRAGSystem the plan promises *exists in code* but
is never the system-under-test in the demo CLI. Faithfulness/Relevance/Correctness judges
require `--metrics` to be selectable and aren't even in the CLI's default metric set
([cli.py:264](src/rageval/cli.py#L264)).

Why this matters: this is the central artifact for the README pitch. A reviewer running
`uv run rageval demo` is being shown deterministic plumbing rated against deterministic
output — a tautology. Correctness ≥70% on first pass (M7 ship criterion) is impossible to
verify because the LLM synthesis path never executes. The judge calibration numbers in
the README are real, but they describe the *judges in isolation*, not their behavior on
real demo answers.

### D2. Stats DB defaults to a 9-player seed — **regression vs plan**
The plan's M3.5 ship criterion ("`SELECT … FROM player_season_stats … LIMIT 5` returns
sensible results") and the explicit verified counts ("992 players, 30 teams, 6,417 games,
… 136,548 player-game rows") imply the real ingestion path is the default. In practice,
`scripts/build_stats_db.py` defaults to `--mode seed` (build_stats_db.py:1098), and the
checked-in `data/nba.db` was built in seed mode (ingestion_log shows source=`seed`,
records_added=49). The "Luka led with 33.9 PPG" answer is therefore not from real data —
it's the literal-row stub from D1. `--mode real` works (raw JSON cached under
`data/raw/`), but the audit cannot confirm it has been run against the current checkout.

### D3. RAGAgent is lexical, not vector — **deviation, neutral-to-mild regression**
[src/rageval/demo/rag_agent.py:54-104](src/rageval/demo/rag_agent.py#L54-L104) implements
a token-overlap scorer with a `log2(len)` length penalty. The plan called for Voyage-3
(or Anthropic) embeddings and `sqlite-vec` search. The `chunk_embeddings` table is
created when the extension loads but is never populated. Documented honestly in the
RAGAgent docstring ("intentionally avoids paid embedding calls") and README, so this is
a known scope cut — not silent. Neutral if framed as a simplification, regression versus
plan ship criteria for M7.

### D4. Synthesizer LLM path exists but is dormant — **regression**
[src/rageval/demo/synthesizer.py:22-46](src/rageval/demo/synthesizer.py#L22-L46) accepts
an `LLMClient` and will use it if injected. Nothing in the codebase injects one for the
default demo. Unit tests cover both paths, but the deployed behavior is the deterministic
stitcher.

### D5. Routing judge is deterministic by design — **neutral**
Plan §M5 named four judges, implying all four are LLM-backed. Repo intentionally
implements `RoutingJudge` as a non-LLM equality check
([src/rageval/metrics/judge.py:377-395](src/rageval/metrics/judge.py#L377-L395)).
Documented choice; sound reasoning (fixed-label classification doesn't need an LLM
judge). Neutral.

### D6. Evaluator has no live progress bar — **mild regression**
Plan called for `rich`-based live ETA and cost display. Implementation prints a static
summary at the end with optional `--verbose` per-case lines after completion
([src/rageval/cli.py:214-228](src/rageval/cli.py#L214-L228)). `rich` is in dependencies
but unused.

### D7. `CaseResult.question_type` added beyond the spec — **improvement**
[src/rageval/types.py:122-123](src/rageval/types.py#L122-L123) adds
`question_type: QuestionType | None = None`, used by `_category_breakdown` in reporting.
A clean schema extension; backwards-compatible because it's optional. Not flagged in
prompt_evolution.md or judge_calibration.md.

### D8. `ExactMatchMetric` is implemented but unwired in the CLI — **neutral**
Defined in `metrics/structured.py:61`, exported, tested. Not in the default CLI metric
set ([cli.py:264-273](src/rageval/cli.py#L264-L273)). Probably correct (exact-match
against narrative `expected_answer` strings would always score 0), but the plan asks for
it without commentary.

### D9. `prompts/judges/routing/v1.txt` is a placeholder — **neutral**
Documented in [docs/prompt_evolution.md](docs/prompt_evolution.md) and the judge's module
docstring. Not loaded at runtime. Honest, but worth noting that it inflates the
"versioned prompts" surface area.

---

## 4. Test coverage and pass rate

Run on 2026-04-26 against current checkout:

| Tool | Result |
|---|---|
| `uv run pytest` | **418 passed**, 0 failed, 0 errors, 0 skipped, 2.51s |
| `uv run ruff check .` | All checks passed (0 issues) |
| `uv run mypy src/` | Success: no issues found in 18 source files |

Notes:

- 16 test files; pytest collected 418 cases.
- All tests use mocked or fake `LLMClient`s — no live Anthropic calls in the suite.
- Coverage of "what would actually fail under live conditions" (real router accuracy,
  real text-to-SQL behavior, real synthesis quality) is **not exercised**. The judge
  calibration documented at 80–100% in `docs/judge_calibration.md` is from a separate
  manual `--no-cache` run (unverified in this audit).

---

## 5. Top 10 concerns / rough edges (ranked)

1. **CLI demo system is a hand-curated lookup, not a RAG system**
   ([src/rageval/cli.py:339-585](src/rageval/cli.py#L339-L585)) — see D1. Anyone running
   `uv run rageval demo` to evaluate the project is rating a stub harness against itself.
   *Fix:* default `_demo_system` to the real `HybridRAGSystem(Router(), SQLAgent(),
   RAGAgent(), Synthesizer(LLMClient()))` and gate the stub behind an `--offline` flag.

2. **`data/nba.db` is the 49-row seed, not real NBA stats**
   ([scripts/build_stats_db.py:1132-1135](scripts/build_stats_db.py#L1132-L1135)) —
   see D2. The "Luka 33.9 PPG" answer in factual-001 comes from a literal-row stub, not
   the database. *Fix:* invert the default to `--mode real`, or document the seed clearly
   and add a script entry like `uv run rageval bootstrap` that runs real ingestion.

3. **`chunk_embeddings` table exists but is empty; RAGAgent never uses it**
   ([src/rageval/demo/rag_agent.py:54-104](src/rageval/demo/rag_agent.py#L54-L104)) —
   token overlap is brittle on short queries and on documents whose article-id text
   shadows content. *Fix:* either drop the embeddings table from schema or wire a real
   embedding step in `scripts/build_corpus.py` and a vector path in RAGAgent.

4. **`--no-cache` flag accepted on `rageval run`/`demo` is a no-op**
   ([src/rageval/cli.py:215-216](src/rageval/cli.py#L215-L216)) — printed only in
   verbose mode. Misleading affordance because no LLM calls happen on those paths today.
   *Fix:* drop the flag from those subcommands until the real system path is wired.

5. **Demo `_demo_sql_for_question` keyword chain is fragile and case-bleeding**
   ([src/rageval/cli.py:529-585](src/rageval/cli.py#L529-L585)) — the `if "joki" in q`
   matches before the more specific `"true shooting"` branch on some inputs; ordering
   matters and is implicit. *Fix:* delete this whole stub when D1 is addressed.

6. **`ExactMatchMetric` is dead code in CLI usage**
   ([src/rageval/metrics/structured.py:61](src/rageval/metrics/structured.py#L61)) — not
   in `_default_metrics()`. Implies completeness it doesn't deliver. *Fix:* either expose
   it via `--metrics exact_match` or delete it; choose deliberately.

7. **`Metric` protocol in `metrics/base.py` is unused**
   ([src/rageval/metrics/base.py](src/rageval/metrics/base.py)) — Evaluator and judges
   share an ad-hoc `evaluate(case, response)` async API that doesn't conform to the
   protocol. *Fix:* either widen the Protocol to match (sync + async, optional `evaluate`
   method) or remove the file.

8. **`prompts/judges/routing/v1.txt` exists only as documentation, not used at runtime**
   ([prompts/judges/routing/v1.txt](prompts/judges/routing/v1.txt)) — easy for a future
   maintainer to assume it's wired. *Fix:* either delete the file (the docs explain the
   choice) or move it to `docs/`.

9. **`prompts/`, `tests/`, `src/rageval/` contain `.DS_Store` files** — committed macOS
   metadata (`prompts/.DS_Store`, `prompts/judges/.DS_Store`, `src/rageval/.DS_Store`,
   `tests/.DS_Store`). `.gitignore` *does* list `.DS_Store`, so these were committed
   before the rule and slipped through. *Fix:* `git rm --cached` on the four files.

10. **`demo-report.html` and `report.html` are tracked in the working tree though
    gitignored** — visible in `ls`, gitignored at `.gitignore:18-19`, but only because
    they were committed once. Real risk: the README's hero claim depends on a stale
    artifact. *Fix:* delete the working-tree copies and regenerate via the CLI.

---

## 6. What's NOT yet built (separate from milestone checklist)

These are gaps that would block a clean v0.1 release framed honestly:

- **End-to-end LLM-driven demo run.** No recorded run of the *real* `HybridRAGSystem` —
  Router + SQLAgent + RAGAgent + Synthesizer with `LLMClient` injected — against the
  test suite. This is the single piece of evidence the README's "evaluates a hybrid RAG
  system" claim depends on. See D1.
- **Real-mode database build verified against current checkout.** `data/raw/` has the
  nba_api JSON cached, but no row counts in the current `data/nba.db` reflect a real
  ingestion. Ship criterion for M3.5 cannot be verified.
- **Vector retrieval / embeddings.** `chunk_embeddings` is unpopulated; no embedder is
  wired. Plan called for Voyage-3 or Anthropic embeddings (M7).
- **Live progress bar / ETA / cost display in evaluator.** `rich` is in deps; not used.
  Plan called this out in M6.
- **`v0.1.0` git tag + release notes.** Plan M9 ship criterion. Wheel and sdist exist
  in `dist/` but no tag.
- **90-second demo video / Loom.** Plan M9; explicitly deferred in README roadmap.
- **Hosted sample report / GitHub Pages preview.** Mentioned as "open question" in plan;
  README roadmap defers it.
- **Smoke-test evidence for Router and SQLAgent.** Plan M3.75 ship criterion ("router
  classifies 10 sample questions correctly; SQL agent answers 5 factual questions") has
  no committed evidence; tests only cover mocked LLM responses.
- **Live judge calibration evidence in this checkout.** `docs/judge_calibration.md`
  cites a 2026-04-25 run, but the cache and any logs are gitignored — a fresh clone
  would need a paid API call to reproduce. Acceptable, but worth flagging.

If the goal is a v0.1 release that matches the README's pitch, items 1–3 above are the
release blockers. Everything else is polish or scope honesty.
