# Prompt Evolution

The notes below document why each prompt exists and what changed during
implementation and calibration. Future prompt versions should add a new
`vN.txt` file and append a short migration note here instead of silently
rewriting history.

## SQL Agent

- Prompt: `prompts/sql_agent/v1.txt`
- Purpose: Convert NBA factual questions into safe read-only SQL over the demo
  schema.
- Evolution note: `v1` emphasizes SQLite compatibility, schema grounding,
  read-only output, and refusal when a question cannot be answered by the
  database. Safety validation lives in code, but the prompt keeps the model on
  the narrow task.
- Prompt: `prompts/sql_agent/v2.txt`
- Evolution note: Live SQL evaluation exposed a mismatch between the v1 prompt
  and strict SQL-row scoring: generated rows often used aliases such as
  `player_name`, omitted identifying entity columns, matched accented player
  names too exactly, or returned NULL-sorted results for unavailable
  Basketball Reference-only stats. `v2` is now the default SQLAgent prompt. It
  instructs the model to include `players.full_name AS full_name` and
  `teams.team_name AS team_name`, preserve `season_id` aliases, use robust
  prefix/nickname matching for accented names and teams, aggregate
  regular-season three-point attempts from `player_game_stats`, use table
  aliases consistently in grouped regular/playoff comparisons, and return a
  clear placeholder row when a requested stat is unavailable in the nba_api
  ingestion schema.

## Router

- Prompt: `prompts/router/v1.txt`
- Purpose: Classify questions as `factual`, `analytical`, `hybrid`, or
  `unanswerable`.
- Evolution note: `v1` was written to separate SQL-only questions from
  article-analysis questions and to make adversarial/future/unavailable
  questions route to `unanswerable`.

## Synthesizer

- Prompt: `prompts/synthesizer/v1.txt`
- Purpose: Combine SQL rows and retrieved article chunks into a cited final
  answer.
- Evolution note: `v1` requires inline citations (`[sql]` and
  `[article:chunk_id]`) and source-grounded answers. The deterministic local
  fallback was later polished to produce concise evidence summaries without
  needing live LLM calls.

## Faithfulness Judge

- Prompt: `prompts/judges/faithfulness/v1.txt`
- Purpose: Decide whether answer claims are supported by supplied sources.
- Evolution note: The prompt focuses on source support, not real-world truth.
  Calibration passed the Milestone 4 threshold, so no new prompt version was
  introduced.

## Relevance Judge

- Prompt: `prompts/judges/relevance/v1.txt`
- Purpose: Decide whether an answer directly addresses the user's question.
- Evolution note: The prompt distinguishes relevance from correctness: an
  answer can be on-topic and still factually wrong. Calibration passed the
  Milestone 5 threshold with `v1`.

## Correctness Judge

- Prompt: `prompts/judges/correctness/v1.txt`
- Purpose: Score candidate answers against expected answers on a 0-4 rubric.
- Evolution note: The implementation added position-swap mitigation around the
  prompt: the judge is run in forward and swapped presentation order, then
  exposes `forward_score`, `swapped_score`, `disagreement`, and
  `disagreement_flag`. The prompt remained `v1` because the main fix was
  orchestration logic, not rubric wording.

## Routing Judge

- Prompt: `prompts/judges/routing/v1.txt`
- Purpose: Placeholder for a possible LLM-assisted routing judge.
- Evolution note: Milestone 5 intentionally settled on deterministic routing
  evaluation. The prompt file remains as documentation for the alternative, but
  `RoutingJudge` does not load it.

## When To Create `v2`

Create a new prompt version when calibration shows a systematic failure that is
best fixed with prompt wording rather than code, fixtures, or metric semantics.
Each new version should record:

- the failed calibration cases,
- the old agreement rate,
- the prompt change,
- the new agreement rate,
- any expected tradeoffs.
