# Judge Calibration

## Why LLM Judges Need Calibration

LLM-as-judge systems are powerful but imperfect. Without calibration they can exhibit:

- **Positivity bias** — tendency to rate answers as correct or faithful when they are not
- **Prompt sensitivity** — small wording changes can shift scores significantly
- **Position bias** — preference for whichever answer appears first in the prompt
- **Reasoning drift** — the judge may explain its score plausibly but reach the wrong conclusion

Calibration means collecting examples where a human has already labeled the correct verdict, then measuring how often the judge agrees.

---

## How to Run

```bash
# Requires ANTHROPIC_API_KEY for LLM-backed judges.
uv run python scripts/calibrate_judge.py                      # all judges
uv run python scripts/calibrate_judge.py faithfulness         # one judge
uv run python scripts/calibrate_judge.py all --threshold 0.8
```

Results are cached in `.rageval_cache/`. A repeated run with the same fixtures
and prompts is near-instant and free. The script prints each judge's agreement
rate and exits non-zero if any judge falls below the threshold.

Deterministic judges (currently only `routing`) require no API key.

**Current status before Milestone 6:** unit tests in `tests/test_judges.py` and
`tests/test_calibrate.py` fully exercise parsing, error handling, tool-use
plumbing, position-swap handling, and agreement math with mocked LLM responses.
The scripted real-LLM calibration run is gated on `ANTHROPIC_API_KEY` and has
not been recorded in this doc yet. Until that run is recorded, faithfulness,
relevance, and correctness should be treated as implemented but not yet
empirically calibrated.

## Method

- Each judge has a fixture at `tests/fixtures/<judge>_calibration.yaml`.
- Each fixture has **10 hand-labeled cases** covering positive and negative labels.
- The calibration script instantiates the judge with a shared `LLMClient`, runs
  it against every case, and compares the judge's verdict to the human label.
- Agreement rate = fraction of cases where the judge's verdict matches the label.
- Default threshold: **≥ 80% agreement** (8/10).
- Calibration is designed to be cheap and repeatable. Disk caching means a second
  run is free unless the prompt or fixtures change.

## Structured Output

All LLM-backed judges use Anthropic **tool-use** for structured output. The
judge prompts describe which tool to call (`record_faithfulness`,
`record_relevance`, `record_correctness`) and the expected schema. The
`LLMClient` surfaces tool-use blocks as `tool_calls` in the returned dict, and
the judge implementations parse them with strict type checks. Malformed outputs
produce a `MetricResult` with `error` set and `value=0.0` so calibration can
distinguish misclassification from protocol errors.

---

## Faithfulness

Faithfulness measures whether every claim in an answer is directly stated or strongly implied by the provided sources. An answer is faithful if a reader could verify each claim by consulting the sources alone.

A claim is **unsupported** if it cannot be derived from the sources, even if factually correct in the real world.

**Calibration set:** `tests/fixtures/faithfulness_calibration.yaml` — 10 NBA examples (5 faithful, 5 unfaithful).

| ID | Question (abbreviated) | Expected |
|----|------------------------|----------|
| faithful-001 | Who led in PPG? | true |
| faithful-002 | LeBron championships? | true |
| faithful-003 | Jokic RPG 2022-23? | true |
| faithful-004 | Best record 2023-24? | true |
| faithful-005 | Curry 3PT%? | true |
| unfaithful-001 | Who led in PPG? | false |
| unfaithful-002 | LeBron championships? | false |
| unfaithful-003 | Jokic APG 2022-23? | false |
| unfaithful-004 | 2023 Finals MVP? | false |
| unfaithful-005 | KD scoring average? | false |

---

## Relevance

Relevance measures whether the answer directly addresses the question. It does not require factual correctness — an answer can be relevant but wrong. Penalizes answers that dodge the question, answer a different question, or contain substantial unrelated material.

**Calibration set:** `tests/fixtures/relevance_calibration.yaml` — 10 NBA examples (5 relevant, 5 irrelevant).

| ID | Question (abbreviated) | Expected |
|----|------------------------|----------|
| relevant-001 | Who led in PPG? | true |
| relevant-002 | LeBron championships? | true |
| relevant-003 | Jokic RPG 2022-23? | true |
| relevant-004 | Best record 2023-24? | true |
| relevant-005 | Curry career 3PT%? | true |
| irrelevant-001 | Who led in PPG? | false |
| irrelevant-002 | LeBron championships? | false |
| irrelevant-003 | Jokic RPG 2022-23? | false |
| irrelevant-004 | Best record 2023-24? | false |
| irrelevant-005 | Curry career 3PT%? | false |

---

## Correctness

Correctness measures how well the answer matches the expected answer on a 0–4 scale:

| Score | Meaning |
|-------|---------|
| 4 | Fully correct — all key information present, no errors |
| 3 | Mostly correct — minor omissions or wording differences |
| 2 | Partially correct — important omissions or some wrong details |
| 1 | Mostly incorrect — only a small correct element |
| 0 | Incorrect — wrong, unsupported, or refuses when it should answer |

`MetricResult.value` = average of forward and swapped scores / 4.0.

**Calibration set:** `tests/fixtures/correctness_calibration.yaml` — 10 NBA examples covering all score levels.

### Position-Swap Bias Mitigation

LLM judges can prefer whichever answer appears first in the prompt. To mitigate
this, `CorrectnessJudge` runs the judge twice:

1. **Forward pass** — Candidate = `response.answer`, Reference = `case.expected_answer`
2. **Swapped pass** — Candidate = `case.expected_answer`, Reference = `response.answer`, with a note that the judge is checking semantic equivalence regardless of order

The final score is the average of both raw scores divided by 4. `MetricResult.details` includes:

- `forward_score` — raw score from the forward pass
- `swapped_score` — raw score from the swapped pass
- `disagreement` — absolute difference between the two scores
- `disagreement_flag` — `True` if `disagreement >= 2`; signals the result is unstable
- `reasoning_forward` and `reasoning_swapped` — judge reasoning for each pass

A high `disagreement` value (≥ 2, flagged explicitly) is a signal to investigate the case manually.

---

## Routing Accuracy

Routing accuracy is evaluated **deterministically** — no LLM call is made. It
compares `response.routing_decision` (a `QuestionType`) against `case.question_type`.

- Value = `1.0` if they match, `0.0` otherwise.
- If `routing_decision` is `None`, value = `0.0`.

The deterministic design is intentional: routing is a supervised classification
task with a fixed set of four outcomes (factual / analytical / hybrid /
unanswerable). Using an LLM to judge an LLM's routing choice would introduce
noise without adding signal. The file `prompts/judges/routing/v1.txt` is kept
as a placeholder for a future LLM-assisted variant if needed; it is intentionally
unused, and a test asserts the file documents this.

**Calibration set:** `tests/fixtures/routing_calibration.yaml` — 10 NBA examples (5 correct routes, 5 incorrect routes). The deterministic judge matches by construction (100%).

---

## Results Table

*(Populate this table after running `uv run python scripts/calibrate_judge.py`.)*

| Judge | Agreement | Threshold | Status |
|-------|-----------|-----------|--------|
| faithfulness | — | ≥ 80% | (run calibration) |
| relevance | — | ≥ 80% | (run calibration) |
| correctness | — | ≥ 80% | (run calibration) |
| routing | 100% | ≥ 80% | PASS (deterministic) |

The numbers above will be filled in on the first scripted live-LLM calibration
run. Prompt changes should be followed by a re-run and an update to this table.

## Milestone 6 Readiness Note

The judge implementations are ready to be wired into the evaluator, but their
live agreement rates are still pending. Milestone 6 can proceed using mocked
judge tests and deterministic fixtures, but any public claim that the LLM-backed
judges are calibrated at or above 80% should wait until the table above is
filled with a real run.

---

## Caveats

**Judge bias.** The judge LLM may systematically favour certain verdicts — e.g., reluctance to call an answer wrong if it sounds confident.

**Prompt sensitivity.** Score distributions can shift with small wording changes in the judge prompt. Changes to any `prompts/judges/*/v1.txt` file should be followed by a re-run of the relevant calibration set.

**Position bias.** Even with position-swap mitigation, the judge may not be fully symmetric. High `disagreement_flag` values in correctness results warrant manual review.

**Calibration set limitations.** Ten examples per judge is enough to catch gross failures but not enough to measure subtle biases. Expand the sets before drawing conclusions about judge reliability.
