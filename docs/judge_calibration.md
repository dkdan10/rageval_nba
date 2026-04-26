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
uv run python scripts/calibrate_judge.py all --threshold 0.8 --no-cache
```

Results are cached in `.rageval_cache/`. A repeated run with the same fixtures
and prompts is near-instant and free. The script prints each judge's agreement
rate and exits non-zero if any judge falls below the threshold. Use
`--no-cache` when recording a fresh live calibration run.

Deterministic judges (currently only `routing`) require no API key.

**Current status before Milestone 6:** live calibration was recorded on
2026-04-26 with `claude-haiku-4-5-20251001`. Faithfulness, relevance,
correctness, and routing all meet the >= 80% agreement threshold. Unit tests in
`tests/test_judges.py` and `tests/test_calibrate.py` also exercise parsing,
error handling, tool-use plumbing, position-swap handling, and agreement math
with mocked LLM responses.

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
2. **Swapped pass** — Reference is shown before Candidate, but the judge still scores `response.answer` as the candidate against `case.expected_answer`

The final score is the average of both raw scores divided by 4. `MetricResult.details` includes:

- `forward_score` — raw score from the forward pass
- `swapped_score` — raw score from the swapped pass
- `disagreement` — absolute difference between the two scores
- `disagreement_flag` — `True` if `disagreement >= 2`; signals the result is unstable
- `reasoning_forward` and `reasoning_swapped` — judge reasoning for each pass

A high `disagreement` value (≥ 2, flagged explicitly) is a signal to investigate the case manually.

Live uncached calibration on 2026-04-25 confirmed the swap fields are surfaced in
`MetricResult.details`. The recorded correctness run exposed `forward_score`,
`swapped_score`, `disagreement`, and `disagreement_flag` for every fixture case;
no case had `disagreement_flag=True` after the position-order fix.

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
unused, and a test asserts the file documents this. This deterministic design is
the intended Milestone 5 implementation and is considered plan-compliant.

**Calibration set:** `tests/fixtures/routing_calibration.yaml` — 10 NBA examples (5 correct routes, 5 incorrect routes). The deterministic judge matches by construction (100%).

---

## Results Table

Run date: 2026-04-26

Model for LLM-backed judges: `claude-haiku-4-5-20251001`

Prompt changes in this calibration pass: none. The correctness implementation
was adjusted so the swapped pass reverses presentation order while still scoring
the original candidate answer against the reference answer.

Reproducibility note: `uv run rageval calibrate faithfulness relevance
correctness routing --no-cache` produced the table below. The `--no-cache` flag bypasses
`.rageval_cache/` so the recorded result reflects fresh Anthropic responses.

| Judge | Agreement | Threshold | Status |
|-------|-----------|-----------|--------|
| faithfulness | 100% (10/10) | >= 80% | PASS |
| relevance | 100% (10/10) | >= 80% | PASS |
| correctness | 80% (8/10) | >= 80% | PASS |
| routing | 100% (10/10) | >= 80% | PASS (deterministic) |

Command summary:

```text
uv run rageval calibrate faithfulness relevance correctness routing --no-cache
Total LLM cost: $0.0606 (42721 in / 6605 out tokens)
[PASS] correctness: 80% (8/10)
[PASS] faithfulness: 100% (10/10)
[PASS] relevance: 100% (10/10)
[PASS] routing: 100% (10/10)
```

Previous script entry point remains equivalent:

```text
uv run python scripts/calibrate_judge.py all --threshold 0.8 --no-cache
```

Correctness swap evidence from an uncached run:

```text
correctness-001: pred=4 human=4 forward=4 swapped=4 disagreement=0 flag=False
correctness-002: pred=3 human=3 forward=3 swapped=3 disagreement=0 flag=False
correctness-003: pred=2 human=2 forward=2 swapped=1 disagreement=1 flag=False
correctness-004: pred=0 human=1 forward=0 swapped=0 disagreement=0 flag=False
correctness-005: pred=0 human=0 forward=0 swapped=0 disagreement=0 flag=False
correctness-006: pred=4 human=4 forward=4 swapped=4 disagreement=0 flag=False
correctness-007: pred=4 human=4 forward=4 swapped=4 disagreement=0 flag=False
correctness-008: pred=1 human=1 forward=1 swapped=1 disagreement=0 flag=False
correctness-009: pred=4 human=3 forward=4 swapped=4 disagreement=0 flag=False
correctness-010: pred=0 human=0 forward=0 swapped=0 disagreement=0 flag=False
agreement 0.8
```

## Milestone 6 Readiness Note

The judge implementations are ready to be wired into the evaluator. Milestone 6
can proceed with the recorded calibration results above and the mocked tests as
regression coverage.

---

## Caveats

**Judge bias.** The judge LLM may systematically favour certain verdicts — e.g., reluctance to call an answer wrong if it sounds confident.

**Prompt sensitivity.** Score distributions can shift with small wording changes in the judge prompt. Changes to any `prompts/judges/*/v1.txt` file should be followed by a re-run of the relevant calibration set.

**Position bias.** Even with position-swap mitigation, the judge may not be fully symmetric. High `disagreement_flag` values in correctness results warrant manual review.

**Calibration set limitations.** Ten examples per judge is enough to catch gross failures but not enough to measure subtle biases. Expand the sets before drawing conclusions about judge reliability.
