# Judge Calibration

## Why LLM Judges Need Calibration

LLM-as-judge systems are powerful but imperfect. Without calibration they can exhibit:

- **Positivity bias** — tendency to rate answers as correct or faithful when they are not
- **Prompt sensitivity** — small wording changes can shift scores significantly
- **Position bias** — preference for whichever answer appears first in the prompt
- **Reasoning drift** — the judge may explain its score plausibly but reach the wrong conclusion

Calibration means collecting examples where a human has already labeled the correct verdict, then measuring how often the judge agrees.

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

LLM judges can prefer whichever answer appears first in the prompt. To mitigate this, `CorrectnessJudge` runs the judge twice:

1. **Forward pass** — Candidate = `response.answer`, Reference = `case.expected_answer`
2. **Swapped pass** — Candidate = `case.expected_answer`, Reference = `response.answer`, with a note that the judge is checking semantic equivalence regardless of order

The final score is the average of both raw scores divided by 4. `MetricResult.details` includes:

- `forward_score` — raw score from the forward pass
- `swapped_score` — raw score from the swapped pass
- `disagreement` — absolute difference between the two scores (high values signal instability)
- `reasoning_forward` and `reasoning_swapped` — judge reasoning for each pass

A high `disagreement` value (≥ 2) is a signal to investigate the case manually.

---

## Routing Accuracy

Routing accuracy is evaluated **deterministically** — no LLM call is made. It compares `response.routing_decision` (a `QuestionType`) against `case.question_type`.

- Value = `1.0` if they match, `0.0` otherwise.
- If `routing_decision` is `None`, value = `0.0`.

**Calibration set:** `tests/fixtures/routing_calibration.yaml` — 10 NBA examples (5 correct routes, 5 incorrect routes).

---

## Current Status

Tests in `tests/test_judges.py` validate the plumbing only — they mock the LLM and verify that judges correctly parse responses, handle errors, and propagate details. They do not measure real judge agreement with the calibration sets.

Real calibration (running judges against the fixture sets and computing agreement with human labels) will be done in a later milestone.

---

## Caveats

**Judge bias.** The judge LLM may systematically favour certain verdicts — e.g., reluctance to call an answer wrong if it sounds confident.

**Prompt sensitivity.** Score distributions can shift with small wording changes in the judge prompt. Changes to any `prompts/judges/*/v1.txt` file should be followed by a re-run of the relevant calibration set.

**Position bias.** Even with position-swap mitigation, the judge may not be fully symmetric. High `disagreement` values in correctness results warrant manual review.

**Calibration set limitations.** Ten examples per judge is enough to catch gross failures but not enough to measure subtle biases. Expand the sets before drawing conclusions about judge reliability.
