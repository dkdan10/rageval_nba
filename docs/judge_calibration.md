# Judge Calibration

## What is Faithfulness?

Faithfulness measures whether every claim in an answer is directly stated or strongly implied by the provided sources. An answer is faithful if a reader could verify each of its claims by consulting the sources alone — without relying on outside knowledge.

A claim is **unsupported** if it cannot be derived from the sources, even if it is factually correct in the real world. General knowledge and reasonable inferences count as faithful only when they are clearly grounded in the provided sources.

## Why LLM Judges Need Calibration

LLM-as-judge systems are powerful but imperfect. Without calibration they can exhibit:

- **Positivity bias** — tendency to rate answers as faithful when they are not
- **Prompt sensitivity** — small wording changes in the judge prompt can shift scores significantly
- **Reasoning drift** — the judge may explain its score plausibly but reach the wrong conclusion

Calibration means collecting examples where a human has already labeled the correct verdict, then measuring how often the judge agrees. Low agreement signals the judge needs a better prompt, a stronger model, or both.

## Calibration Set

The fixture at `tests/fixtures/faithfulness_calibration.yaml` contains 10 NBA-themed examples (5 faithful, 5 unfaithful).

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

Unfaithful examples include: wrong player named, inflated statistics, claims derived from sources that do not contain the relevant data, and specific facts not present in any source.

## Current Status

Tests in `tests/test_judges.py` validate the plumbing only — they mock the LLM and verify that the judge correctly parses responses, handles errors, and formats source context. They do not measure real judge agreement with the calibration set.

Real calibration (running the judge against the 10-example set and measuring agreement with `human_label`) will be done in a later milestone once the judge is deployed against actual LLM calls.

## Caveats

**Judge bias.** The judge LLM may have systematic tendencies — for example, it may be reluctant to call an answer unfaithful if the answer sounds confident or matches its pretraining knowledge.

**Prompt sensitivity.** The faithfulness verdict can shift depending on how sources are formatted, how the question is framed, or the exact wording of the judge instructions. Changes to `prompts/judges/faithfulness/v1.txt` should be followed by a re-run of the calibration set.

**Unsupported claims vs. reasonable inference.** There is a judgment call between a claim that is "strongly implied" by a source and one that is "not directly stated." The prompt instructs the judge to allow inferences only when grounded in the sources, but borderline cases will always exist. Human review of the calibration set is the only reliable check.
