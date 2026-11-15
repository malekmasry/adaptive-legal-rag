# Stress20: completed 15 November 2026

Code: `9bbcfa1`, based on Moonwalk `f19a327`. All 20 cases completed for all
three systems; 60 evaluations in two answer orders, no missing judgments.
15 offline implementation tests passed. Frozen input and code hashes verified.

| System | PSB-rule proxy /100 | Literal-gold sensitivity /100 |
|---|---:|---:|
| conditional35, historical answers | 55.99 | 9.28 |
| v8, fresh unchanged ZIP | 55.84 | 20.23 |
| hybrid35v8, fresh unchanged ZIP | 44.83 | 20.23 |

The historical platform gap (35 versus 12.5) was **not reproduced**. The extra
fact zeroing hypothesis reverses the historical order and cannot explain it on
this cohort. Literal-gold sensitivity deliberately violates the FAQ for true
extra facts; it is not another official score. Scores are relative to a generated
local baseline, not percentages of completely correct answers.

Observed failures include withheld arithmetic despite retrieving the rate,
missed supporting clauses, needless entity clarification, and unrelated context
added to otherwise correct answers. The simple judge also confuses missing
retrieved evidence with false statements and sometimes disagrees with its own
numeric preference. Raw decisions were retained, not edited to fit known scores.

Stress20 took 884.343 seconds, excluding the interrupted parent pilot. Fresh v8
and hybrid generation took 557.156 and 392.968 seconds externally; both returned
20 rows without terminal technical_error. They ran concurrently with shared API
slots and cold local BGE, so these are not platform latency estimates.

Recorded usage including the parent pilot: 1,037,292 DeepSeek tokens and
1,930,503 Qwen input tokens. Three interrupted requests and a timeout have
incomplete usage; these totals are not exact billed cost. Only DeepSeek used
the user chat API; only Qwen3-Reranker-8B used the separate reranker API.

This is an adversarial selection of previously exposed development questions,
not a holdout or a calibrated leaderboard predictor. Historical conditional35
answers were reused; the comparison is not a fully controlled regeneration.
Private questions, answers, logs and credentials remain outside the public repo.
Aggregate data: [results-stress20.json](results-stress20.json).
