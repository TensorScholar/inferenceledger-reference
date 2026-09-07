# Interpretation — InferenceLedger experiment results

## What `all_pass: true` means

The gate behaves non-compensatorily across all seven demonstrations:
cost wins never offset reliability or latency misses, unknown billing
always forces REVIEW-or-worse, and bounded tradeoffs inside pre-declared
budgets still SHIP. Uncertainty is preserved end-to-end (unknown attempts
are counted, never priced as zero).

## What it does NOT mean

- Not a migration recommendation for any real provider/model. Attempt
  streams are synthetic; latency inputs are example p95s, not measurements.
- Not billing-grade accounting. The model shows *how to avoid lying about
  money* (preserve unknowns, expose retries), not what anything costs.
- Not a re-analysis of the historical 5-request smoke. That frozen L2
  artifact had zero retries and is retained unchanged; these scenarios are
  separate illustrations, not generalizations from n=5.

## How to challenge this

1. Change `max_latency_regression_ms` in the tradeoff scenarios and confirm
   the SHIP/NO-GO boundary moves exactly with the budget — the gate must be
   budget-driven, not vibes-driven.
2. Add an all-unknown-cost candidate and confirm it can never SHIP no matter
   how cheap its known portion looks.
3. Ask for a production shadow evaluation: that is the correct next step and
   is listed as future validation, not present evidence.
