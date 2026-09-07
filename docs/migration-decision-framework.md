# Migration decision framework — InferenceLedger

How to decide a model/provider migration with the reference cost model.
This is the operational companion to `docs/economics-model.md` (unit of
evidence, known-vs-unknown money, non-compensatory margins) and
`docs/measurement-assumptions.md` (what the numbers assume).

## Inputs required before any verdict

1. **Attempt streams** for baseline and candidate over the same workload
   mix (attempt = one provider call, retries first-class, unknown billing
   preserved as `None` — never zeroed).
2. **Reliability floor** (`min_success_rate`) set by the owning team.
3. **Latency budget** (`max_latency_regression_ms`) set by the owning team.
4. **Known-cost comparison** on aggregates with unknown attempts reported
   separately.

Missing any input → verdict is REVIEW by default. The gate never infers
missing inputs in the candidate's favor.

## Verdict table

| Condition | Verdict | Rationale |
|-----------|---------|-----------|
| Candidate success < floor, or < baseline | NO-GO | Reliability is non-compensatory; no cost win offsets it |
| p95 latency regression > budget | NO-GO | Latency breach blocks even with a cost win (`latency_tradeoff_example`) |
| Any unknown-billing attempts | REVIEW (at best) | Uncertainty propagates: price the unknown first (`reliability_gating_example`) |
| Cheaper + reliable + within latency + zero unknown | SHIP | The only SHIP path (`multi_provider_example`, `gate_boundary_matrix.clean_win_ship`) |
| Cheaper + slightly slower *within* budget | SHIP | Bounded tradeoff is allowed (`latency_cost_tradeoff_within_margin`) |

## Worked boundary (see `gate_boundary_matrix` scenario)

- reliability miss + 10x cost win → NO-GO
- latency breach + 10x cost win → NO-GO
- full reliability + 2 unknown attempts → REVIEW
- clean win within latency budget → SHIP

## What this framework does NOT do

- Price unknown billing (it quarantines it).
- Set your reliability floor or latency budget (team-owned inputs).
- Generalize from the 5-request historical smoke (frozen L2 artifact; zero
  retries; not a migration study).
- Replace production shadow evaluation. This framework triages *whether a
  migration deserves* a shadow run, in which order, and with what stop
  conditions — it is not the shadow run.
