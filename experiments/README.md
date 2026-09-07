# Experiments — InferenceLedger reference model

Deterministic migration-economics demonstrations (L2). Educational model
with synthetic attempt streams; not production traffic, not billing-grade.

## Run

From `inferenceledger-reference/`:

```
python experiments/run_experiment.py
python -m pytest reference/tests/ -q   # from inferenceledger-reference/reference/
```

## Layout

- `run_experiment.py` — deterministic runner (stdlib only).
- `results.json` — machine-readable output (regenerated on each run).
- `interpretation.md` — what each verdict means and what is NOT claimed.

## Result schema

Each entry: `scenario, configuration, expected_behavior,
observed_behavior, pass, evidence_level, limitations`.

## Scenarios (7)

1. `multi_provider_cheaper_ok` — cheaper provider, same reliability → SHIP.
2. `retry_economics_unknown` — flaky candidate burns retries + unknown billing → NO-GO.
3. `latency_regression_blocks_cost_win` — cost win with latency breach → NO-GO.
4. `latency_cost_tradeoff_within_margin` — cost win within latency budget → SHIP.
5. `reliability_gating_unknown_forces_review` — reliable but unknown billing → REVIEW (uncertainty propagation).
6. `unknown_cost_preserved` — unknown attempts counted separately, never zeroed.
7. `gate_boundary_matrix` — four gate boundaries pinned in one sweep.
