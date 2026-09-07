# Changelog — InferenceLedger showcase

## v4 (2026-09-07) — standalone public repository

- 7 reference-model scenarios: multi-provider, retry economics, latency
  tradeoff on both sides of the budget, reliability gating with uncertainty
  propagation, gate boundary matrix.
- Docs: migration decision framework, rejected alternatives, operational
  lessons, tradeoffs.
- Standalone repo layout: `make verify`, CI workflow, LICENSE, examples.

Evidence: reference-model results are L2 (synthetic attempts, not production
traffic). The historical 5-request smoke is a frozen L2 artifact, not a cost
study. No billing-grade claims.

## v3 — attempt-aware model + gate

- Attempt-level cost model, non-compensatory gate, experiment runner, ledger.
