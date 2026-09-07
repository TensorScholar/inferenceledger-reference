# InferenceLedger

**Attempt-aware cost and reliability evidence for inference migrations.**

Invariant: migration evidence is paired, attempt-aware, and non-compensatory. Unknown cost is not rewritten as zero. A cost win cannot offset a required reliability or latency miss.

## Proof moment

Historical five-request smoke: 69.96% lower calculated cost, 5/5 successes, 100% validator pass, **zero retries**. Frozen **L2**. Not current product validation. Not retry economics.

Current-head attempt model: **L1**. Public reference examples (multi-provider, retries, unknown, latency): **L2** for the *model*, not for a production bill.

## Run

```
cd reference && python -m pytest tests/ -q && cd ..
python experiments/run_experiment.py
```

[Economics model](docs/economics-model.md) · [Measurement assumptions](docs/measurement-assumptions.md) · [Failure analysis](docs/failure-analysis.md)

## This standalone repository

This folder is a complete public repo: `git init && make verify` reproduces
everything (reference tests, 7 gate scenarios, evidence integrity). Try
`python3 examples/quickstart.py` first.

- Demonstrates: attempt-aware cost aggregation + non-compensatory migration
  gating on synthetic attempt streams.
- Intentionally excluded: production traffic, provider pricing, billing-grade
  accounting, the historical 5-request smoke (frozen L2 artifact, not a cost study).
- Evidence maturity: L2 for the reference model; no L3.
- Limitations: latency inputs are example p95s, not measurements; unknown
  billing is quarantined to REVIEW, never priced.
