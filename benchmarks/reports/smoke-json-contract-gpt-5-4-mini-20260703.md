# Smoke Benchmark Evidence: `gpt-5.4-mini`

## Verdict

Partially defensible.

This run is valid smoke evidence for provider connectivity, usage capture, cost
accounting, latency recording, route tracing, and deterministic JSON-contract quality
checks. It is not evidence of cost savings because it has no optimized candidate run.

## Configuration

| Field | Value |
| --- | --- |
| Date | 2026-07-03 |
| Provider | OpenAI-compatible FreeModel endpoint |
| Model | `gpt-5.4-mini` |
| Strategy | `single_model` |
| Workload | `benchmarks/workloads/smoke.jsonl` |
| Workload SHA256 | `c87a143808dc950f7cc229b0806c4ecfd9ee70b5ce0c568a0f5c73dc60c3b520` |
| Requests | 5 |

## Result

| Metric | Value |
| --- | ---: |
| Successes | 5 |
| Failures | 0 |
| Quality pass rate | 100.00% |
| Quality score average | 1.0000 |
| Provider attempts | 5 |
| Provider retries | 0 |
| Prompt tokens | 4562 |
| Completion tokens | 63 |
| Total tokens | 4625 |
| Estimated cost | $0.00111300 |
| Latency p50 | 3659 ms |
| Latency p95 | 17037 ms |

## Reproduction Command

```bash
set -a; source .env; set +a

.venv/bin/python scripts/run_benchmark.py run \
  --workload benchmarks/workloads/smoke.jsonl \
  --strategy single_model \
  --model gpt-5.4-mini \
  --max-tokens 64 \
  --temperature 0 \
  --max-estimated-cost-usd 0.01 \
  --run-id smoke-json-contract-gpt-5-4-mini-20260703 \
  --report-path reports/benchmarks/latest-smoke-json-contract.json \
  --ledger-path reports/benchmarks/latest-smoke-json-contract.jsonl \
  --route-ledger-path reports/benchmarks/latest-smoke-json-contract-routes.jsonl
```

## Blocking Issues Before Savings Claims

- No candidate strategy was compared against this baseline.
- The smoke workload has only five deterministic tasks.
- The latency p95 is based on too few samples for general conclusions.
- Quality checks validate constrained JSON contracts, not broad semantic quality.

## Acceptable Claim

This artifact supports the claim that the project can run a real provider-backed
single-model smoke benchmark and record usage, latency, cost, retry, and quality
evidence.

It does not support a routing optimization or cost savings claim.
