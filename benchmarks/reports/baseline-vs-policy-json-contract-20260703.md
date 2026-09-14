# Baseline vs Policy Benchmark: JSON Contract Smoke Suite

## Verdict

Partially defensible.

The policy run is comparable to the fixed-model baseline for this five-request
JSON-contract smoke workload under the benchmark's original comparability rules. It reduced
**calculated execution cost from provider-reported usage and the repository pricing table** while
preserving deterministic quality pass rate. The result is not provider billing proof, broad
evidence of semantic quality parity, retry/fallback economics, or production readiness.

## Configuration

| Field | Baseline | Candidate |
| --- | --- | --- |
| Date | 2026-07-03 | 2026-07-03 |
| Provider | OpenAI-compatible FreeModel endpoint | OpenAI-compatible FreeModel endpoint |
| Strategy | `single_model` | `policy` |
| Model profile | `gpt-5.4` | `gpt-5.4-mini`, `gpt-5.4`, `gpt-5.5` |
| Workload | `benchmarks/workloads/smoke.jsonl` | `benchmarks/workloads/smoke.jsonl` |
| Quality gate | Deterministic JSON field equality | Deterministic JSON field equality |
| Requests | 5 | 5 |

Workload SHA256:

```text
c87a143808dc950f7cc229b0806c4ecfd9ee70b5ce0c568a0f5c73dc60c3b520
```

## Result

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Calculated cost | $0.00371000 | $0.00111450 | -$0.00259550 |
| Cost delta | n/a | n/a | -69.96% |
| Latency p50 | 2976 ms | 2579 ms | -397 ms |
| Latency p95 | 11588 ms | 8681 ms | -2907 ms |
| Successes | 5 | 5 | 0 |
| Failures | 0 | 0 | 0 |
| Quality pass rate | 100.00% | 100.00% | 0.00 pp |
| Provider retries | 0 | 0 | 0 |

The candidate selected `gpt-5.4-mini` for all five requests.

The cost values above were calculated from provider-reported token usage using the repository
pricing table active for these runs. They are not a provider invoice or authoritative
provider-reported charge.

## Reproduction Commands

```bash
set -a; source .env; set +a

.venv/bin/python scripts/run_benchmark.py run \
  --workload benchmarks/workloads/smoke.jsonl \
  --strategy single_model \
  --model gpt-5.4 \
  --max-tokens 64 \
  --temperature 0 \
  --max-estimated-cost-usd 0.05 \
  --run-id baseline-json-contract-gpt-5-4-20260703

.venv/bin/python scripts/run_benchmark.py run \
  --workload benchmarks/workloads/smoke.jsonl \
  --strategy policy \
  --economy-model gpt-5.4-mini \
  --standard-model gpt-5.4 \
  --premium-model gpt-5.5 \
  --max-tokens 64 \
  --temperature 0 \
  --max-estimated-cost-usd 0.05 \
  --policy-min-quality-score 0.45 \
  --policy-cost-weight 0.70 \
  --policy-latency-weight 0.10 \
  --policy-quality-weight 0.20 \
  --run-id policy-json-contract-freemodel-20260703

.venv/bin/python scripts/run_benchmark.py compare \
  --baseline-run-id baseline-json-contract-gpt-5-4-20260703 \
  --candidate-run-id policy-json-contract-freemodel-20260703 \
  --comparison-path reports/benchmarks/latest-json-contract-baseline-vs-policy-comparison.json
```

## Blocking Issues Before Broader Claims

- The workload has only five requests.
- Quality is deterministic JSON-contract quality, not broad semantic quality.
- The policy quality floor is heuristic; actual quality is enforced after execution by the
  benchmark comparison gate.
- Latency p95 is reported for transparency but is not statistically strong at this sample size.
- The result uses one provider endpoint and one run per strategy.
- Both runs recorded zero provider retries, so this artifact contains no evidence about retry or
  fallback economics.
- The current request-level ledger cannot reconstruct individual provider attempts or authoritative
  billing for failed attempts.

## Acceptable Claim

For the committed five-request JSON-contract smoke workload, the policy strategy produced a run
that met the original benchmark's narrow comparability rule, had no deterministic quality
pass-rate regression, and had lower **calculated cost from observed provider usage under the
repository pricing table** than the fixed `gpt-5.4` baseline.

Do not generalize this result to provider billing, retry/fallback economics, broad workloads,
production traffic, semantic quality parity, cross-provider migration, or the current commercial
product thesis.
