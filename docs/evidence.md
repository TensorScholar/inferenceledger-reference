# Evidence and decision model

InferenceLedger is useful only if a reviewer can tell **what was observed, what was inferred, what
is comparable, and what remains unknown**. This document is the compact index for those semantics
and the evidence already committed to the repository.

## Evidence rules

### 1. Preserve attempts, not just final requests

A successful final response does not erase failed or billable attempts. Provider attempts have
independent outcome, latency, usage, error, and cost evidence. Request-level cost is complete only
when the required attempt-chain evidence is complete.

### 2. Monetary evidence carries origin

The code distinguishes:

- calculated-from-usage cost with provider/model/date pricing provenance;
- execution-stack-reported cost with source provenance;
- unknown cost with no numeric amount.

Prospective pilot structures add exact decimal-string amounts and rates before any float-based
statistical projection. Unknown cost is never silently represented as zero.

### 3. Completeness is not validity

`decision-evidence-completeness-v1` asks whether every required evidence surface is present or
explicitly unavailable. `EXPLICITLY_UNAVAILABLE` can satisfy the *representation* contract while
still being unusable to support the substantive claim. This is why a bundle can be complete and the
final decision can still be `ABSTAIN`/`INCONCLUSIVE`.

### 4. Comparability is semantic

Provenance proves where evidence came from; it does not prove that two fields mean the same thing.
The OpenRouter experiment is the concrete counterexample: exact monetary reconciliations matched,
while the frozen token-field comparison did not. The historical `VALIDATED_MISMATCH` classification
is intentionally preserved.

### 5. The gate is an intersection

Every required check must independently support approval. Quality, reliability, latency, cost,
critical segments, pair coverage, and evidence semantics do not compensate for one another. A
candidate that is relatively better can still be rejected or produce an abstention if an absolute
floor or required evidence surface is not satisfied.

## Decision vocabulary

The historical Change Gate emits `SHIP`, `NO_GO`, `REVIEW`, or `INCONCLUSIVE`. The frozen study
contract maps these to:

```text
SHIP         -> APPROVE
NO_GO        -> REJECT
REVIEW       -> ABSTAIN
INCONCLUSIVE -> ABSTAIN
```

The mapping is implemented in `src/inference_engine/benchmarking/pilot_contract.py`. Historical
artifacts keep their original vocabulary; reconstruction does not relabel prior results.

`ABSTAIN` is a valid bounded decision, not a crashed run. It is also not a success metric:
blanket abstention has no operational utility.

## Canonical evidence index

| Claim | Evidence | Observed | Does not prove |
| --- | --- | --- | --- |
| Paired published-reference change can be gated without erasing uncertainty. | [`ifstruct-json-mode-20260911`](../benchmarks/reports/ifstruct-json-mode-20260911/decision.md) | `INCONCLUSIVE`; structure `0/100 → 0/100`; p50 improved; cost checks had no variation | Quality improvement, invoice truth, official IFStruct leaderboard, customer traffic |
| Relative improvement need not become approval. | [`local-json-mode-20260911`](../benchmarks/reports/local-json-mode-20260911/decision.md) | Exact-field `0/32 → 8/32`; still `INCONCLUSIVE` | Approval; independently Git-proven pre-registration; customer value |
| Exact monetary match does not imply token-field comparability. | [`openrouter-billing-reconciliation-6b-20260911T220449Z`](../benchmarks/reports/openrouter-billing-reconciliation-6b-20260911T220449Z/decision.md) | `VALIDATED_MISMATCH`; four frozen monetary rules matched; R1 token rule did not; `$0.00070845` | Direct OpenAI ledger truth, invoice equivalence, generalized billing correctness |

### IFStruct published-reference change gate

**Claim supported:** the repository can pair a pinned, licensed published-reference subset across
two local execution arms and preserve workload identity, attempt evidence, structural quality,
latency, cost semantics, provenance, and a non-compensatory gate result.

**Recorded result:** `INCONCLUSIVE`.

**Why:** the candidate passed the configured latency and binary-harm checks, but required cost checks
were inconclusive because the paired cost difference had no empirical variation. Both arms also
scored `0/100` on the repository's JSON-path structural validator, so this artifact is not evidence
of broad quality improvement.

Artifacts:

- decision: `benchmarks/reports/ifstruct-json-mode-20260911/decision.json`
- human report: `benchmarks/reports/ifstruct-json-mode-20260911/decision.md`
- frozen experiment: `benchmarks/experiments/ifstruct-json-mode-20260911/experiment.json`
- workload: `benchmarks/workloads/ifstruct-v1.0-json-wrapper-n100.jsonl`
- upstream provenance/license: `benchmarks/references/ifstruct-v1.0/`
- pairing/statistics: `benchmarks/reports/ifstruct-json-mode-20260911/paired-evidence.json`

Limitations are encoded directly in the decision bundle. This is not an official IFStruct
leaderboard result, production evidence, customer evidence, or provider-invoice validation.

### Local synthetic JSON-mode change gate

**Claim supported:** the same decision machinery can represent a real relative improvement without
turning it into an approval when the broader evidence contract is not satisfied.

**Recorded result:** `INCONCLUSIVE`.

The candidate improved exact-field structural acceptance from `0/32` to `8/32`, while successful
latency increased. The declared pre-registration was not independently Git-proven before execution;
the decision preserves that weaker chronology classification instead of upgrading it post hoc.

Artifacts live under `benchmarks/reports/local-json-mode-20260911/` with the frozen experiment under
`benchmarks/experiments/local-json-mode-20260911/`.

### OpenRouter billing reconciliation

**Claim supported:** for one frozen 50-request OpenRouter gateway micro-experiment, the repository
can reconcile several monetary surfaces exactly while preserving a token-semantics mismatch.

**Recorded result:** `VALIDATED_MISMATCH`.

Facts preserved in the report:

- 50 planned / 50 completed / 50 successful requests;
- zero retries;
- response-cost sum, reconstructed OpenRouter tariff cost, generation `total_cost`, and dedicated-key
  usage delta reconciled under the frozen monetary rules;
- dedicated-key usage delta: `$0.00070845`;
- the frozen R1 token comparison mismatched on the prompt-token field;
- direct OpenAI billing-ledger and final-invoice layers were not observed.

Artifacts:

- decision: `benchmarks/reports/openrouter-billing-reconciliation-6b-20260911T220449Z/decision.md`
- structured decision: `benchmarks/reports/openrouter-billing-reconciliation-6b-20260911T220449Z/decision.json`
- frozen experiment: `benchmarks/experiments/openrouter-billing-reconciliation-6b-20260911T220449Z/experiment.json`
- frozen protocol: `docs/11_OPENROUTER_GATEWAY_BILLING_VALIDATION.md`
- post-hoc semantics audit: `docs/12_OPENROUTER_EVIDENCE_SEMANTICS.md`

The post-hoc semantics audit explains the native/normalized token fields but does not rewrite the
historical R1 outcome.

These three examples are the canonical public evidence examples. Other committed files under
`benchmarks/reports/` are retained for historical reproducibility and provenance and must not be
interpreted automatically as additional headline or canonical public claims.

## Publication redactions

Four committed JSON artifacts contained a developer-machine absolute path in `spec_path`. Public
copies replace only that absolute path with the equivalent repository-relative path. Exact original
bytes are retained in the non-public reconstruction review package, and original/published hashes
are recorded in `benchmarks/PUBLICATION_REDACTIONS.json`.

This is a publication transform, not an evidence reinterpretation. Metrics, thresholds, decisions,
usage, cost, timestamps, workload hashes, commit identities, and classifications are unchanged.

## Claims not yet supported

The repository does not currently establish that:

- InferenceLedger makes better real-world migration decisions than a strong conventional workflow;
- a customer has adopted or validated the system;
- any demonstrated savings generalize to production workloads;
- the OpenRouter result is direct OpenAI billing-ledger or invoice validation;
- the local IFStruct or synthetic workloads establish broad semantic quality;
- an `APPROVE` result would by itself prove production readiness.

Those boundaries are product behavior, not marketing disclaimers: when evidence is absent or
semantically incompatible, the system is expected to abstain.
