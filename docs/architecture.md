# Architecture

InferenceLedger is organized around one narrow boundary: **turn execution evidence into a bounded
migration decision without erasing uncertainty or provenance**.

The architecture diagram in the repository README is the canonical system view. This document maps
that view to code. Implementation paths below are relative to `src/inference_engine/` unless they
point at `benchmarks/`.

## Core path

| Stage | Primary implementation | Responsibility |
| --- | --- | --- |
| Frozen change contract | `benchmarking/experiment_provenance.py`, experiment JSON under `benchmarks/experiments/` | Bind workload identity, baseline/candidate definitions, policy, chronology, and pre-registration evidence. |
| Execution / import | `application/services/benchmarking/litellm_arm_service.py`, provider adapters, importers | Execute or import one arm while retaining source semantics. |
| Attempt-chain evidence | `domain/models/execution.py`, `domain/models/economics.py`, `benchmarking/sqlite_ledger.py` | Represent retries/fallback attempts, usage, outcomes, known/unknown monetary evidence, and exact economic provenance. |
| Run evidence | `application/services/benchmarking/run_evidence_service.py` | Persist report, segment, context, and ledger state through one application boundary. |
| Pairing + segments + statistics | `benchmarking/paired_comparison.py`, `segmentation.py`, `statistics.py` | Pair by workload identity, compute coverage, central paired uncertainty, and critical-segment evidence. |
| Semantic validation | `benchmarking/pilot_semantics.py`, `evidence_completeness.py`, `reconciliation.py` | Reject silent gaps and distinguish representational completeness from substantive comparability. |
| Decision | `benchmarking/change_gate.py`, `decision_report.py`, `application/services/benchmarking/migration_decision_service.py` | Apply non-compensatory requirements and emit the authoritative decision bundle. |
| Study output | `benchmarking/pilot_contract.py`, `pilot_analysis.py`, `pilot_decision.py` | Map gate vocabulary to `APPROVE / REJECT / ABSTAIN` and preserve pilot-level evidence contracts. |

## The economic unit is the provider attempt

`ProviderAttempt` is first-class. A retry is not folded into the successful final response, and a
future fallback must appear as another attempt. Each attempt carries exactly one monetary evidence
class:

- `CALCULATED_FROM_USAGE`: amount reconstructed from observed usage plus complete pricing provenance;
- `REPORTED_BY_EXECUTION_STACK`: amount reported by an external execution stack with source identity;
- `UNKNOWN`: no amount is attached.

This prevents a failed or retried request from becoming economically invisible and prevents unknown
cost from being aggregated as zero.

For prospective pilot accounting, `domain/models/economics.py` also provides decimal-string money
surfaces so exact sums do not depend on binary floating-point arithmetic. Statistical analysis may
later consume numeric projections, but the authoritative monetary representation remains explicit.

## The gate is non-compensatory

The Change Gate is an intersection, not a weighted score. A favorable latency result cannot cancel
a failed quality requirement; a relative quality improvement cannot bypass an absolute candidate
floor; an unavailable required evidence surface cannot be converted into a pass.

The current statistical core includes paired workload comparison, exact one-sided binary harm
bounds where applicable, central bootstrap evidence for continuous paired differences, segment
coverage, and explicit `insufficient_variation`/inconclusive states. Tail latency values in existing
decision artifacts are descriptive unless an inferential tail method is explicitly supplied.

## Evidence import is not evidence promotion

External data enters through adapters/importers but retains its source semantics. For example,
LiteLLM `response_cost` is stored as execution-stack-reported evidence; it does not become provider
invoice truth merely because the value is reproducible. The OpenRouter reconciliation experiment
similarly keeps gateway-reported cost, key-usage delta, token surfaces, and unavailable direct
provider/invoice layers distinct.

## Supporting executor code

The repository predates the current assurance thesis and still contains tested execution machinery:
API adapters, routing policies, batching, caching, model backends, and pricing utilities. These are
retained because they support controlled execution, historical tests, or compatibility. They are
**not** a claim that InferenceLedger is a production gateway or observability platform.

The public reconstruction intentionally avoids deleting functioning code solely because it belongs
to an older product framing. Unreferenced monitoring examples that named metrics the repository did
not emit were removed as misleading public surface; the exact source copies remain in the review
package used for reconstruction audit.

## Storage and reproducibility

The reference implementation is local-first:

- JSON/JSONL artifacts preserve experiment and execution evidence;
- SQLite stores benchmark traces, attempt chains, reports, contexts, and provenance-bearing cost;
- workload files are hashed and paired by stable workload item identity;
- decision bundles retain explicit limitations and unsupported claims;
- licensed benchmark material carries nested license/NOTICE/source provenance.

Nothing in this architecture implies a hosted control plane, distributed ledger, billing system of
record, or production-scale data plane.
