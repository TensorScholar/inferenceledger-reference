# Migration Decision `local-json-mode-20260911`

**Final decision:** `INCONCLUSIVE`

This artifact reconstructs one baseline → candidate Change Gate decision from paired
workload items, imported executions, and attempt-level cost evidence.

## Identity

- Generated at (UTC): `2026-09-11T00:08:08.732495+00:00`
- Classification: `SYNTHETIC`
- Workload: `benchmarks/workloads/json-contract-synthetic-v1.jsonl`
- Workload SHA-256: `04157fc5e5c084c25c7705c6f729b5697f3667ed2086d2291fdcb3762a039933`
- Workload items: `32`

## Runtime

- litellm_version: `1.100.1`
- ollama_version: `0.34.0`
- model_argument: `ollama/gemma3:270m`
- model_digest_prefix: `e7d36fb2c3b3293c`
- endpoint: `http://127.0.0.1:11434`
- host_arch: `arm64`
- runtime_note: `Apple M2 / Metal local Ollama. LiteLLM installed only in an isolated environment outside the repository.`
- withdrawn_runtime_fields: `{"captured_at_utc": {"reason": "captured_at_utc is later than generated_at_utc and cannot be a capture timestamp for this decision artifact", "status": "invalid", "value": "2026-09-11T00:30:00+00:00"}}`
- execution_interval: `{"baseline_payload_count": 32, "candidate_payload_count": 32, "earliest_start_utc": "2026-09-11T00:06:44.229721+00:00", "end_field": "endTime", "latest_end_utc": "2026-09-11T00:07:38.755322+00:00", "payload_count": 64, "source": "litellm_standard_logging_payload", "start_field": "startTime"}`
- artifact_generated_at_utc: `2026-09-11T00:08:08.732495+00:00`
- pre_registration: `{"declared_locked_at_utc": "2026-09-11T00:00:00+00:00", "declared_status": "PRE_REGISTERED_BEFORE_EXECUTION", "evidence_class": "INSUFFICIENT_EVIDENCE", "git_proven_before_execution": false, "reason": "The specification declared PRE_REGISTERED_BEFORE_EXECUTION, but the specification and result artifacts first entered Git history together in commit 67825bb5e533fe9e301fc7a29b1ae741affbbc75 at 2026-09-11T00:12:06Z, after source telemetry latest endTime 2026-09-11T00:07:38.755322Z. Git cannot independently prove the policy was immutable before execution.", "source_commit_sha": null, "spec_commit_sha": null, "spec_committed_and_clean": false, "spec_sha256": "feb7f9ab2c6e11feaca2aa65a19eeccef50dc7f5f65318c7ee2666402d8196c0", "workload_sha256": "04157fc5e5c084c25c7705c6f729b5697f3667ed2086d2291fdcb3762a039933"}`

## Pre-registration

- Declared status: `PRE_REGISTERED_BEFORE_EXECUTION`
- Declared locked_at_utc: `2026-09-11T00:00:00+00:00`
- Git-proven before execution: `False`
- Evidence class: `INSUFFICIENT_EVIDENCE`
- Reason: The specification declared PRE_REGISTERED_BEFORE_EXECUTION, but the specification and result artifacts first entered Git history together in commit 67825bb5e533fe9e301fc7a29b1ae741affbbc75 at 2026-09-11T00:12:06Z, after source telemetry latest endTime 2026-09-11T00:07:38.755322Z. Git cannot independently prove the policy was immutable before execution.

## Baseline

- Run id: `local-json-mode-20260911-baseline`
- Definition: LiteLLM non-streaming completion to ollama/gemma3:270m with temperature=0.0, max_tokens=128, and no response_format. JSON shape is requested only in the prompt.
- Rationale: This is the current prompt-only structured-output practice for the JSON-contract workload.

## Candidate

- Run id: `local-json-mode-20260911-candidate`
- Definition: Same model, endpoint, temperature, max_tokens, and prompts as baseline, plus LiteLLM response_format={"type":"json_object"} so the execution stack requests JSON mode.
- Rationale: Enabling execution-stack JSON mode is a real inference-policy change teams actually consider for JSON-contract tasks. The workload is held constant.

## Execution counts

- Baseline requests: 32
- Candidate requests: 32
- Baseline successes/failures: 32/0
- Candidate successes/failures: 32/0
- Baseline attempts/retries: 32/0
- Candidate attempts/retries: 32/0

## Pairing audit

- Pairing key: `workload_item_id`
- Baseline count: 32
- Candidate count: 32
- Matched pairs: 32
- Coverage: 1.0
- Unmatched baseline ids: `[]`
- Unmatched candidate ids: `[]`
- Duplicate ids present: `[]`
- Ambiguous: `False`

## Quality

- Evaluator: `json_field_equals exact JSON field equality; not semantic quality`
- Baseline pass rate: 0.0 (0/32)
- Candidate pass rate: 0.25 (8/32)

## Latency

- Baseline p50/p95: 378 / 734 ms
- Candidate p50/p95: 449 / 1249 ms
- Note: Observed nearest-rank percentiles only. This report does not emit inferential p95/p99 confidence intervals.

## Cost evidence

- Baseline: `{'attempt_cost_kinds': {'reported_by_execution_stack': 32}, 'cost_sources': {'litellm_standard_logging_payload': 32}, 'reported_total_usd': 0.0, 'request_cost_evidence_complete': True, 'unknown_attempt_count': 0}`
- Candidate: `{'attempt_cost_kinds': {'reported_by_execution_stack': 32}, 'cost_sources': {'litellm_standard_logging_payload': 32}, 'reported_total_usd': 0.0, 'request_cost_evidence_complete': True, 'unknown_attempt_count': 0}`
- Rule: LiteLLM response_cost remains REPORTED_BY_EXECUTION_STACK. Local numeric zero is not provider-invoice evidence.

## Change Gate

- Decision: `INCONCLUSIVE`
- Pass/fail/review/inconclusive: 2/0/4/6

- `overall:mean_cost_delta` [inconclusive] statistical evidence status is insufficient_variation (observed=0.0, bound=None, threshold=0.0)
- `overall:failure_harm` [review] observed harm is within margin but exact uncertainty bound crosses the boundary (observed=0.0, bound=0.08936819898626476, threshold=0.05)
- `overall:mean_successful_latency_delta` [pass] upper confidence bound is within the configured margin (observed=137.40625, bound=363.7049572112619, threshold=750.0)
- `overall:accepted_outcome_harm` [review] observed harm is within margin but exact uncertainty bound crosses the boundary (observed=0.0, bound=0.08936819898626476, threshold=0.05)
- `overall:mean_provider_attempt_delta` [inconclusive] statistical evidence status is insufficient_variation (observed=0.0, bound=None, threshold=0.0)
- `overall:mean_provider_retry_delta` [inconclusive] statistical evidence status is insufficient_variation (observed=0.0, bound=None, threshold=0.0)
- `segment:output_contract=json:mean_cost_delta` [inconclusive] statistical evidence status is insufficient_variation (observed=0.0, bound=None, threshold=0.0)
- `segment:output_contract=json:failure_harm` [review] observed harm is within margin but exact uncertainty bound crosses the boundary (observed=0.0, bound=0.08936819898626476, threshold=0.05)
- `segment:output_contract=json:mean_successful_latency_delta` [pass] upper confidence bound is within the configured margin (observed=137.40625, bound=370.5995311572383, threshold=750.0)
- `segment:output_contract=json:accepted_outcome_harm` [review] observed harm is within margin but exact uncertainty bound crosses the boundary (observed=0.0, bound=0.08936819898626476, threshold=0.05)
- `segment:output_contract=json:mean_provider_attempt_delta` [inconclusive] statistical evidence status is insufficient_variation (observed=0.0, bound=None, threshold=0.0)
- `segment:output_contract=json:mean_provider_retry_delta` [inconclusive] statistical evidence status is insufficient_variation (observed=0.0, bound=None, threshold=0.0)

## Limitations

- Intervals are candidate-minus-baseline central mean-difference uncertainty estimates under empirical workload-item resampling; they are not production guarantees.
- Successful-latency evidence is conditioned on workload items that succeeded in both runs; failure-rate evidence must be interpreted alongside it.
- Accepted-outcome evidence is emitted only where request-level acceptance is determinable in both runs; incomplete quality evidence reduces pair coverage.
- Failure-rate and accepted-outcome intervals use paired BCa risk-difference approximation and require at least 10 discordant pairs; they are not exact McNemar-compatible intervals.
- Samples with zero empirical paired-difference variance retain the observed effect but suppress confidence intervals rather than claiming degenerate certainty.
- High-tail latency inference, including p95/p99 confidence intervals, is intentionally not produced by this central paired bootstrap primitive.
- Configured sample and discordant-pair floors are conservative product policies, not universal statistical power guarantees.
- This is an intersection benchmark gate: favorable checks never offset failed or inconclusive required checks.
- SHIP means the configured replayable benchmark gate passed; it is not a production reliability or causal-effect guarantee.
- Observed p95/p99 checks are descriptive hard guards unless a future tail-inference method is explicitly supplied.

## Unsupported claims

- production proven
- invoice validated
- commercially validated
- customer-workload validated
- p95/p99 inferential latency claims
- SHIP is not production readiness
- local reported zero cost is not invoice validation
- synthetic workload is not a customer workload
