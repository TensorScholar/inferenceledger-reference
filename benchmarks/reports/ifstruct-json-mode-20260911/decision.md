# Migration Decision `ifstruct-json-mode-20260911`

**Final decision:** `INCONCLUSIVE`

This artifact reconstructs one baseline → candidate Change Gate decision from paired
workload items, imported executions, and attempt-level cost evidence.

## Identity

- Generated at (UTC): `2026-09-11T02:08:27.558603+00:00`
- Classification: `PUBLISHED_REFERENCE`
- Workload: `benchmarks/workloads/ifstruct-v1.0-json-wrapper-n100.jsonl`
- Workload SHA-256: `35648eb88ea2b07008113e913b406186f5c9399cff3cf88bf853600fed2f7192`
- Workload items: `100`

## Runtime

- endpoint: `http://127.0.0.1:11434`
- host_arch: `arm64`
- litellm_version: `1.100.1`
- model_argument: `ollama/gemma3:270m`
- model_digest: `e7d36fb2c3b3293cfe56d55889867a064b3a2b22e98335f2e6e8a387e081d6be`
- model_digest_prefix: `e7d36fb2c3b3293c`
- ollama_version: `0.34.0`
- python_version: `3.12.10`
- runtime_note: `Apple M2 / Metal local Ollama. LiteLLM installed only in an isolated environment outside the repository. Cost is REPORTED_BY_EXECUTION_STACK.`
- execution_interval: `{"earliest_start_utc": "2026-09-11T01:22:32.043512+00:00", "end_field": "endTime", "latest_end_utc": "2026-09-11T02:07:45.932555+00:00", "payload_count": 200, "source": "litellm_standard_logging_payload", "start_field": "startTime"}`

## Pre-registration

- Declared status: `PRE_REGISTERED_BEFORE_EXECUTION`
- Declared locked_at_utc: `None`
- Git-proven before execution: `True`
- Evidence class: `GIT_PROVEN`
- Reason: experiment specification was committed and unmodified in Git, and the source commit timestamp precedes captured execution telemetry

## Baseline

- Run id: `ifstruct-json-mode-20260911-baseline`
- Definition: LiteLLM non-streaming completion to ollama/gemma3:270m with temperature=0.0, max_tokens=1024, and no response_format. Structure is requested only by the published IFStruct prompt. This matches IFStruct's unconstrained decoding setting for the JSON-wrapper subset.
- Rationale: Official IFStruct scores unconstrained generations. The baseline therefore does not enable execution-stack JSON mode.

## Candidate

- Run id: `ifstruct-json-mode-20260911-candidate`
- Definition: Same model, endpoint, temperature, max_tokens, and prompts as baseline, plus LiteLLM response_format={"type":"json_object"}. JSON Schema constrained decoding is not enabled.
- Rationale: Enabling generic JSON mode is the smallest technically valid structured-output policy change that this stack can express. It is not equivalent to enforcing each item's JSON Schema.

## Execution counts

- Baseline requests: 100
- Candidate requests: 100
- Baseline successes/failures: 100/0
- Candidate successes/failures: 100/0
- Baseline attempts/retries: 100/0
- Candidate attempts/retries: 100/0

## Pairing audit

- Pairing key: `workload_item_id`
- Baseline count: 100
- Candidate count: 100
- Matched pairs: 100
- Coverage: 1.0
- Unmatched baseline ids: `()`
- Unmatched candidate ids: `()`
- Duplicate ids present: `()`
- Ambiguous: `False`

## Quality

- Evaluator: `IFStruct JSON-path structure checks (parse, code-block, commentary, wrapper, item count, JSON Schema). Binary pass only if all stages have zero errors. Not semantic quality.`
- Baseline pass rate: 0.0 (0/100)
- Candidate pass rate: 0.0 (0/100)

## Latency

- Baseline p50/p95: 14226 / 32220 ms
- Candidate p50/p95: 9912 / 25626 ms
- Note: Observed nearest-rank percentiles only. This report does not emit inferential p95/p99 confidence intervals.

## Cost evidence

- Baseline: `{'attempt_cost_kinds': {'reported_by_execution_stack': 100}, 'cost_sources': {'litellm_standard_logging_payload': 100}, 'unknown_attempt_count': 0, 'request_cost_evidence_complete': True, 'reported_total_usd': 0.0}`
- Candidate: `{'attempt_cost_kinds': {'reported_by_execution_stack': 100}, 'cost_sources': {'litellm_standard_logging_payload': 100}, 'unknown_attempt_count': 0, 'request_cost_evidence_complete': True, 'reported_total_usd': 0.0}`
- Rule: LiteLLM response_cost remains REPORTED_BY_EXECUTION_STACK. Local numeric zero is not provider-invoice evidence.

## Change Gate

- Decision: `INCONCLUSIVE`
- Pass/fail/review/inconclusive: 6/0/0/2


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
- official IFStruct leaderboard comparison
- JSON mode equals JSON Schema constrained decoding
- p95/p99 inferential latency claims
- SHIP is not production readiness
- local reported zero cost is not invoice validation
- this workload is not a customer or production workload
