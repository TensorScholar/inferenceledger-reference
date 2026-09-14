# Real LiteLLM Local Runtime Pilot

**Date:** 2026-09-10 UTC
**Evidence class:** FACT — real local LiteLLM execution-stack telemetry; not provider-invoice evidence

## Source

- LiteLLM: `1.100.1`, installed only in an isolated pilot environment outside the repository.
- Ollama: `0.34.0` at `http://127.0.0.1:11434`.
- Model: `gemma3:270m`, digest `e7d36fb2c3b3293cfe56d55889867a064b3a2b22e98335f2e6e8a387e081d6be`.
- Runtime: GGUF, `268.10M`, `Q8_0`, Apple M2 / Metal.
- LiteLLM call: `model=ollama/gemma3:270m`, `api_base=http://127.0.0.1:11434`, non-streaming completion.
- Capture mechanism: LiteLLM `CustomLogger` callback reading the native `standard_logging_object` emitted by that execution.
- Raw payload: retained outside version control at `/private/tmp/inferenceledger-litellm-pilot/raw/standard_logging_payload.json`.
- Raw payload SHA-256: `b59f39ede3ffd9e93305d4643bc04ae53dd6d7bc555157c01e7283a46df38506`.

The endpoint and installed model were independently verified before the LiteLLM call. The direct Ollama response and the LiteLLM response both completed successfully; this was not a fixture, mock, monkeypatch, or synthetic provider response.

## Observed trace

One source record and one provider attempt were captured:

- `id`: `chatcmpl-e45e1979-01cf-489f-87e3-1da8c47a8c66`
- `trace_id`: `509535a6-c09a-44f2-9a4c-1b6e0bf27437`
- `litellm_call_id`: `7994f489-19f1-4cab-b195-ea8c6ec5e04f`
- status: `success`
- provider/model: `ollama` / `gemma3:270m`
- usage: 24 prompt, 7 completion, 31 total tokens
- response time: `0.5035700798034668` seconds; imported latency `504 ms`
- `response_cost`: `0.0`
- `response_cost_failure_debug_info`: absent
- `cache_hit`: `null`
- retries/fallbacks: none observed; one attempt, zero retries

No genuine retry/fallback chain arose from this zero-cost run. Retry/fallback external validation therefore remains **INSUFFICIENT EVIDENCE / INCONCLUSIVE**.

The committed sanitized source is [litellm-local-gemma3-270m-20260910-sanitized-source.json](litellm-local-gemma3-270m-20260910-sanitized-source.json). Only prompt and response content were redacted. Evidence-bearing identifiers, status, provider/model, timing, usage, cost, cache, and error fields were preserved. The raw source remains outside version control.

## Source-to-ledger audit

The captured payload was passed unchanged to the existing LiteLLM importer. It produced:

- provider `ollama`, model `gemma3:270m`;
- one succeeded `ProviderAttempt`;
- cost evidence `REPORTED_BY_EXECUTION_STACK`, source `litellm_standard_logging_payload`, source record id equal to the native `id`;
- `estimated_cost_usd=0.0`, `cost_evidence_complete=true`;
- `cache_hit=None`, preserving the native unknown cache state;
- request cost equal to the single known reported attempt cost.

The zero cost is a LiteLLM execution-stack report for a local Ollama model. It is not a provider invoice, and no pricing provenance was invented.

## Persistence and reconciliation

- JSONL round-trip: **FACT — PASS**. The restored `RequestTrace` equals the imported trace.
- SQLite trace round-trip: **FACT — PASS**. The restored trace equals the imported trace.
- SQLite provider-usage round-trip: **FACT — PASS**. Provider/model, usage, cost provenance, attempt chain, and `cache_hit=None` agree with the imported trace.
- SQLite report: `0.0 USD`, cost evidence complete for the one observed attempt.
- Reconciliation: **FACT — `missing_route`**. No InferenceLedger route decision was executed in this pilot, so route-versus-execution cost delta is not comparable and remains `null`.

One real compatibility defect was discovered and fixed: the source used `cache_hit=null`, while the importer and persistence model previously coerced that state to `false`. The importer, JSONL model, SQLite schema/migration, and regression tests now preserve unknown cache state as `null`. The provider/model split (`ollama/gemma3:270m` input versus `ollama` + `gemma3:270m` canonical fields) is intentional normalization, not a discrepancy.

## Boundary of the evidence

**FACT:** this pilot proves that one real non-streaming LiteLLM runtime payload from a local Ollama execution can be captured, imported, persisted, and read back without discrepancy.

**INFERENCE:** the observed local execution-stack integration shape is compatible with InferenceLedger.

**INSUFFICIENT EVIDENCE / INCONCLUSIVE:** callback completeness, streaming behavior, real retry/fallback amplification, cloud-provider reliability, provider invoice equivalence, production readiness, commercial validity, and general statistical validity.

This artifact does not start provider/invoice validation or the real migration experiment.
