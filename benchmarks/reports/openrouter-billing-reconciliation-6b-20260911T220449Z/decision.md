# OpenRouter gateway billing reconciliation `openrouter-billing-reconciliation-6b-20260911T220449Z`

**Classification:** `VALIDATED_MISMATCH`

This is `GATEWAY_BILLING_VALIDATION`. It is not OpenAI billing-ledger validation,
not invoice validation, and not a Change Gate SHIP.

- Execution gateway: `openrouter`
- Requested model: `openai/gpt-4o-mini-2024-07-18`
- Provider routing: `provider.only=["openai"]`, `provider.order=["openai"]`, `allow_fallbacks=false`
- Protocol / source commit: `6a8522a8921bfd3b1d86ba6fa953e508a8660139`
- Spec SHA-256: `50248d33884b0e77fc5ff0eab2e05b9fa30deaa871f4aad16ff0bc4db1f56f39`
- Workload SHA-256: `2b986ccfd0f23cb0c8d6ea2f87a38358d55760ffe79a81b5fa541c08ab90f24f`
- Pre-registration: `GIT_PROVEN`; commit timestamp `2026-09-11T22:19:57Z` precedes first paid start `2026-09-11T22:26:13.558943Z`
- Planned requests: `50`
- Completed requests: `50`
- Successful requests: `50`
- Failures: `0`
- Retries: `0`
- Response cost sum C: `0.00070845`
- Reconstructed cost sum B: `0.000708450`
- Generation total_cost sum D: `0.00070845`
- Key usage delta E: `0.00070845`
- Reported generation `upstream_inference_cost` sum F: `0`
- Direct OpenAI billing validated: `False`
- Final invoice validated: `False`

## Rules

- `R1`: `MISMATCH` — frozen comparable fields were response `prompt_tokens`/`completion_tokens` vs generation `tokens_prompt`/`tokens_completion`
- `R2`: `MATCH` — reconstructed OpenRouter tariff cost vs response `usage.cost`
- `R3`: `MATCH` — response `usage.cost` vs generation `total_cost`
- `R4`: `MATCH` — exact Decimal equality of sum(C) vs dedicated-key usage delta E
- `R5`: `MATCH` — exact Decimal equality of sum(D) vs E

## R1 observation

**FACT:** all 50 successful requests have:

- response `prompt_tokens` == generation `native_tokens_prompt` (range 84–96)
- response `completion_tokens` == generation `tokens_completion` == generation `native_tokens_completion` == 1
- generation `tokens_prompt` == 64 on every request
- therefore response `prompt_tokens` != generation `tokens_prompt` on every request

The frozen R1 rule compared response usage tokens with generation `tokens_prompt`/`tokens_completion` as the comparable billed fields, and preserved native vs normalized prompt divergence as not forced equal. Under that frozen rule, R1 is `MISMATCH`. This report does not relabel that result after seeing the native-token agreement.

## Cache accounting

**FACT:** all 50 requests report `cached_tokens=0`, `cache_write_tokens=0`, and generation `native_tokens_cached=0`. Generation `cache_discount` is null, not zero.

## Upstream cost F

**FACT:** generation metadata `upstream_inference_cost` is `0` on all 50 requests (`is_byok=false`). Sum F from that field is `0`.

**FACT:** response `usage.cost_details.upstream_inference_cost` equals response `usage.cost` on all 50 requests and therefore also equals C/D/E (`0.00070845`). That field is OpenRouter-reported, not independent OpenAI billing.

**NOT ALLOWED:** treating either F surface as OpenAI ledger or invoice truth. Layers G and H were not observed.

## Other observations

- **FACT:** reported upstream provider is `OpenAI` on all 50 generations; requested and returned model is `openai/gpt-4o-mini-2024-07-18`.
- **FACT:** `is_byok=false` on all 50 generations.
- **FACT:** chat responses are `object=chat.completion` with `stream=false` requested. Generation metadata reports `streamed=true` on all 50. This was not a frozen halt condition; it is recorded, not repaired after the fact.
- **FACT:** `service_tier` is null on all 50 responses/generations.
- **FACT:** dedicated-key pre-run `usage=0`, `limit=0.05`, `limit_remaining=0.05`; post-run `usage=0.00070845`, `limit_remaining=0.04929155`.

## Spend

Actual OpenRouter gateway usage deducted on the dedicated key: USD `0.00070845`.

- below software guard USD 0.025: YES
- below provider-side key limit USD 0.05: YES
- below user authorization USD 0.20: YES
