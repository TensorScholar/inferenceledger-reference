# OpenRouter gateway billing validation protocol

**Mission:** 6B-OR — OpenRouter request-level gateway billing reconciliation.  
**Classification:** `GATEWAY_BILLING_VALIDATION`.  
**This is not:** `OPENAI_PROVIDER_BILLING_VALIDATION`, `FINAL_INVOICE_VALIDATION`, customer validation, or a Change Gate SHIP.  
**Pre-registration freeze time (UTC):** `2026-09-11T22:04:49Z`.  
**First-party documentation access:** `2026-09-11T22:02:53Z`–`2026-09-11T22:04:49Z`.

Language in this note is tagged:

- **FACT** — observed in this repository or current first-party OpenRouter documentation/catalog
- **INFERENCE** — follows from those facts
- **HYPOTHESIS** — testable by the frozen paid run, not established
- **INSUFFICIENT EVIDENCE** — not verified; must not be treated as true

## 1. Claim boundary

A successful `VALIDATED_MATCH` may establish only that, for this isolated micro-experiment:

1. real paid external inference was executed through OpenRouter;
2. request-level token usage returned through OpenRouter was captured;
3. InferenceLedger independently reconstructed cost from a frozen OpenRouter tariff;
4. reconstructed cost can be compared with OpenRouter response-level accounting;
5. OpenRouter response-level accounting can be compared with OpenRouter generation-level accounting;
6. summed request/generation costs can be compared with the cumulative usage delta of the dedicated OpenRouter key;
7. OpenRouter can report which upstream provider served each generation;
8. OpenRouter may report an upstream inference cost.

Even if every required comparable surface matches, this mission does **not** establish:

- direct OpenAI billing;
- OpenAI Costs API agreement;
- OpenAI invoice truth;
- independent verification of OpenRouter's upstream cost against OpenAI;
- universal OpenRouter accuracy;
- all OpenRouter models/providers;
- production economics;
- customer value;
- commercial readiness;
- Change Gate SHIP;
- InferenceLedger production readiness.

## 2. Verified OpenRouter contracts (Phase 1)

| Topic | Source URL | Access (UTC) | Current semantics used by this freeze |
| --- | --- | --- | --- |
| Chat Completions | `https://openrouter.ai/api/v1/chat/completions` (OpenAI-compatible; model page `https://openrouter.ai/openai/gpt-4o-mini-2024-07-18`) | 2026-09-11 | Non-streaming `POST`. Model snapshot `openai/gpt-4o-mini-2024-07-18`. |
| Current key | `https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key` and `https://openrouter.ai/docs/api_reference/limits` | 2026-09-11 | `GET /api/v1/key` returns `limit`, `limit_remaining`, `limit_reset`, `usage`, BYOK usage, `is_management_key`, `is_provisioning_key`. `limit` may be `null` if unlimited. |
| Generation metadata | `https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation` | 2026-09-11 | `GET /api/v1/generation?id=` exposes `provider_name`, token fields, `total_cost`, `upstream_inference_cost`, `service_tier`, `is_byok`. |
| Usage accounting | `https://openrouter.ai/docs/cookbook/administration/usage-accounting` | 2026-09-11 | Non-streaming responses automatically include usage. Deprecated `usage: {include: true}` is not required. `usage` may include `prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_tokens_details.cached_tokens`, `prompt_tokens_details.cache_write_tokens`, `cost`, `cost_details.upstream_inference_cost`. Token counts are described as native-tokenizer counts. |
| Provider routing | `https://openrouter.ai/docs/guides/routing/provider-selection` | 2026-09-11 | `provider.only`, `provider.order`, and `provider.allow_fallbacks` are current. Base slug `openai` does **not** match service-tier endpoints such as `openai/fast`. |
| Response caching | `https://openrouter.ai/docs/guides/features/response-caching` | 2026-09-11 | Opt-in via `X-OpenRouter-Cache: true` or presets. Default is off. Distinct from upstream prompt caching. |
| Model catalog / endpoints | `https://openrouter.ai/api/v1/models` and `https://openrouter.ai/api/v1/models/openai/gpt-4o-mini-2024-07-18/endpoints` | 2026-09-11T22:02:53.267052Z | Snapshot exists. One endpoint: `provider_name=OpenAI`, `tag=openai`. Chat Completions parameters include `max_tokens` and `temperature`. |

**FACT — A.** Detailed usage is automatic on non-streaming responses. This experiment does not send the deprecated include flag.

**FACT — B/C/D.** The usage, generation, and key fields listed in the mission brief are present in current first-party docs.

**FACT — E.** Strongest unambiguous restriction used here:

```json
"provider": {"only": ["openai"], "order": ["openai"], "allow_fallbacks": false}
```

**FACT — F.** OpenRouter response caching is separate from upstream prompt caching. This experiment sends neither `X-OpenRouter-Cache` nor a cache-enabling preset.

## 3. Frozen model and tariff

Target model: `openai/gpt-4o-mini-2024-07-18` (exact snapshot; not `openai/gpt-4o-mini`, not auto-router).

**FACT** from `GET https://openrouter.ai/api/v1/models/openai/gpt-4o-mini-2024-07-18/endpoints` at `2026-09-11T22:02:53.267052+00:00`:

| Component | Catalog field | Per token | Per 1,000,000 tokens |
| --- | --- | --- | --- |
| Input | `pricing.prompt` | `0.00000015` | `0.15` |
| Cached input | `pricing.input_cache_read` | `0.000000075` | `0.075` |
| Output | `pricing.completion` | `0.0000006` | `0.60` |
| Cache write | not listed | `null` (not assumed zero if writes appear) | not listed |
| Per-request fee | `per_request_limits` on catalog row is `null`; endpoint pricing has `discount: 0` and no request-fee field | none observed | none observed |

Endpoint count: **1**. Upstream provider name: **OpenAI**. `supports_implicit_caching`: **false**.

Pricing provenance is experiment-local. Production `DEFAULT_PRICING` / `openai-standard-2026-09-03` is direct-OpenAI identity and is not reused.

If a response reports `cache_write_tokens > 0`, reconstructed cost B is `INCONCLUSIVE` because this snapshot did not list a write rate.

## 4. Identity

| Concept | Frozen value |
| --- | --- |
| Execution gateway | `openrouter` |
| Requested model | `openai/gpt-4o-mini-2024-07-18` |
| Expected reported upstream provider | `OpenAI` / `openai` |
| Direct upstream billing validated | `false` |
| Final invoice validated | `false` |

`execution_provider` must not become `openai` merely because OpenAI served the underlying model.

## 5. Spend contract

Three independent ceilings. None is a target.

| Level | Amount USD | Role |
| --- | --- | --- |
| User authorization | `0.20` | Maximum the user authorized. Not a spend target. |
| Provider-side dedicated key | `<= 0.05` | Independently re-observed via `GET /api/v1/key`. If `limit` is null or `limit > 0.05`, **no inference**. |
| Software guard | `0.025` | Local stop threshold. Do not intentionally spend this amount. |

Conservative full-run maximum, calculated from the frozen tariff before execution:

- 50 requests × 3000 UTF-8 prompt bytes treated as a token upper bound × `0.00000015`
- plus 50 × 8 output tokens × `0.0000006`
- equals **`0.02274`**, which is **strictly below** `0.025`

Every serialized user prompt must be `<= 3000` UTF-8 bytes.

Runtime rule before each next request:

`cumulative(response usage.cost) + conservative_max(next request) < 0.025`

The next-request conservative amount is the greater of the actual next prompt's byte-bound cost and the frozen 3000-byte per-request bound.

## 6. Runtime freeze

| Field | Frozen value |
| --- | --- |
| Planned paid requests | 50 |
| Temperature | 0 |
| Stream | false |
| Max output tokens | 8 |
| Tools / web search / images / audio / files / reasoning / plugins / structured-output surcharge | forbidden |
| Application retries | 0 |
| SDK retries | 0 |
| Provider fallbacks | false |
| Response-cache opt-in | false |
| Service tier request | omitted / default. Do not request flex, priority, or fast. Do not use `:nitro` or `:floor`. |
| BYOK | forbidden |
| Continuation after inference failure | **not allowed**. Record the failure and STOP. Do not replace items. Do not exceed 50. |
| First-request gate | After request 1 and its generation metadata: expected snapshot, upstream OpenAI, acceptable service tier, response `usage.cost` present, generation `total_cost` present. Any miss → STOP before request 2. |

Generation metadata GET polling (not inference):

- max attempts 10
- initial delay 1s, multiplier 2, max delay 8s

Key usage settlement polling (not inference):

- max attempts 12
- delay 5s
- stable reads 2

If usage never settles to an exact match: `INCONCLUSIVE`. Do not manufacture a `MATCH`.

## 7. Evidence layers

| Layer | Name | Meaning |
| --- | --- | --- |
| A | `PROVIDER_RESPONSE_USAGE` | OpenRouter response usage tokens |
| B | `INFERENCELEDGER_RECONSTRUCTED_OPENROUTER_COST` | A × frozen OpenRouter tariff |
| C | `OPENROUTER_RESPONSE_REPORTED_COST` | response `usage.cost` |
| D | `OPENROUTER_GENERATION_TOTAL_COST` | generation `total_cost` |
| E | `OPENROUTER_DEDICATED_KEY_USAGE_DELTA` | post-run key `usage` − pre-run key `usage` |
| F | `OPENROUTER_REPORTED_UPSTREAM_INFERENCE_COST` | OpenRouter-reported upstream cost |
| G | `DIRECT_UPSTREAM_PROVIDER_BILLING` | **NOT OBSERVED** |
| H | `FINAL_INVOICE` | **NOT OBSERVED** |

Never collapse F into G. Never collapse E into H. Missing optional fields are `null`, not zero.

## 8. Reconciliation policy (frozen before results)

Classifications: `MATCH`, `MISMATCH`, `INCONCLUSIVE`, `NOT_COMPARABLE`. Do not use `PASS`.

**R1** — comparable billed token fields only: response `prompt_tokens` vs generation `tokens_prompt`; response `completion_tokens` vs generation `tokens_completion`; cached tokens only when both sides are present. Exact integer equality. Native tokenizer fields are preserved separately and are not forced equal when docs describe different families.

**R2** — B vs C, exact `Decimal` equality when reconstruction is possible. Missing cached split or unlisted cache-write charge → `INCONCLUSIVE` / `NOT_COMPARABLE` as specified in code. No post-hoc epsilon.

**R3** — C vs D, exact `Decimal` equality.

**R4** — sum(C) vs E, exact `Decimal` equality after settlement. Other-key traffic → `INCONCLUSIVE`.

**R5** — sum(D) vs E, same rule.

**R6 / F** — informational. Allowed interpretation: OpenRouter reported upstream provider X and upstream cost Y. Not allowed: OpenAI independently billed Y.

Purchase/credit-buy fees are out of scope. This experiment uses existing credits.

## 9. Final experiment classification

Exactly one of: `VALIDATED_MATCH`, `VALIDATED_MISMATCH`, `INCONCLUSIVE`, `BLOCKED`.

`VALIDATED_MATCH` requires isolated dedicated-key preflight, upstream provider consistent with the frozen route, R1–R5 `MATCH`, no unknown material billable component, spend inside all limits, complete provenance, and Git-proven pre-registration.

`VALIDATED_MISMATCH` is used when a required comparable surface disagrees.

`INCONCLUSIVE` is used for incomplete evidence, contamination, incomparable semantics, unsettled accounting, unexpected routing, or missing required cost fields.

`BLOCKED` is used when no defensible paid experiment could begin.

## 10. Git-proven pre-registration

Before first paid inference: harness, tests, workload, pricing snapshot, this protocol, and the experiment spec are committed; worktree clean; spec bytes equal HEAD; branch pushed; self-hosted CI passed; commit SHA recorded.

CI must never execute the paid OpenRouter experiment. Paid execution is local and requires `--execute-paid`.

## 11. Raw vs sanitized evidence

Raw `/api/v1/key`, chat, and generation payloads stay outside Git under:

`/private/tmp/inferenceledger-openrouter-6b/<experiment-id>/`

Committed reports contain sanitized metadata, SHA-256 of raw bytes, and UTC acquisition times. They must not contain API keys, Authorization headers, key labels, or creator user ids.

## 12. Architecture choice

No production `PricingTable` row is added. An experiment-local pricing snapshot is consumed by the harness so this OpenRouter gateway execution cannot inherit direct-OpenAI provenance. `ProviderAttempt` is not widened; gateway vs upstream identity lives in experiment artifacts. The only production-adjacent addition is `require_clean_git_worktree` for the execution boundary.
