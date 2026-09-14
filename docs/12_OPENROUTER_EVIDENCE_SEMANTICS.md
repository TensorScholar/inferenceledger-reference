# OpenRouter evidence semantics (Mission 6C)

**Mission:** 6C — post-hoc OpenRouter evidence-semantics resolution.  
**Spend:** **$0**. Zero OpenRouter, inference, account, metadata, or paid calls.  
**Historical experiment:** `openrouter-billing-reconciliation-6b-20260911T220449Z` (Mission 6B-OR).  
**Official-docs access:** 2026-09-12. OpenAPI `3.1.0`, title `OpenRouter API`, version `1.0.0`.

This note is **not** a rerun of Mission 6B-OR and **does not** relabel its stored result.

Language in this note is tagged:

- **FACT** — observed in this repository, the frozen 6B-OR artifacts, or current first-party OpenRouter schema/docs
- **INFERENCE** — follows from those facts
- **HYPOTHESIS** — plausible, not established
- **INSUFFICIENT EVIDENCE** — not verified; must not be treated as true
- **POST-HOC** — interpretation after seeing 6B-OR results; not retroactive pre-registration

## 1. Historical result invariant

**FACT:** Mission 6B-OR frozen R1 failed. Stored R1 status is `MISMATCH`.

**FACT:** Historical overall classification remains `VALIDATED_MISMATCH`. Canonical files were not edited:

- `benchmarks/reports/openrouter-billing-reconciliation-6b-20260911T220449Z/decision.json`
- `benchmarks/reports/openrouter-billing-reconciliation-6b-20260911T220449Z/decision.md`
- `benchmarks/reports/openrouter-billing-reconciliation-6b-20260911T220449Z/reconciliation.json`
- request evidence, generation evidence, experiment spec, pricing snapshot, raw-evidence manifest

**INFERENCE:** The economic B→C→D→E chain reconciled independently of R1. Exact Decimal equality:

- reconstructed tariff cost B using response/native prompt tokens = `0.000708450`
- response `usage.cost` C = `0.00070845`
- generation `total_cost` D = `0.00070845`
- dedicated-key usage delta E = `0.00070845`

**POST-HOC:** Any clarified semantic interpretation below is not retroactive pre-registration. Frozen 6B-OR R1 compared response `prompt_tokens` with generation `tokens_prompt` as “comparable billed token fields.” That rule still returns `MISMATCH`.

## 2. Official field descriptions (2026-09-12)

Sources, in the mission-required order:

1. OpenAPI JSON/YAML: `https://openrouter.ai/openapi.json` and `https://openrouter.ai/openapi.yaml`
2. API reference: `https://openrouter.ai/docs/api_reference/overview`
3. Usage accounting: `https://openrouter.ai/docs/cookbook/administration/usage-accounting`
4. Generation metadata reference: `https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation`
5. Official SDK field table: `https://openrouter.ai/docs/client-sdks/python/components/generationresponsedata`

### 2.1 Chat Completions response usage

| Field | OpenAPI `ChatUsage` description | API reference / usage-accounting prose |
| --- | --- | --- |
| `usage.prompt_tokens` | “Number of tokens in the prompt” | TypeScript comment: “Including images, input audio, and tools if any.” Prose: token counts in the completions response are calculated using the model’s native tokenizer; credit usage and model pricing are based on these native token counts. |
| `usage.completion_tokens` | “Number of tokens in the completion” | “The tokens generated” |
| `usage.total_tokens` | “Total number of tokens” | “Sum of the above two fields” |
| `usage.prompt_tokens_details.cached_tokens` | “Cached prompt tokens” | “Tokens cached by the endpoint”; usage-accounting: tokens read from cache |
| `usage.cost` | “Cost of the completion” | TypeScript: “Cost in credits (optional).” Usage-accounting: “The total amount charged to your account.” |
| `usage.cost_details` | `$ref: CostDetails` (“Breakdown of upstream inference costs”) | — |
| `usage.cost_details.upstream_inference_cost` | **no description in OpenAPI** | Usage-accounting: “The actual cost charged by the upstream AI provider.” |

**FACT:** OpenAPI `ChatUsage.prompt_tokens` is as thin as generation `tokens_prompt` (“Number of tokens in the prompt”).

**FACT:** Separate official prose (API reference “Querying cost and stats” and usage-accounting) states that completions-response token counts use the model’s native tokenizer and that credits/pricing are based on those counts.

**INFERENCE:** Response `prompt_tokens` is the billed native-tokenizer surface for OpenRouter credits. Code label: `BILLING_OBSERVED`. That label is not licensed by the OpenAPI one-line description alone.

### 2.2 Generation metadata

Official OpenAPI `GenerationResponse.data` descriptions:

| Field | Official description |
| --- | --- |
| `tokens_prompt` | “Number of tokens in the prompt” |
| `tokens_completion` | “Number of tokens in the completion” |
| `native_tokens_prompt` | “Native prompt tokens as reported by provider” |
| `native_tokens_completion` | “Native completion tokens as reported by provider” |
| `native_tokens_cached` | “Native cached tokens as reported by provider” |
| `total_cost` | “Total cost of the generation in USD” |
| `usage` | “Usage amount in USD” |
| `upstream_inference_cost` | “Cost charged by the upstream provider” |
| `streamed` | “Whether the response was streamed” |

**FACT:** `native_tokens_prompt` is explicitly provider-reported native prompt tokens.

**FACT:** `tokens_prompt` is **not** described as provider-native, gateway-normalized, or billed. The word “tokens” in the name is not a semantics contract.

**INSUFFICIENT EVIDENCE:** official documentation does not establish that `tokens_prompt` is comparable with response `prompt_tokens`, nor that it is a billed field, nor that it is a gateway-normalized tokenizer count.

**INFERENCE:** Cross-surface comparison of `tokens_prompt` with response `prompt_tokens` is `NOT_COMPARABLE` under future rules, not a billed `MISMATCH`.

Do not promote third-party SDK commentary (for example OpenRouterTeam skills text that restates “Prompt token count”) to official truth.

## 3. Post-hoc 6B token economics

Computed from committed sanitized 6B-OR evidence only. Frozen tariff: input `0.00000015` USD/token, cached input `0.000000075`, output `0.0000006`. Cached tokens were 0 on all 50.

| Surface | Sum | Notes |
| --- | --- | --- |
| response `prompt_tokens` | **4523** | range 84–96 |
| generation `tokens_prompt` | **3200** | 64 on all 50 |
| generation `native_tokens_prompt` | **4523** | equals response prompt tokens on all 50 |
| response `completion_tokens` | **50** | 1 on all 50 |
| generation `tokens_completion` | **50** | 1 on all 50 |
| generation `native_tokens_completion` | **50** | 1 on all 50 |

Hypothetical reconstructed totals under the frozen tariff (cached = 0, completion = 1 per request):

| Prompt token surface used | Hypothetical reconstructed sum | Equals C/D/E? |
| --- | --- | --- |
| generation `tokens_prompt` (always 64) | **0.000510000** | **no** |
| response `prompt_tokens` / generation `native_tokens_prompt` | **0.000708450** | **yes** (`C=D=E=0.00070845`) |

**FACT:** The monetary surfaces that matched OpenRouter charged cost and dedicated-key usage are response/native prompt tokens, not `tokens_prompt`.

**POST-HOC INFERENCE:** Historical R1 exposed a **semantic mismatch** between an undocumented generation integer (`tokens_prompt`) and the billed native-tokenizer response field, not a disagreement in charged OpenRouter USD.

**HYPOTHESIS:** Constant `tokens_prompt=64` is a gateway-normalized tokenizer count. Official docs do **not** say this. Code must **not** label it `GATEWAY_NORMALIZED`.

**INSUFFICIENT EVIDENCE:** that OpenRouter’s two upstream-cost fields equal independent OpenAI billing (layer G was not observed).

## 4. Future comparability contract (v2)

Distinguish **field presence** from **semantic comparability**. Do not compare two integers because both are named “tokens.”

Frozen **v1** (Mission 6B-OR, default when `reconciliation_semantics` is omitted):

- required: response `prompt_tokens` vs generation `tokens_prompt`; response `completion_tokens` vs generation `tokens_completion`
- native vs normalized divergence is recorded, not forced equal
- historical replay of 6B-OR remains `R1=MISMATCH` / overall `VALIDATED_MISMATCH`

Future **v2** (`reconciliation_semantics: "v2"`):

| Left | Right | Comparable? | Why |
| --- | --- | --- | --- |
| `response.usage.prompt_tokens` | `generation.native_tokens_prompt` | yes, when provider identity is known and both values are present | native tokenizer billed surface vs official provider-reported native prompt tokens |
| `response.usage.completion_tokens` | `generation.native_tokens_completion` | yes, same provenance conditions | official native completion tokens as reported by provider |
| `response.usage.prompt_tokens_details.cached_tokens` | `generation.native_tokens_cached` | yes, when both present and provider known | official native cached tokens as reported by provider |
| `response.usage.prompt_tokens` | `generation.tokens_prompt` | **no** → `NOT_COMPARABLE` | `tokens_prompt` remains `UNKNOWN` |
| `response.usage.completion_tokens` | `generation.tokens_completion` | **no** → `NOT_COMPARABLE` | `tokens_completion` remains `UNKNOWN` |

v2 must not fall back to comparing `tokens_prompt` when native fields are missing. Missing native fields are `INCONCLUSIVE` or `NOT_COMPARABLE`, not billed `MISMATCH`.

Native-field agreement under v2 does **not** rewrite stored 6B-OR artifacts.

No provider-specific hack is introduced. The comparable pairs above follow OpenRouter’s own `native_*` / “as reported by provider” provenance plus official completions native-tokenizer billing prose.

## 5. Streamed metadata anomaly

**FACT:** 6B-OR requested Chat Completions `stream=false` and received `object=chat.completion` on all 50. Generation metadata `streamed=true` on all 50.

**FACT:** Current official schema describes `streamed` as “Whether the response was streamed.”

**INSUFFICIENT EVIDENCE:** whether that field means client-visible delivery, upstream/internal transport, or something else. Official docs do not explain internal/upstream streaming for this field.

Classification: `PROVIDER_METADATA_CONTRACT_ANOMALY`.

Future evidence logic must not treat `generation.streamed` as proof of client-visible streaming. `generation_streamed_proves_client_visible_streaming(*)` is always `false`.

This was not a frozen 6B halt condition and is not turned into one now.

## 6. Upstream cost surfaces

Two OpenRouter-reported fields exist. They disagreed in 6B-OR:

| Surface | 6B-OR aggregate | Official description |
| --- | --- | --- |
| generation `upstream_inference_cost` | **0** on all 50; sum **0** | OpenAPI: “Cost charged by the upstream provider.” Usage-accounting: when obtained via generation ID, this field is only available for BYOK; otherwise 0 or null. 6B-OR had `is_byok=false`. |
| response `usage.cost_details.upstream_inference_cost` | sum **0.00070845** (equals C/D/E) | OpenAPI: **no field description**. Usage-accounting: “The actual cost charged by the upstream AI provider.” |

**FACT:** The surfaces are not interchangeable in this evidence.

**INSUFFICIENT EVIDENCE:** which, if either, equals independent OpenAI billed cost.

Required future rule: OpenRouter-reported upstream cost remains informational. It **must not** participate in a required financial validation gate merely because the field name sounds authoritative. Neither surface is treated as OpenAI truth.

## 7. Versioning

| Version | Where | Behavior |
| --- | --- | --- |
| `v1` | default; frozen 6B-OR spec omits the field | historical `reconcile_tokens_r1` |
| `v2` | explicit `reconciliation_semantics: "v2"` | `reconcile_tokens_r1_v2` |

Do not overwrite v1. Do not regenerate 6B-OR reports.

## 8. Raw evidence archive

**FACT:** Raw 6B-OR payloads were copied, without modifying source bytes, from `/private/tmp/inferenceledger-openrouter-6b/openrouter-billing-reconciliation-6b-20260911T220449Z/` to the local non-Git path:

`~/.local/share/inferenceledger/evidence/openrouter-billing-reconciliation-6b-20260911T220449Z/`

Directory mode `700`, files mode `600`. SHA-256 of all **102** raw artifacts matched the committed `raw-evidence-manifest.json`. Hash failures: **none**. The `/private/tmp` source was not deleted. Raw payloads are not in Git.

## 9. Supported / unsupported claims

Supported:

- Mission 6B-OR remains `VALIDATED_MISMATCH` under frozen v1 R1.
- B→C→D→E matched exactly for this isolated OpenRouter micro-experiment.
- Future v2 will not treat undocumented `tokens_prompt` as a billed equivalent.

Unsupported:

- direct OpenAI billing validated
- OpenAI Costs API / invoice validated
- OpenRouter upstream cost independently verified against OpenAI
- `tokens_prompt` officially defined as gateway-normalized
- `generation.streamed` as client-visible streaming proof
- production economics, customer value, Change Gate SHIP
