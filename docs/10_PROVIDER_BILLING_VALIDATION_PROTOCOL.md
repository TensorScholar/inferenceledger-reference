# Provider billing validation protocol

**Mission:** 6A — feasibility audit and 6B protocol lock.  
**Access date for first-party documentation:** 2026-09-11.  
**Mission 6A spend:** **$0**. No inference, Usage, Costs, billing, or account-mutation calls were made.  
**Readiness:** `READY_AFTER_USER_ACTION`.  
**Selected first provider:** OpenAI (official `api.openai.com` Chat Completions).  
**This document is not a 6B pre-registration.** Account, project, API-key, and spend-ceiling parameters are unknown.

Language in this note is tagged:

- **FACT** — observed in this repository, this process environment, or current first-party provider docs
- **INFERENCE** — follows from those facts
- **HYPOTHESIS** — testable in Mission 6B, not established
- **INSUFFICIENT EVIDENCE** — not verified; must not be treated as true

## 1. Why this protocol exists

InferenceLedger can currently represent:

- provider/model identity on an execution attempt
- request-level usage/token accounting
- versioned pricing provenance
- reconstructed monetary estimates
- LiteLLM/execution-stack reported monetary values

It cannot yet establish:

- agreement with a provider billing ledger
- invoice equivalence
- financial reconciliation accuracy

Historical local LiteLLM/Ollama zeros, reconstructed FreeModel OpenAI-compatible costs, and Change Gate `INCONCLUSIVE` decisions are **not** OpenAI billing evidence.

## 2. Repository facts (Phase 0 / Phase 1)

Inspected at canonical main `be5179f069c78b4e0d574f56c7ecf975cb61650b` before this documentation branch.

### FACT — existing capability

| Capability | Where | Notes |
| --- | --- | --- |
| OpenAI-compatible Chat Completions transport | `OpenAIBackend` | Uses `openai.AsyncOpenAI`; SDK `max_retries=0`; bounded `RetryPolicy` |
| `OPENAI_API_KEY` consumption | CLI smoke, `scripts/run_benchmark.py`, FastAPI `/inference`, skipped real-provider test | Process `os.getenv`; not the unused `Settings.openai_api_key` field |
| `OPENAI_BASE_URL` resolution | same call sites | `None` → official SDK default; custom URL is **not** OpenAI billing identity |
| Execution vs pricing identity | `OpenAIBackend`, benchmark `_resolve_*_provider` | Direct OpenAI (`base_url is None`) defaults both to `openai`. Custom base URL without explicit names is `openai-compatible-unknown` and unpriced. Custom URL cannot inherit OpenAI prices by guess. |
| Usage capture | `response.usage` | `prompt_tokens`, `completion_tokens`, `total_tokens` required; missing usage fails closed |
| Cached tokens | `_extract_cached_tokens` | `usage.prompt_tokens_details.cached_tokens` when present; otherwise `0` |
| Success/failure attempts | `ProviderAttempt` | Retries are new attempts; failed attempts keep error type/status |
| Reconstructed cost | `CostCalculator` + `PricingTable` | `CostEvidenceKind.CALCULATED_FROM_USAGE` with frozen table version and record id |
| Execution-stack reported cost | LiteLLM importer | `CostEvidenceKind.REPORTED_BY_EXECUTION_STACK`; mutually exclusive with calculated cost on one attempt |
| Estimate vs observed reconciliation | `benchmarking/reconciliation.py` | Route estimate vs execution-chain cost; **not** provider billing |
| Git-proven pre-registration | `experiment_provenance.py` | Mission 4.1 contract; 6B must reuse it |
| Real-provider test gate | `tests/integration/test_real_provider.py` | Skips unless `OPENAI_API_KEY` is present; default model `gpt-5.4-mini` |

### FACT — gaps (not 6A production blockers)

| Gap | Consequence |
| --- | --- |
| Provider response/request `id` is not retained | Cannot join a Chat Completions `id` after the fact. Usage/Costs APIs are aggregates anyway, so run-level reconciliation does not require this field. |
| `service_tier` is not sent or retained | 6B must not use Fast/Priority; default standard tier only. |
| Attempt start/end are `perf_counter` latency plus a completion `timestamp` | Enough to bound a UTC execution interval if the run stays inside one Costs day. Not a provider timestamp. |
| Cache-write tokens are not a first-class field | Material for GPT-5.6+ tariffs; avoided by choosing `gpt-4o-mini`. |
| `CostEvidenceKind` has no billing-ledger or invoice member | Layers E/F belong in 6B experiment artifacts, not on `ProviderAttempt`. |
| No Usage/Costs client | 6B capture may be a one-off audited script; do not speculatively productize ingestion. |
| No Anthropic/Gemini execution adapter | Selecting those providers would add architecture. |

### FACT — historical OpenAI-compatible evidence is not billing evidence

`benchmarks/reports/smoke-json-contract-gpt-5-4-mini-20260703.md` records an OpenAI-compatible **FreeModel** endpoint. A custom `OPENAI_BASE_URL` must never be treated as proof of genuine OpenAI provider billing.

### NOT NEEDED before a valid 6B run

- New provider adapter
- New `CostEvidenceKind` members
- Productized Usage/Costs ingestion
- Changing Mission 4/5 artifacts
- Promoting reconstructed FreeModel cost to invoice truth

## 3. Environment capability (Phase 2)

Inspected **process environment only** on 2026-09-11. Untracked `.env` files, shell history, Keychain, and credential files were **not** read. Presence does not prove validity, entitlement, project identity, or funding.

| Variable | Classification |
| --- | --- |
| `OPENAI_API_KEY` | **FACT:** `ABSENT` in this process |
| `OPENAI_ADMIN_KEY` | **FACT:** `ABSENT` in this process |
| `OPENAI_BASE_URL` | **FACT:** `UNSET` |
| `ANTHROPIC_API_KEY` | **FACT:** `ABSENT` in this process |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | **FACT:** `ABSENT` in this process |

`OPENAI_BASE_URL` allowed classes (do not echo values): `UNSET`, `OFFICIAL_OPENAI_HOST`, `CUSTOM_OR_NON_OPENAI_HOST`, `UNPARSEABLE`.

**INSUFFICIENT EVIDENCE:** key validity, billing entitlement, organization/project identity, prepaid/postpaid plan, credit grants, whether keys exist in other user sessions.

Contract used by this repository:

- execution: `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`)
- organization usage/cost: `OPENAI_ADMIN_KEY` (documented by OpenAI; not consumed by current code)
- `Settings.openai_api_key` / `Settings.anthropic_api_key` are unused by the live OpenAI path

## 4. Provider feasibility matrix (Phase 3)

Evaluated from first-party docs on 2026-09-11. No provider was chosen because it was cheapest.

| Criterion | OpenAI | Anthropic (Claude Platform) | Gemini Developer API |
| --- | --- | --- | --- |
| Existing execution path | **FACT:** yes, Chat Completions | **FACT:** no adapter; `anthropic_api_key` unused | **FACT:** no adapter |
| New adapter work | none for execution | new Messages adapter | new Gemini adapter plus Cloud identity |
| Request-level usage | Chat Completions `usage` | Messages `usage` (not wired) | generateContent usage (not wired) |
| Provider usage API | `GET /v1/organization/usage/completions` | `GET /v1/organizations/usage_report/messages` | AI Studio usage dashboard; no equivalent first-party completions usage API in the Gemini billing guide |
| Provider monetary API | `GET /v1/organization/costs` | `GET /v1/organizations/cost_report` | Cloud Billing / AI Studio; Prepay latency ~10 minutes |
| Usage granularity | `1m` / `1h` / `1d` | `1m` / `1h` / `1d` | dashboard/billing, not request rows |
| Cost granularity | **FACT:** `1d` only | **FACT:** `1d` only | Cloud Billing daily/export; not request-level |
| Project/key isolation | Projects + project API keys; Usage/Costs filter `project_ids`; Usage also `api_key_ids` / `models` | Workspaces + API keys; Admin API unavailable for individual accounts | Cloud project + API key; billing account may span projects |
| Execution credential | project API key | workspace/org API key | API key |
| Admin/billing credential | Admin API key; **cannot** call non-admin endpoints | Admin API key `sk-ant-admin…` | Cloud Billing IAM, not a Gemini inference key |
| Minimum funding | **FACT:** `gpt-4o-mini` Free tier “Not supported”; paid account required. **INSUFFICIENT EVIDENCE** of this org’s credits/tier | Organization required for Admin API; **INSUFFICIENT EVIDENCE** of minimum prepaid | **FACT:** paid setup documents a **$5** Prepay minimum (or Postpay assignment) |
| Expected 6B spend | micro-pilot; see §12 | similar, but adapter cost dominates | $5 floor can exceed a micro-pilot |
| Docs quality | Usage + Costs API reference + cookbook + Help Center | Usage & Cost Admin API is explicit | Billing guide is strong on plans, weak on request-level ledger APIs |
| Reconstruct billing evidence | Costs amount `{currency,value}` + line items | cost report amounts in USD cents-as-decimal-strings | Cloud Billing export, not a first-class inference ledger |
| Ledger vs invoice | Help Center: Cost export/API is the path that reconciles to invoiced API consumption; invoices themselves may lack line detail (Enterprise note from 2026-04-01) | Cost API described for billing match; still not the PDF invoice | Cloud invoice is a different system |
| Contamination risk | org-level traffic unless dedicated project + quiet window | workspace contamination; playground usage has null API key | billing-account-wide Gemini usage; ~10 min pipeline lag |

**INFERENCE:** OpenAI is the strongest first provider because it matches the existing transport, already has OpenAI pricing rows, and exposes both a usage aggregate and a monetary Costs API without a new adapter.

Anthropic’s Admin usage/cost APIs are capable, but they require a new execution adapter and an organization (Admin API is unavailable for individual accounts). Gemini’s paid path adds Cloud Billing, a $5 Prepay floor for many new paid accounts, and no Chat-Completions-shaped usage/cost pair. Neither is a reason to avoid OpenAI.

## 5. Official OpenAI documentation (verified 2026-09-11)

Primary first-party sources:

| Topic | URL |
| --- | --- |
| Completions Usage API | https://developers.openai.com/api/reference/resources/admin/subresources/organization/subresources/usage/methods/completions/ |
| Costs API | https://developers.openai.com/api/reference/resources/admin/subresources/organization/subresources/usage/methods/costs/ |
| Admin APIs / Admin key | https://developers.openai.com/api/docs/guides/admin-apis |
| Usage + Costs cookbook (OpenAI) | https://developers.openai.com/cookbook/examples/completions_usage_api |
| Reviewing usage and costs | https://help.openai.com/en/articles/10478918-reviewing-api-usage-and-costs |
| Cost export / invoice reconciliation | https://help.openai.com/en/articles/20001072-how-do-i-export-monthly-usage-details-from-the-api-usage-dashboard |
| Production org/project/keys | https://developers.openai.com/api/docs/guides/production-best-practices |
| RBAC / project boundaries | https://developers.openai.com/api/docs/guides/rbac |
| Pricing | https://developers.openai.com/api/docs/pricing |
| `gpt-4o-mini` | https://developers.openai.com/api/docs/models/gpt-4o-mini |
| `gpt-5.4-mini` | https://developers.openai.com/api/docs/models/gpt-5.4-mini |
| `gpt-5.6-luna` | https://developers.openai.com/api/docs/models/gpt-5.6-luna |

Anthropic / Gemini sources used only for the matrix:

- https://platform.claude.com/docs/en/manage-claude/usage-cost-api
- https://ai.google.dev/gemini-api/docs/billing

Do not treat blog posts or third-party dashboards as billing truth when these pages exist.

### FACT — Usage API (`GET /v1/organization/usage/completions`)

- Requires an **Admin API key** (`Authorization: Bearer $OPENAI_ADMIN_KEY`). Admin keys cannot be used for non-administration endpoints.
- Time filter: `start_time` required (Unix seconds, inclusive); `end_time` optional.
- `bucket_width`: `1m`, `1h`, or `1d` (default `1d`). Limits: `1m` default 60 max 1440; `1h` default 24 max 168; `1d` default 7 max 31.
- Filter: `project_ids`, `user_ids`, `api_key_ids`, `models`, `batch`.
- `group_by` (cookbook + result schema): including `model`, `project_id`, `api_key_id`, and related dimensions. Without `group_by`, those fields return `null`.
- Results include `input_tokens`, `output_tokens`, `num_model_requests`, and `input_cached_tokens`. Official text: input tokens **include cached and cache-write tokens**.
- This is **not** a request-level ledger.

### FACT — Costs API (`GET /v1/organization/costs`)

- Requires an **Admin API key**.
- Represents financial cost (`amount.value` + `amount.currency`).
- `bucket_width` currently **only `1d`**.
- Filter: `project_ids`, `api_key_ids`.
- `group_by`: `project_id`, `line_item`, `api_key_id` (and combinations). **Not** `model`.
- Cookbook example values are high-precision floats (IEEE-looking), not a documented rounding quantum.

### FACT — dashboard / invoice relationship

- Usage dashboard is UTC. 1-minute intervals apply to **usage**, not billing data.
- Help Center: for a breakdown that **reconciles to invoiced API consumption**, use **Cost data** grouped by **line item**, calendar-month UTC, 1-day interval.
- Enterprise invoices from 2026-04-01 may omit detailed API costs; Cost export replaces that detail.
- Credit grants and API usage answer different questions.
- Scale Tier bundle costs are organization-level, not project-level. Scale Tier can show usage with **zero project spend**.
- Playground traffic is billable API usage.

**INFERENCE:** Costs API / Cost dashboard/export is layer **E** (provider billing ledger). A PDF/statement is layer **F**. 6B must not claim **F** if only **E** is captured.

**INSUFFICIENT EVIDENCE:** a guaranteed Usage/Costs processing SLA. 6B must record acquisition time and allow an explicit delay window. Absent data after that window is `INCONCLUSIVE`, not `MATCH`.

## 6. Evidence truth hierarchy (Phase 4)

Do not collapse D, E, and F.

| Layer | Name | Meaning | Current mapping |
| --- | --- | --- | --- |
| A | `PROVIDER_RESPONSE_USAGE` | Tokens/usage returned with the inference response | `ProviderAttempt` token fields / `UsageMetrics` |
| B | `INFERENCELEDGER_RECONSTRUCTED_COST` | Tokens × frozen `PricingTable` | `CALCULATED_FROM_USAGE` + pricing provenance |
| C | `EXECUTION_STACK_REPORTED_COST` | Amount an execution stack reported (e.g. LiteLLM `response_cost`) | `REPORTED_BY_EXECUTION_STACK` |
| D | `PROVIDER_USAGE_AGGREGATE` | Provider Usage API buckets | **no domain type**; 6B artifact |
| E | `PROVIDER_BILLING_LEDGER` | Provider Costs API / Cost export intended to reconcile with billing | **no domain type**; 6B artifact |
| F | `FINAL_INVOICE` | Independently obtained invoice/statement | **no domain type**; out of 6B unless the user supplies one |

Smallest concrete gap: D/E/F are experiment evidence packs (sanitized JSON + SHA-256 of raw bytes stored outside Git). That is not a 6A production-code change.

6B primary path is **direct `OpenAIBackend`**, not LiteLLM, so **C** is expected absent and comparison 3 is `NOT_COMPARABLE`. Do not add LiteLLM merely to populate C.

## 7. Billing attribution (Phase 5)

**FACT:** a request-level telemetry row cannot be joined one-to-one to a daily Costs row.

**FACT:** Costs cannot group by model.

**INFERENCE:** exact billing attribution is only defensible with isolation:

1. dedicated OpenAI **project** for this experiment only
2. dedicated **project API key** used by 6B and nothing else
3. **no unrelated traffic** in that project during the UTC billing day that contains the run (no Playground, no CI, no other apps)
4. **one model**
5. recorded execution interval from telemetry timestamps
6. optional project spend limit slightly above the user-approved ceiling

If a dedicated project is unavailable, classify billing attribution as **`BLOCKED` / `AMBIGUOUS`**. Do not weaken this to finish 6B.

Record project identity in committed evidence as a **SHA-256 of the project id** (or another irreversible hash), not the raw `proj_…` string unless a later spec proves the identifier is non-sensitive and necessary.

## 8. Security contract

| Secret | Role | Storage |
| --- | --- | --- |
| `OPENAI_API_KEY` | Chat Completions execution | environment / secret store only |
| `OPENAI_ADMIN_KEY` | Usage + Costs + org admin | environment / secret store only; never used for inference |

Never store either in Git, report artifacts, logs, fixtures, or screenshots. Do not log `Authorization` headers.

Raw Usage/Costs payloads may contain `organization_id`, `project_id`, `api_key_id`, `user_id`. Keep raw bytes **outside VCS**. Commit only sanitized aggregates plus `sha256` of the raw file and the acquisition timestamp.

6A/6B must not create projects, keys, or spend. Those are **user** actions.

## 9. Mission 6B experiment (design only — do not execute)

### Identity (to be frozen in a future Git-proven spec)

| Field | Proposed value | Freeze rule |
| --- | --- | --- |
| Experiment id | `openai-billing-ledger-6b` | exact id committed before any call |
| Classification | `PROVIDER_BILLING_LEDGER_VALIDATION` | not Change Gate, not customer workload, not `FINAL_INVOICE` |
| Provider | `openai` | `OPENAI_BASE_URL` must be `UNSET` or `OFFICIAL_OPENAI_HOST` |
| Model | `gpt-4o-mini` | optional snapshot `gpt-4o-mini-2024-07-18` |
| Endpoint | Chat Completions via existing `OpenAIBackend` | no streaming; `RetryPolicy(max_attempts=1)` unless a later spec says otherwise |
| Isolation | dedicated project + dedicated key + quiet UTC day | if missing → do not run |

Do **not** create the spec file until the user supplies project hash, spend ceiling, and confirmation that keys exist.

### Pre-registration (reuse Mission 4.1)

Before the first 6B network call:

- experiment spec committed, clean, and `GIT_PROVEN`
- pricing table version + model record frozen
- reconciliation policy and classifications frozen
- spend ceiling frozen (user-chosen)
- isolation commitments frozen
- `scripts/run_litellm_workload.py`-style dirty-spec fail-closed applied to whatever 6B runner is used

### Execution evidence per request

Preserve:

- `workload_item_id`
- InferenceLedger request/run id
- provider response id **if captured** (recommended in the 6B harness; not required for run-level join)
- model, provider
- start/end timestamps (harness wall clock in UTC)
- input, cached input, output, total tokens
- service tier (`standard` / omitted; never Fast/Priority)
- status, attempt count, retry count
- execution-stack reported cost (likely absent)
- reconstructed cost + pricing provenance

### Provider-side usage evidence (after execution)

Capture Usage API with:

- query interval covering the execution interval plus a documented delay margin, aligned to bucket edges
- `project_ids` = the dedicated project
- `group_by` including `model`, `project_id`, `api_key_id`
- `bucket_width=1m` for the run window (and `1d` for the billing day)
- request count, input/output/cached totals
- source response SHA-256, acquisition timestamp
- processing-delay caveat

### Provider-side cost evidence (after execution)

Capture Costs API with:

- the UTC day(s) overlapping the run
- `project_ids` filter
- `group_by` including `project_id` and `line_item` (and `api_key_id` if used)
- currency, amount, granularity `1d`
- source response SHA-256, acquisition timestamp

If Costs is empty the same day, wait and retry; still empty → `INCONCLUSIVE`, do not invent rows.

## 10. Reconciliation targets and classifications

Compare at least:

1. Sum of A (request-level provider-response usage) vs D (Usage API aggregate) for the isolated project/key/model/window
2. Sum of B (reconstructed cost) vs E (Costs API project/day, relevant line items)
3. Sum of C vs E — `NOT_COMPARABLE` if C is absent
4. Attempt/retry amplification **only if genuine retries occur**. Do not force retries. Absent retries is not a failure.

Outcomes: `MATCH`, `MISMATCH`, `INCONCLUSIVE`, `NOT_COMPARABLE`. Do not use `PASS` for structurally non-comparable evidence.

| Situation | Classification |
| --- | --- |
| Comparable integers/amounts agree under the frozen policy | `MATCH` |
| Comparable tokens or ledger amounts disagree materially | `MISMATCH` |
| Delay, empty Costs, credits/Scale Tier ambiguity, extra project traffic | `INCONCLUSIVE` |
| Different time bounds, missing C, Costs vs invoice, custom base URL | `NOT_COMPARABLE` or `INCONCLUSIVE` as specified in the 6B spec |

## 11. Tolerance methodology (freeze before 6B, not after)

**FACT:** OpenAI docs do not publish a reconstructed-cost vs Costs-API epsilon.

**FACT:** Usage token fields are integers. Costs `amount.value` in the cookbook is a JSON number with many decimals.

**Required pre-6B freeze** (accept or replace in the spec; do not tune after seeing residuals):

1. **Tokens (A vs D):** exact integer equality on `num_model_requests`, `output_tokens`, and `input_tokens` after summing comparable buckets. If `cached_tokens` are all zero, `sum(prompt_tokens)` must equal `input_tokens`. If cached tokens are non-zero, also require `sum(cached_tokens) == input_cached_tokens`. Any leftover must be explained as documented cache-write inclusion or else `MISMATCH` / `INCONCLUSIVE` — do not silently drop it.
2. **Money (B vs E):** compare `Decimal` sums. Primary rule is exact equality of reconstructed `Decimal` vs `Decimal(str(amount.value))` summed over isolated project/day line items in USD. If JSON numbers prevent exact decimal reconstruction, that is `INCONCLUSIVE` until the spec records the payload’s actual decimal text.
3. **Optional quantum, only if the spec explicitly adopts it before execution:** one input token at the frozen model tariff (`gpt-4o-mini` input = `$0.15 / 1e6` = `$1.5e-7`). Derived from the smallest billed unit, not from 6B residuals. If adopted, `|B − E| ≤ 1 input-token quantum` may be `MATCH`; larger gaps are `MISMATCH` unless an independently documented discount/credit/Scale Tier explanation exists (then `INCONCLUSIVE`, not a silent pass).
4. **Credits, discounts, Scale Tier, Fast/Priority, data-residency uplift, cache writes:** any of these present and unquantified → monetary comparison `INCONCLUSIVE`.
5. Never choose a tolerance after observing 6B results.

## 12. Model and pricing audit

Repository table: `PRICING_TABLE_VERSION = openai-standard-2026-09-03`, `PRICING_OBSERVED_AT = 2026-09-03`.

| Model | Repo rates (in / cached / out per 1M) | Official 2026-09-11 | Verdict |
| --- | --- | --- | --- |
| `gpt-4o-mini` | 0.15 / 0.075 / 0.60 | **FACT:** 0.15 / 0.075 / 0.60; Chat Completions listed; Free tier not supported; snapshot `gpt-4o-mini-2024-07-18` | **current** for this model |
| `gpt-5.4-mini` | 0.75 / 0.075 / 4.50 | **FACT:** same rates; still documented | **current** but ~5× input and 7.5× output vs `gpt-4o-mini`; reasoning tokens add reconstruction risk |
| `gpt-5.6-luna` | **absent** | **FACT:** 0.20 / 0.02 / 1.20 plus cache-write 1.25× and long-context multipliers | **not cheaper** than `gpt-4o-mini` for short text; would require a pricing-table update before use |
| `gpt-5.4-nano` | absent | listed as $0.20 input / $1.25 output on the `gpt-5.4-mini` comparison table | **not cheaper** than `gpt-4o-mini` |
| Featured 5.6/6 family | absent except older 5.4/5.5/5.3-codex rows | pricing page leads with `gpt-6-astra` / `gpt-5.6-*` | table is **partially stale as a catalog**, not as the `gpt-4o-mini` row |

**INFERENCE:** 6B should use **`gpt-4o-mini`**. It is still supported, already priced in-repo at current official rates, cheaper than Luna for this micro-pilot, and avoids cache-write / long-context / reasoning tariffs.

Do not silently edit `pricing.py` from memory. A Luna experiment would need a new versioned pricing record **before** execution.

### Estimated 6B spend (no charge in 6A)

Illustrative arithmetic at official `gpt-4o-mini` rates, **not** a commitment:

- 100 requests × (900 input + 20 output) tokens ≈ `$0.0147`
- Distinguishing from a true zero is plausible because Costs examples report sub-cent floats
- A slightly larger prompt or ~150–400 requests lands near **`$0.02–$0.15`**

**Recommended estimated required spend:** `$0.02–$0.15`.  
**Hard spend ceiling:** **not chosen by 6A**. User must set it before 6B.  
**Required hard user approval before any 6B spend:** **YES**.

## 13. Production code decision

**No new production code in 6A.** Existing types can carry A/B/C. D/E/F are experiment artifacts. Dropped provider request ids are a 6B harness improvement, not a blocker for isolated run-level reconciliation.

## 14. Claims 6B could establish (if `MATCH` under isolation)

- A and D agree for one isolated OpenAI project/key/model window
- B and E agree for that project’s Costs day/line items under the frozen monetary policy
- InferenceLedger reconstructed OpenAI pricing is not contradicted by the provider billing ledger for that micro-pilot
- Custom `OPENAI_BASE_URL` / FreeModel history remains out of scope

## 15. Claims 6B still could NOT establish

- `FINAL_INVOICE` / PDF statement equivalence (F)
- credit-grant net cash vs ledger gross
- Scale Tier, Fast/Priority, batch, tools, or other models
- LiteLLM `response_cost` invoice truth (C absent on the primary path)
- customer-workload, commercial, or Change Gate SHIP
- production readiness
- callback completeness of any gateway
- general statistical validity of routing savings

## 16. Exact next user action

Before Mission 6B may start:

1. Confirm a **paid** OpenAI organization (Free tier does not support `gpt-4o-mini`).
2. Create a **dedicated experiment project** and a **dedicated project API key**. Do not commit the key.
3. Create an **Admin API key** for Usage/Costs. Do not use it for inference. Do not commit it.
4. Confirm **no other traffic** will use that project on the experiment UTC day.
5. Set a **hard spend ceiling** (user-chosen) and any project spend limit.
6. Export `OPENAI_API_KEY` and `OPENAI_ADMIN_KEY` in the execution environment; leave `OPENAI_BASE_URL` unset.
7. Reply with: project-id hash (not the raw id unless you accept committing it), spend ceiling, and approval to spend up to that ceiling.
8. Only then may a Git-proven 6B spec be committed.

Until those exist, 6B is not authorized. Mission 6A stops here.
