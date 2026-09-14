"""Paid-run harness for Mission 6B-OR. Unit tests inject a fake transport; no live calls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .experiment_provenance import (
    ExecutionInterval,
    build_execution_provenance,
    require_committed_clean_experiment_spec,
)
from .harness import load_workload
from .openrouter_billing import (
    CLASSIFICATION,
    EXECUTION_GATEWAY,
    MAX_OUTPUT_TOKENS,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    OPENROUTER_KEY_URL,
    REQUESTED_MODEL,
    SOFTWARE_GUARD_USD,
    Clock,
    EvidenceClass,
    ExperimentClassification,
    FrozenTariff,
    HttpResponse,
    HttpTransport,
    KeyReadinessStatus,
    OpenRouterBillingError,
    OpenRouterTransportError,
    ResponseUsage,
    RuleResult,
    SpendPolicy,
    SystemClock,
    allowed_service_tier,
    allowed_upstream_provider,
    assert_raw_dir_outside_git,
    assess_key_readiness,
    budget_allows_next_request,
    chat_completion_body,
    chat_completion_headers,
    classify_experiment,
    combine_required_statuses,
    conservative_request_cost_usd,
    ensure_private_directory,
    env_key_presence,
    extract_response_usage,
    format_decimal,
    generation_lookup_url,
    key_usage_delta,
    load_frozen_tariff,
    load_json_preserving_decimals,
    load_spend_policy,
    observed_cost_for_budget,
    optional_decimal,
    parse_object_payload,
    prompt_utf8_len,
    reconcile_reconstructed_vs_response_r2,
    reconcile_response_vs_generation_cost_r3,
    reconcile_sum_vs_key_delta,
    reconstruct_openrouter_cost,
    require_conservative_budget,
    require_experiment_id,
    require_git_execution_boundary,
    require_openrouter_gateway,
    require_openrouter_model,
    require_present_openrouter_key,
    sanitize_chat_completion_payload,
    sanitize_generation_payload,
    sanitize_key_payload,
    utc_now_iso,
    write_private_bytes,
)
from .openrouter_evidence_semantics import (
    StreamedFieldClassification,
    StreamedFieldInterpretation,
    interpret_generation_streamed_field,
    resolve_reconciliation_semantics,
    select_r1_reconciler,
)

_DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class PollPolicy:
    generation_max_attempts: int
    generation_initial_delay_seconds: float
    generation_backoff_multiplier: float
    generation_max_delay_seconds: float
    key_max_attempts: int
    key_delay_seconds: float
    key_stable_reads: int


@dataclass(frozen=True)
class AttemptRecord:
    workload_item_id: str
    local_request_id: str
    started_at_utc: str
    ended_at_utc: str
    latency_ms: int
    http_status: int | None
    error_class: str | None
    retry_count: int
    generation_id: str | None
    requested_model: str
    response_model: str | None
    execution_gateway: str
    reported_upstream_provider: str | None
    upstream_id: str | None
    service_tier: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None
    response_cost: Decimal | None
    reconstructed_cost: Decimal | None
    generation_total_cost: Decimal | None
    upstream_inference_cost: Decimal | None
    response_upstream_inference_cost: Decimal | None
    generation_upstream_inference_cost: Decimal | None
    cache_discount: Decimal | None
    is_byok: bool | None
    attempt_status: str
    charged: bool
    stop_reason: str | None
    raw_response_sha256: str | None
    raw_generation_sha256: str | None


def load_poll_policy(payload: Mapping[str, Any]) -> PollPolicy:
    generation = payload["generation_metadata_poll"]
    key = payload["key_usage_settlement_poll"]
    if not isinstance(generation, Mapping) or not isinstance(key, Mapping):
        raise OpenRouterBillingError("poll policy blocks are required")
    return PollPolicy(
        generation_max_attempts=int(generation["max_attempts"]),
        generation_initial_delay_seconds=float(generation["initial_delay_seconds"]),
        generation_backoff_multiplier=float(generation["backoff_multiplier"]),
        generation_max_delay_seconds=float(generation["max_delay_seconds"]),
        key_max_attempts=int(key["max_attempts"]),
        key_delay_seconds=float(key["delay_seconds"]),
        key_stable_reads=int(key["stable_reads"]),
    )


def validate_experiment_spec(spec: Mapping[str, Any]) -> None:
    require_experiment_id(str(spec["experiment_id"]))
    if spec.get("classification") != CLASSIFICATION:
        raise OpenRouterBillingError("classification must be GATEWAY_BILLING_VALIDATION")
    runtime = spec["runtime"]
    if not isinstance(runtime, Mapping):
        raise OpenRouterBillingError("runtime block is required")
    require_openrouter_gateway(str(runtime["execution_gateway"]))
    require_openrouter_model(str(runtime["requested_model"]))
    if runtime.get("application_retries") != 0 or runtime.get("sdk_retries") != 0:
        raise OpenRouterBillingError("retries must be frozen at 0")
    if runtime.get("allow_fallbacks") is not False:
        raise OpenRouterBillingError("allow_fallbacks must be false")
    if list(runtime.get("provider_only") or []) != ["openai"]:
        raise OpenRouterBillingError("provider_only must be exactly ['openai']")
    if list(runtime.get("provider_order") or []) != ["openai"]:
        raise OpenRouterBillingError("provider_order must be exactly ['openai']")
    if runtime.get("stream") is not False:
        raise OpenRouterBillingError("streaming is forbidden")
    resolve_reconciliation_semantics(spec)
    if runtime.get("response_cache_opt_in") is not False:
        raise OpenRouterBillingError("OpenRouter response-cache opt-in is forbidden")
    if runtime.get("service_tier") not in {None, "default"}:
        raise OpenRouterBillingError("service_tier must be omitted or default")
    if int(runtime["max_tokens"]) != MAX_OUTPUT_TOKENS:
        raise OpenRouterBillingError("max_tokens must be 8")
    if float(runtime["temperature"]) != 0:
        raise OpenRouterBillingError("temperature must be 0")


def get_key_payload(
    transport: HttpTransport,
    *,
    api_key: str,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> tuple[bytes, dict[str, Any]]:
    response = transport.send(
        "GET",
        OPENROUTER_KEY_URL,
        headers=chat_completion_headers(api_key),
        body=None,
        timeout_seconds=timeout_seconds,
    )
    if response.status_code != 200:
        raise OpenRouterBillingError(f"key metadata HTTP {response.status_code}")
    return response.body, parse_object_payload(response.body)


def issue_chat_completion(
    transport: HttpTransport,
    *,
    api_key: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> HttpResponse:
    body = json.dumps(
        chat_completion_body(
            model=REQUESTED_MODEL,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            provider_only=["openai"],
            provider_order=["openai"],
            allow_fallbacks=False,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return transport.send(
        "POST",
        OPENROUTER_CHAT_COMPLETIONS_URL,
        headers=chat_completion_headers(api_key),
        body=body,
        timeout_seconds=timeout_seconds,
    )


def poll_generation_metadata(
    transport: HttpTransport,
    *,
    api_key: str,
    generation_id: str,
    policy: PollPolicy,
    clock: Clock,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> tuple[bytes, dict[str, Any]] | None:
    delay = policy.generation_initial_delay_seconds
    url = generation_lookup_url(generation_id)
    for attempt in range(1, policy.generation_max_attempts + 1):
        response = transport.send(
            "GET",
            url,
            headers=chat_completion_headers(api_key),
            body=None,
            timeout_seconds=timeout_seconds,
        )
        if response.status_code == 200:
            return response.body, parse_object_payload(response.body)
        if attempt >= policy.generation_max_attempts:
            return None
        clock.sleep(delay)
        delay = min(delay * policy.generation_backoff_multiplier, policy.generation_max_delay_seconds)
    return None


def poll_key_until_settled(
    transport: HttpTransport,
    *,
    api_key: str,
    policy: PollPolicy,
    clock: Clock,
    target_delta_candidates: Sequence[Decimal],
    pre_usage: Decimal,
) -> tuple[bytes, dict[str, Any], Decimal, bool]:
    previous: Decimal | None = None
    stable = 0
    last_body = b""
    last_payload: dict[str, Any] = {}
    last_delta = Decimal("0")
    for attempt in range(1, policy.key_max_attempts + 1):
        last_body, last_payload = get_key_payload(transport, api_key=api_key)
        usage = optional_decimal(_key_data(last_payload).get("usage"), field="usage")
        if usage is None:
            raise OpenRouterBillingError("post-run key usage missing")
        last_delta = key_usage_delta(pre_usage=pre_usage, post_usage=usage)
        if last_delta in target_delta_candidates:
            return last_body, last_payload, last_delta, True
        if previous is not None and usage == previous:
            stable += 1
            if stable >= policy.key_stable_reads:
                return last_body, last_payload, last_delta, True
        else:
            stable = 1
        previous = usage
        if attempt >= policy.key_max_attempts:
            break
        clock.sleep(policy.key_delay_seconds)
    return last_body, last_payload, last_delta, False


def run_openrouter_billing_experiment(
    *,
    spec_path: Path,
    output_dir: Path,
    raw_dir: Path,
    repo_root: Path,
    transport: HttpTransport,
    env: Mapping[str, str],
    clock: Clock | None = None,
    execute_paid: bool,
    key_preflight_only: bool = False,
) -> dict[str, Any]:
    clock = clock or SystemClock()
    spec = _load_json_object(spec_path)
    validate_experiment_spec(spec)
    spend = load_spend_policy(spec["spend_policy"])
    tariff = load_frozen_tariff(_load_json_object(Path(str(spec["pricing"]["path"]))))
    require_conservative_budget(tariff, spend)
    poll_policy = load_poll_policy(spec["poll_policy"])
    semantics = resolve_reconciliation_semantics(spec)
    runtime = spec["runtime"]
    max_tokens = int(runtime["max_tokens"])
    if env_key_presence(env) != "PRESENT":
        return _blocked_result(
            spec=spec,
            spend=spend,
            reason="OPENROUTER_API_KEY is ABSENT",
            user_action="Export OPENROUTER_API_KEY for the dedicated OpenRouter key in this process.",
        )
    if not execute_paid and not key_preflight_only:
        snapshot = require_committed_clean_experiment_spec(spec_path)
        return {
            "classification": "NOT_EXECUTED",
            "reason": "paid execution was not requested",
            "spec_sha256": snapshot.spec_sha256,
            "source_commit_sha": snapshot.source_commit_sha,
            "openrouter_api_key": env_key_presence(env),
        }

    source_commit = require_git_execution_boundary(spec_path)
    snapshot = require_committed_clean_experiment_spec(spec_path)
    workload_path = Path(str(spec["workload"]["path"]))
    expected_sha = str(spec["workload"]["sha256"])
    actual_sha = sha256(workload_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise OpenRouterBillingError("workload SHA-256 does not match the pre-registered experiment")
    assert_raw_dir_outside_git(raw_dir, repo_root)
    ensure_private_directory(raw_dir)
    ensure_private_directory(raw_dir / "responses")
    ensure_private_directory(raw_dir / "generations")
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = require_present_openrouter_key(env)
    try:
        pre_raw, pre_payload = get_key_payload(transport, api_key=api_key)
    except (OpenRouterBillingError, OpenRouterTransportError) as exc:
        return _blocked_result(
            spec=spec,
            spend=spend,
            reason=str(exc),
            user_action="Confirm the dedicated OpenRouter key is valid and readable via GET /api/v1/key.",
        )
    readiness = assess_key_readiness(pre_payload, spend=spend)
    pre_sha = write_private_bytes(raw_dir / "pre-key.raw.json", pre_raw)
    pre_acquired = utc_now_iso(clock)
    if not readiness.ready:
        return _blocked_result(
            spec=spec,
            spend=spend,
            reason="provider key is not ready for this experiment",
            user_action=readiness.user_action,
            key_limit=readiness.limit,
            key_limit_remaining=readiness.limit_remaining,
            key_usage=readiness.usage,
            pre_key_sha256=pre_sha,
        )
    if key_preflight_only:
        return {
            "classification": "KEY_PREFLIGHT_READY",
            "status": KeyReadinessStatus.READY.value,
            "experiment_id": spec["experiment_id"],
            "provider_key_limit_usd": format_decimal(readiness.limit) if readiness.limit is not None else None,
            "provider_key_limit_remaining_usd": (
                format_decimal(readiness.limit_remaining) if readiness.limit_remaining is not None else None
            ),
            "provider_key_usage_usd": format_decimal(readiness.usage) if readiness.usage is not None else None,
            "pre_key_sha256": pre_sha,
            "pre_key_acquired_at_utc": pre_acquired,
            "completed_request_count": 0,
            "direct_openai_billing_validated": False,
        }

    workload = load_workload(workload_path)
    if len(workload) != spend.planned_request_count:
        raise OpenRouterBillingError("workload item count does not match the frozen spend policy")
    for item in workload:
        if prompt_utf8_len(item.prompt) > spend.prompt_byte_bound:
            raise OpenRouterBillingError("workload prompt exceeds the frozen byte bound")

    attempts: list[AttemptRecord] = []
    sanitized_chats: list[dict[str, Any]] = []
    sanitized_generations: list[dict[str, Any]] = []
    generation_payloads: dict[str, dict[str, Any]] = {}
    cumulative = Decimal("0")
    stop_reason: str | None = None
    protocol_violation = False
    other_traffic = False
    pre_usage = readiness.usage
    assert pre_usage is not None

    for item in workload:
        next_conservative = conservative_request_cost_usd(
            prompt_bytes=prompt_utf8_len(item.prompt),
            max_output_tokens=max_tokens,
            tariff=tariff,
        )
        worst_case_next = max(next_conservative, spend.conservative_per_request_usd)
        allowed, reason = budget_allows_next_request(cumulative, worst_case_next, spend)
        if not allowed:
            stop_reason = reason
            protocol_violation = True
            break
        local_id = str(uuid4())
        started = clock.now_utc()
        try:
            http = issue_chat_completion(
                transport,
                api_key=api_key,
                prompt=item.prompt,
                max_tokens=max_tokens,
            )
        except OpenRouterTransportError:
            ended = clock.now_utc()
            attempts.append(
                _failed_attempt(
                    workload_item_id=item.id,
                    local_request_id=local_id,
                    started=started,
                    ended=ended,
                    http_status=None,
                    error_class="network_error",
                    stop_reason="inference_transport_failure",
                )
            )
            stop_reason = "inference_transport_failure"
            protocol_violation = True
            break
        ended = clock.now_utc()
        response_sha = write_private_bytes(raw_dir / "responses" / f"{item.id}.json", http.body)
        if http.status_code == 200:
            try:
                chat_payload = parse_object_payload(http.body)
            except OpenRouterBillingError:
                chat_payload = None
            if chat_payload is not None:
                sanitized_chats.append(
                    {
                        "workload_item_id": item.id,
                        "raw_sha256": response_sha,
                        **sanitize_chat_completion_payload(chat_payload),
                    }
                )
        record = _interpret_chat_response(
            workload_item_id=item.id,
            local_request_id=local_id,
            started=started,
            ended=ended,
            http=http,
            raw_sha=response_sha,
            tariff=tariff,
        )
        if record.generation_id is not None and http.status_code == 200:
            polled = poll_generation_metadata(
                transport,
                api_key=api_key,
                generation_id=record.generation_id,
                policy=poll_policy,
                clock=clock,
            )
            if polled is None:
                record = _replace(record, stop_reason=record.stop_reason or "generation_metadata_missing")
                protocol_violation = True
            else:
                raw, payload = polled
                gen_sha = write_private_bytes(raw_dir / "generations" / f"{record.generation_id}.json", raw)
                generation_payloads[record.workload_item_id] = payload
                sanitized_generations.append(
                    {
                        "workload_item_id": record.workload_item_id,
                        "raw_sha256": gen_sha,
                        **sanitize_generation_payload(payload),
                    }
                )
                record = _with_generation(record, payload, gen_sha)
        elif http.status_code == 200:
            record = _replace(record, stop_reason=record.stop_reason or "generation_id_missing")
            protocol_violation = True

        if record.response_cost is not None:
            cumulative += observed_cost_for_budget(record.response_cost, worst_case_next)
        elif record.attempt_status == "succeeded":
            cumulative += worst_case_next
        attempts.append(record)

        first_request = len(attempts) == 1
        halt = _first_or_runtime_halt(record, first_request=first_request)
        if halt is not None:
            stop_reason = halt
            protocol_violation = True
            break
        if record.attempt_status != "succeeded":
            stop_reason = record.stop_reason or record.error_class or "inference_failure"
            protocol_violation = True
            break

    successful_costs = [item.response_cost for item in attempts if item.response_cost is not None]
    generation_costs = [
        item.generation_total_cost for item in attempts if item.generation_total_cost is not None
    ]
    reconstructed_costs = [
        item.reconstructed_cost for item in attempts if item.reconstructed_cost is not None
    ]
    generation_upstream_costs = [
        item.generation_upstream_inference_cost
        for item in attempts
        if item.generation_upstream_inference_cost is not None
    ]
    response_upstream_costs = [
        item.response_upstream_inference_cost
        for item in attempts
        if item.response_upstream_inference_cost is not None
    ]
    sum_c = _sum_optional(successful_costs)
    sum_d = _sum_optional(generation_costs)
    sum_b = _sum_optional(reconstructed_costs)
    sum_f_generation = _sum_optional(generation_upstream_costs)
    sum_f_response = _sum_optional(response_upstream_costs)

    targets = [value for value in (sum_c, sum_d) if value is not None]
    post_raw, post_payload, delta_e, settled = poll_key_until_settled(
        transport,
        api_key=api_key,
        policy=poll_policy,
        clock=clock,
        target_delta_candidates=targets,
        pre_usage=pre_usage,
    )
    post_sha = write_private_bytes(raw_dir / "post-key.raw.json", post_raw)
    post_acquired = utc_now_iso(clock)
    post_usage = optional_decimal(_key_data(post_payload).get("usage"), field="usage")
    if post_usage is None:
        raise OpenRouterBillingError("post-run usage missing")
    if sum_c is not None and delta_e > sum_c:
        other_traffic = True

    rules = _run_rules(
        attempts=attempts,
        generation_payloads=generation_payloads,
        sum_c=sum_c,
        sum_d=sum_d,
        key_delta=delta_e,
        settled=settled,
        other_traffic=other_traffic,
        semantics=semantics,
    )
    streamed_interpretation = _streamed_field_summary(sanitized_chats, sanitized_generations)
    completed = len(attempts)
    successful = sum(1 for item in attempts if item.attempt_status == "succeeded")
    classification = classify_experiment(
        rules,
        blocked=False,
        completed_requests=completed,
        planned_requests=spend.planned_request_count,
        successful_paid_requests=successful,
        protocol_violation=protocol_violation,
    )
    interval = _interval_from_attempts(attempts)
    provenance = None
    if interval is not None:
        provenance = build_execution_provenance(
            snapshot=snapshot,
            interval=interval,
            workload_sha256=actual_sha,
            declared_status=str(spec.get("status")) if spec.get("status") is not None else None,
            declared_locked_at_utc=(
                str(spec["locked_at_utc"]) if spec.get("locked_at_utc") is not None else None
            ),
            generated_at_utc=utc_now_iso(clock),
            require_spec_precedes_execution=True,
        )

    upstream_names = sorted(
        {
            item.reported_upstream_provider
            for item in attempts
            if item.reported_upstream_provider is not None
        }
    )
    report = {
        "experiment_id": spec["experiment_id"],
        "classification": classification.value,
        "execution_gateway": EXECUTION_GATEWAY,
        "requested_model": REQUESTED_MODEL,
        "source_commit_sha": source_commit,
        "spec_commit_sha": snapshot.spec_commit_sha,
        "spec_sha256": snapshot.spec_sha256,
        "workload_sha256": actual_sha,
        "pricing_table_version": tariff.pricing_table_version,
        "pricing_observed_at": tariff.pricing_observed_at,
        "pricing_source_url": tariff.source_url,
        "user_authorized_maximum_usd": format_decimal(spend.user_authorized_maximum_usd),
        "provider_side_maximum_usd": format_decimal(spend.provider_side_maximum_usd),
        "software_guard_usd": format_decimal(spend.software_guard_usd),
        "conservative_full_run_usd": format_decimal(spend.conservative_full_run_usd),
        "provider_key_limit_usd": format_decimal(readiness.limit) if readiness.limit is not None else None,
        "planned_request_count": spend.planned_request_count,
        "completed_request_count": completed,
        "successful_request_count": successful,
        "failure_count": sum(1 for item in attempts if item.attempt_status != "succeeded"),
        "retry_count": sum(item.retry_count for item in attempts),
        "stop_reason": stop_reason,
        "protocol_violation": protocol_violation,
        "reported_upstream_providers": upstream_names,
        "token_totals": _token_totals(attempts),
        "reconstructed_cost_sum": format_decimal(sum_b) if sum_b is not None else None,
        "response_cost_sum": format_decimal(sum_c) if sum_c is not None else None,
        "generation_total_cost_sum": format_decimal(sum_d) if sum_d is not None else None,
        "key_usage_delta": format_decimal(delta_e),
        "upstream_inference_cost_sum": (
            format_decimal(sum_f_generation) if sum_f_generation is not None else None
        ),
        "generation_upstream_inference_cost_sum": (
            format_decimal(sum_f_generation) if sum_f_generation is not None else None
        ),
        "response_upstream_inference_cost_sum": (
            format_decimal(sum_f_response) if sum_f_response is not None else None
        ),
        "key_usage_settled": settled,
        "other_traffic_suspected": other_traffic,
        "reconciliation_semantics": semantics,
        "rules": [_rule_dict(rule) for rule in rules],
        "streamed_field_interpretation": streamed_interpretation,
        "upstream_cost_interpretation": (
            "FACT: OpenRouter reports two upstream-cost surfaces. "
            "generation.upstream_inference_cost and response usage.cost_details."
            "upstream_inference_cost are stored separately and are not a required "
            "financial gate. NOT independent OpenAI billing. Layer G and final invoice "
            "are NOT OBSERVED."
        ),
        "spend_within_software_guard": (
            (sum_c is not None and sum_c < SOFTWARE_GUARD_USD) or (sum_c is None and cumulative < SOFTWARE_GUARD_USD)
        ),
        "pre_key": {
            "acquired_at_utc": pre_acquired,
            "raw_sha256": pre_sha,
            "sanitized": sanitize_key_payload(pre_payload),
        },
        "post_key": {
            "acquired_at_utc": post_acquired,
            "raw_sha256": post_sha,
            "sanitized": sanitize_key_payload(post_payload),
        },
        "attempts": [_attempt_dict(item) for item in attempts],
        "provenance": provenance,
        "direct_openai_billing_validated": False,
        "final_invoice_validated": False,
    }
    _write_report_bundle(
        output_dir,
        spec=spec,
        tariff=tariff,
        report=report,
        attempts=attempts,
        sanitized_chats=sanitized_chats,
        sanitized_generations=sanitized_generations,
    )
    return report


def _first_or_runtime_halt(record: AttemptRecord, *, first_request: bool) -> str | None:
    if record.is_byok is True:
        return "byok_traffic"
    if record.reasoning_tokens is not None and record.reasoning_tokens > 0:
        return "unexpected_reasoning_tokens"
    if record.response_model is not None and record.response_model != REQUESTED_MODEL:
        return "unexpected_response_model"
    if record.reported_upstream_provider is not None and not allowed_upstream_provider(
        record.reported_upstream_provider
    ):
        return "unexpected_upstream_provider"
    if not allowed_service_tier(record.service_tier):
        return "unexpected_service_tier"
    if record.stop_reason:
        return record.stop_reason
    if first_request:
        if record.attempt_status != "succeeded":
            return record.error_class or "first_request_failed"
        if record.response_cost is None:
            return "missing_response_cost"
        if record.generation_id is None:
            return "generation_id_missing"
        if record.generation_total_cost is None:
            return "missing_generation_total_cost"
        if not allowed_upstream_provider(record.reported_upstream_provider):
            return "unexpected_upstream_provider"
    return None


def _interpret_chat_response(
    *,
    workload_item_id: str,
    local_request_id: str,
    started: datetime,
    ended: datetime,
    http: HttpResponse,
    raw_sha: str,
    tariff: FrozenTariff,
) -> AttemptRecord:
    latency_ms = max(int((ended - started).total_seconds() * 1000), 0)
    if http.status_code != 200:
        return AttemptRecord(
            workload_item_id=workload_item_id,
            local_request_id=local_request_id,
            started_at_utc=started.astimezone(UTC).isoformat(),
            ended_at_utc=ended.astimezone(UTC).isoformat(),
            latency_ms=latency_ms,
            http_status=http.status_code,
            error_class=_http_error_class(http.status_code),
            retry_count=0,
            generation_id=None,
            requested_model=REQUESTED_MODEL,
            response_model=None,
            execution_gateway=EXECUTION_GATEWAY,
            reported_upstream_provider=None,
            upstream_id=None,
            service_tier=None,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cached_input_tokens=None,
            cache_write_tokens=None,
            reasoning_tokens=None,
            response_cost=None,
            reconstructed_cost=None,
            generation_total_cost=None,
            upstream_inference_cost=None,
            response_upstream_inference_cost=None,
            generation_upstream_inference_cost=None,
            cache_discount=None,
            is_byok=None,
            attempt_status="failed",
            charged=False,
            stop_reason=None,
            raw_response_sha256=raw_sha,
            raw_generation_sha256=None,
        )
    payload = parse_object_payload(http.body)
    usage = extract_response_usage(payload)
    model = payload.get("model") if isinstance(payload.get("model"), str) else None
    provider = payload.get("provider") if isinstance(payload.get("provider"), str) else None
    stop_reason = None
    if model is not None and model != REQUESTED_MODEL:
        stop_reason = "unexpected_response_model"
    if provider is not None and not allowed_upstream_provider(provider):
        stop_reason = "unexpected_upstream_provider"
    if usage is not None and usage.is_byok is True:
        stop_reason = "byok_traffic"
    if usage is not None and usage.reasoning_tokens is not None and usage.reasoning_tokens > 0:
        stop_reason = "unexpected_reasoning_tokens"
    reconstructed = None
    if usage is not None:
        reconstructed = reconstruct_openrouter_cost(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            tariff=tariff,
        )
    finish = None
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        finish_raw = choices[0].get("finish_reason")
        finish = str(finish_raw) if finish_raw is not None else None
    generation_id = payload.get("id") if isinstance(payload.get("id"), str) else None
    missing_usage = usage is None
    if missing_usage and stop_reason is None:
        stop_reason = "missing_usage"
    if usage is not None and usage.cost is None and stop_reason is None:
        stop_reason = "missing_response_cost"
    succeeded = usage is not None and stop_reason is None
    return AttemptRecord(
        workload_item_id=workload_item_id,
        local_request_id=local_request_id,
        started_at_utc=started.astimezone(UTC).isoformat(),
        ended_at_utc=ended.astimezone(UTC).isoformat(),
        latency_ms=latency_ms,
        http_status=http.status_code,
        error_class=None,
        retry_count=0,
        generation_id=generation_id,
        requested_model=REQUESTED_MODEL,
        response_model=model,
        execution_gateway=EXECUTION_GATEWAY,
        reported_upstream_provider=provider,
        upstream_id=None,
        service_tier=None,
        finish_reason=finish,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        cached_input_tokens=usage.cached_input_tokens if usage else None,
        cache_write_tokens=usage.cache_write_tokens if usage else None,
        reasoning_tokens=usage.reasoning_tokens if usage else None,
        response_cost=usage.cost if usage else None,
        reconstructed_cost=reconstructed,
        generation_total_cost=None,
        upstream_inference_cost=None,
        response_upstream_inference_cost=usage.upstream_inference_cost if usage else None,
        generation_upstream_inference_cost=None,
        cache_discount=None,
        is_byok=usage.is_byok if usage else None,
        attempt_status="succeeded" if succeeded else "failed",
        charged=usage is not None and usage.cost is not None and usage.cost > 0,
        stop_reason=stop_reason,
        raw_response_sha256=raw_sha,
        raw_generation_sha256=None,
    )


def _with_generation(
    record: AttemptRecord,
    payload: Mapping[str, Any],
    raw_sha: str,
) -> AttemptRecord:
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return _replace(record, stop_reason=record.stop_reason or "generation_payload_missing")
    provider = (
        data.get("provider_name") if isinstance(data.get("provider_name"), str) else record.reported_upstream_provider
    )
    stop_reason = record.stop_reason
    if provider is not None and not allowed_upstream_provider(provider):
        stop_reason = "unexpected_upstream_provider"
    if not allowed_service_tier(data.get("service_tier")):
        stop_reason = "unexpected_service_tier"
    model = data.get("model") if isinstance(data.get("model"), str) else record.response_model
    if model is not None and model != REQUESTED_MODEL:
        stop_reason = "unexpected_response_model"
    if data.get("is_byok") is True:
        stop_reason = "byok_traffic"
    generation_cost = optional_decimal(data.get("total_cost"), field="total_cost") if "total_cost" in data else None
    generation_upstream = (
        optional_decimal(data.get("upstream_inference_cost"), field="upstream_inference_cost")
        if "upstream_inference_cost" in data
        else None
    )
    cache_discount = (
        optional_decimal(data.get("cache_discount"), field="cache_discount") if "cache_discount" in data else None
    )
    succeeded = record.attempt_status == "succeeded" and stop_reason is None
    return AttemptRecord(
        workload_item_id=record.workload_item_id,
        local_request_id=record.local_request_id,
        started_at_utc=record.started_at_utc,
        ended_at_utc=record.ended_at_utc,
        latency_ms=record.latency_ms,
        http_status=record.http_status,
        error_class=record.error_class,
        retry_count=0,
        generation_id=record.generation_id,
        requested_model=record.requested_model,
        response_model=model,
        execution_gateway=record.execution_gateway,
        reported_upstream_provider=provider,
        upstream_id=data.get("upstream_id") if isinstance(data.get("upstream_id"), str) else None,
        service_tier=data.get("service_tier") if isinstance(data.get("service_tier"), str) else None,
        finish_reason=(
            str(data.get("finish_reason")) if data.get("finish_reason") is not None else record.finish_reason
        ),
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        cached_input_tokens=record.cached_input_tokens,
        cache_write_tokens=record.cache_write_tokens,
        reasoning_tokens=record.reasoning_tokens,
        response_cost=record.response_cost,
        reconstructed_cost=record.reconstructed_cost,
        generation_total_cost=generation_cost,
        upstream_inference_cost=generation_upstream,
        response_upstream_inference_cost=record.response_upstream_inference_cost,
        generation_upstream_inference_cost=generation_upstream,
        cache_discount=cache_discount,
        is_byok=True if data.get("is_byok") is True else record.is_byok,
        attempt_status="succeeded" if succeeded else "failed",
        charged=record.charged,
        stop_reason=stop_reason,
        raw_response_sha256=record.raw_response_sha256,
        raw_generation_sha256=raw_sha,
    )


def _replace(record: AttemptRecord, **changes: Any) -> AttemptRecord:
    payload = record.__dict__.copy()
    payload.update(changes)
    return AttemptRecord(**payload)


def _run_rules(
    *,
    attempts: Sequence[AttemptRecord],
    generation_payloads: Mapping[str, Mapping[str, Any]],
    sum_c: Decimal | None,
    sum_d: Decimal | None,
    key_delta: Decimal | None,
    settled: bool,
    other_traffic: bool,
    semantics: str,
) -> list[RuleResult]:
    r1_reconcile = select_r1_reconciler(semantics)
    r1_statuses: list[RuleResult] = []
    r2_statuses: list[RuleResult] = []
    r3_statuses: list[RuleResult] = []
    for attempt in attempts:
        if attempt.attempt_status != "succeeded":
            continue
        generation = generation_payloads.get(attempt.workload_item_id)
        generation_data = generation.get("data") if isinstance(generation, Mapping) else None
        usage = None
        if attempt.prompt_tokens is not None and attempt.completion_tokens is not None:
            usage = ResponseUsage(
                prompt_tokens=attempt.prompt_tokens,
                completion_tokens=attempt.completion_tokens,
                total_tokens=attempt.total_tokens or 0,
                cached_input_tokens=attempt.cached_input_tokens,
                cache_write_tokens=attempt.cache_write_tokens,
                reasoning_tokens=attempt.reasoning_tokens,
                cost=attempt.response_cost,
                upstream_inference_cost=attempt.response_upstream_inference_cost,
                is_byok=attempt.is_byok,
            )
        r1_statuses.append(
            r1_reconcile(
                response=usage,
                generation=generation_data if isinstance(generation_data, Mapping) else None,
            )
        )
        r2_statuses.append(
            reconcile_reconstructed_vs_response_r2(
                reconstructed=attempt.reconstructed_cost,
                response_cost=attempt.response_cost,
                cache_discount=attempt.cache_discount,
            )
        )
        r3_statuses.append(
            reconcile_response_vs_generation_cost_r3(
                response_cost=attempt.response_cost,
                generation_total_cost=attempt.generation_total_cost,
            )
        )
    r1 = _collapse_item_rules("R1", r1_statuses)
    r2 = _collapse_item_rules("R2", r2_statuses)
    r3 = _collapse_item_rules("R3", r3_statuses)
    r4 = reconcile_sum_vs_key_delta(
        rule_id="R4",
        amount_sum=sum_c,
        key_delta=key_delta,
        settled=settled,
        other_traffic_suspected=other_traffic,
    )
    r5 = reconcile_sum_vs_key_delta(
        rule_id="R5",
        amount_sum=sum_d,
        key_delta=key_delta,
        settled=settled,
        other_traffic_suspected=other_traffic,
    )
    return [r1, r2, r3, r4, r5]


def _collapse_item_rules(rule_id: str, results: Sequence[RuleResult]) -> RuleResult:
    if not results:
        return RuleResult(
            rule_id=rule_id,
            required=True,
            status=EvidenceClass.INCONCLUSIVE,
            reason="no successful items to compare",
        )
    status = combine_required_statuses([item.status for item in results])
    return RuleResult(
        rule_id=rule_id,
        required=True,
        status=status,
        reason=f"{rule_id} combined across {len(results)} successful item(s)",
    )


def _write_report_bundle(
    output_dir: Path,
    *,
    spec: Mapping[str, Any],
    tariff: FrozenTariff,
    report: Mapping[str, Any],
    attempts: Sequence[AttemptRecord],
    sanitized_chats: Sequence[Mapping[str, Any]],
    sanitized_generations: Sequence[Mapping[str, Any]],
) -> None:
    decision = {
        "experiment_id": spec["experiment_id"],
        "classification": report["classification"],
        "experiment_classification": CLASSIFICATION,
        "execution_gateway": EXECUTION_GATEWAY,
        "requested_model": REQUESTED_MODEL,
        "direct_openai_billing_validated": False,
        "final_invoice_validated": False,
        "supported_claims_if_validated_match": spec.get("supported_claims"),
        "unsupported_claims": spec.get("unsupported_claims"),
        "rules": report["rules"],
        "upstream_cost_interpretation": report["upstream_cost_interpretation"],
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "reconciliation.json").write_text(
        json.dumps({"rules": report["rules"], "attempts": report["attempts"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pricing-source.json").write_text(
        json.dumps(
            {
                "provider": tariff.provider,
                "model": tariff.model,
                "pricing_table_version": tariff.pricing_table_version,
                "pricing_observed_at": tariff.pricing_observed_at,
                "source_url": tariff.source_url,
                "input_per_token": format_decimal(tariff.input_per_token),
                "cached_input_per_token": format_decimal(tariff.cached_input_per_token),
                "output_per_token": format_decimal(tariff.output_per_token),
                "cache_write_per_token": (
                    format_decimal(tariff.cache_write_per_token) if tariff.cache_write_per_token is not None else None
                ),
                "why_experiment_local": (
                    "Production DEFAULT_PRICING is openai-standard-2026-09-03 for direct OpenAI "
                    "identity. This experiment prices an OpenRouter gateway execution."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "runtime.json").write_text(
        json.dumps(
            {
                "source_commit_sha": report["source_commit_sha"],
                "spec_commit_sha": report["spec_commit_sha"],
                "planned_request_count": report["planned_request_count"],
                "completed_request_count": report["completed_request_count"],
                "stop_reason": report["stop_reason"],
                "protocol_violation": report["protocol_violation"],
                "provenance": report["provenance"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "key-accounting.json").write_text(
        json.dumps({"pre": report["pre_key"], "post": report["post_key"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "request-evidence.jsonl").open("w", encoding="utf-8") as handle:
        for payload in sanitized_chats:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    with (output_dir / "generation-evidence.jsonl").open("w", encoding="utf-8") as handle:
        for payload in sanitized_generations:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    (output_dir / "raw-evidence-manifest.json").write_text(
        json.dumps(
            {
                "pre_key_sha256": report["pre_key"]["raw_sha256"],
                "post_key_sha256": report["post_key"]["raw_sha256"],
                "responses": [
                    {"workload_item_id": item.workload_item_id, "sha256": item.raw_response_sha256}
                    for item in attempts
                ],
                "generations": [
                    {
                        "workload_item_id": item.workload_item_id,
                        "generation_id": item.generation_id,
                        "sha256": item.raw_generation_sha256,
                    }
                    for item in attempts
                    if item.raw_generation_sha256 is not None
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "experiment-spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (output_dir / "decision.md").write_text(_decision_markdown(report, spec), encoding="utf-8")


def _decision_markdown(report: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    lines = [
        f"# OpenRouter gateway billing reconciliation `{spec['experiment_id']}`",
        "",
        f"**Classification:** `{report['classification']}`",
        "",
        "This is GATEWAY_BILLING_VALIDATION. It is not OpenAI billing-ledger validation,",
        "not invoice validation, and not a Change Gate SHIP.",
        "",
        f"- Execution gateway: `{report['execution_gateway']}`",
        f"- Requested model: `{report['requested_model']}`",
        f"- Protocol commit: `{report['spec_commit_sha']}`",
        f"- Source commit at execution: `{report['source_commit_sha']}`",
        f"- Planned requests: `{report['planned_request_count']}`",
        f"- Completed requests: `{report['completed_request_count']}`",
        f"- Successful requests: `{report['successful_request_count']}`",
        f"- Failures: `{report['failure_count']}`",
        f"- Retries: `{report['retry_count']}`",
        f"- Response cost sum C: `{report['response_cost_sum']}`",
        f"- Reconstructed cost sum B: `{report['reconstructed_cost_sum']}`",
        f"- Generation total_cost sum D: `{report['generation_total_cost_sum']}`",
        f"- Key usage delta E: `{report['key_usage_delta']}`",
        f"- Generation upstream_inference_cost sum: `{report.get('generation_upstream_inference_cost_sum')}`",
        f"- Response cost_details.upstream_inference_cost sum: `{report.get('response_upstream_inference_cost_sum')}`",
        f"- Reconciliation semantics: `{report.get('reconciliation_semantics')}`",
        f"- Direct OpenAI billing validated: `{report['direct_openai_billing_validated']}`",
        f"- Final invoice validated: `{report['final_invoice_validated']}`",
        "",
        "## Rules",
        "",
    ]
    for rule in report["rules"]:
        lines.append(f"- `{rule['rule_id']}`: `{rule['status']}` — {rule['reason']}")
    lines.extend(["", "## Upstream cost surfaces", "", str(report["upstream_cost_interpretation"]), ""])
    streamed = report.get("streamed_field_interpretation")
    if isinstance(streamed, Mapping):
        lines.extend(
            [
                "## Streamed metadata",
                "",
                f"- classification: `{streamed.get('classification')}`",
                f"- proves_client_visible_streaming: `{streamed.get('proves_client_visible_streaming')}`",
                f"- reason: {streamed.get('reason')}",
                "",
            ]
        )
    return "\n".join(lines)


def _failed_attempt(
    *,
    workload_item_id: str,
    local_request_id: str,
    started: datetime,
    ended: datetime,
    http_status: int | None,
    error_class: str,
    stop_reason: str | None,
) -> AttemptRecord:
    return AttemptRecord(
        workload_item_id=workload_item_id,
        local_request_id=local_request_id,
        started_at_utc=started.astimezone(UTC).isoformat(),
        ended_at_utc=ended.astimezone(UTC).isoformat(),
        latency_ms=max(int((ended - started).total_seconds() * 1000), 0),
        http_status=http_status,
        error_class=error_class,
        retry_count=0,
        generation_id=None,
        requested_model=REQUESTED_MODEL,
        response_model=None,
        execution_gateway=EXECUTION_GATEWAY,
        reported_upstream_provider=None,
        upstream_id=None,
        service_tier=None,
        finish_reason=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cached_input_tokens=None,
        cache_write_tokens=None,
        reasoning_tokens=None,
        response_cost=None,
        reconstructed_cost=None,
        generation_total_cost=None,
        upstream_inference_cost=None,
        response_upstream_inference_cost=None,
        generation_upstream_inference_cost=None,
        cache_discount=None,
        is_byok=None,
        attempt_status="failed",
        charged=False,
        stop_reason=stop_reason,
        raw_response_sha256=None,
        raw_generation_sha256=None,
    )


def _blocked_result(
    *,
    spec: Mapping[str, Any],
    spend: SpendPolicy,
    reason: str,
    user_action: str | None,
    key_limit: Decimal | None = None,
    key_limit_remaining: Decimal | None = None,
    key_usage: Decimal | None = None,
    pre_key_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": spec["experiment_id"],
        "classification": ExperimentClassification.BLOCKED.value,
        "status": KeyReadinessStatus.READY_AFTER_USER_ACTION.value,
        "reason": reason,
        "user_action": user_action,
        "execution_gateway": EXECUTION_GATEWAY,
        "requested_model": REQUESTED_MODEL,
        "user_authorized_maximum_usd": format_decimal(spend.user_authorized_maximum_usd),
        "provider_side_maximum_usd": format_decimal(spend.provider_side_maximum_usd),
        "software_guard_usd": format_decimal(spend.software_guard_usd),
        "provider_key_limit_usd": format_decimal(key_limit) if key_limit is not None else None,
        "provider_key_limit_remaining_usd": (
            format_decimal(key_limit_remaining) if key_limit_remaining is not None else None
        ),
        "provider_key_usage_usd": format_decimal(key_usage) if key_usage is not None else None,
        "pre_key_sha256": pre_key_sha256,
        "completed_request_count": 0,
        "successful_request_count": 0,
        "retry_count": 0,
        "direct_openai_billing_validated": False,
        "final_invoice_validated": False,
    }


def _interval_from_attempts(attempts: Sequence[AttemptRecord]) -> ExecutionInterval | None:
    if not attempts:
        return None
    starts = [datetime.fromisoformat(item.started_at_utc) for item in attempts]
    ends = [datetime.fromisoformat(item.ended_at_utc) for item in attempts]
    return ExecutionInterval(
        source="openrouter_chat_completions_wall_clock",
        start_field="started_at_utc",
        end_field="ended_at_utc",
        earliest_start_utc=min(starts).isoformat(),
        latest_end_utc=max(ends).isoformat(),
        payload_count=len(attempts),
    )


def _token_totals(attempts: Sequence[AttemptRecord]) -> dict[str, int | None]:
    def _sum_known(values: Sequence[int | None]) -> int | None:
        if any(value is None for value in values):
            if all(value is None for value in values):
                return None
            return None
        return sum(value for value in values if value is not None)

    succeeded = [item for item in attempts if item.attempt_status == "succeeded"]
    return {
        "prompt_tokens": _sum_known([item.prompt_tokens for item in succeeded]),
        "completion_tokens": _sum_known([item.completion_tokens for item in succeeded]),
        "total_tokens": _sum_known([item.total_tokens for item in succeeded]),
        "cached_input_tokens": _sum_known([item.cached_input_tokens for item in succeeded]),
        "cache_write_tokens": _sum_known([item.cache_write_tokens for item in succeeded]),
    }


def _sum_optional(values: Sequence[Decimal]) -> Decimal | None:
    if not values:
        return None
    total = Decimal("0")
    for value in values:
        total += value
    return total


def _key_data(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise OpenRouterBillingError("key payload missing data object")
    return data


def _http_error_class(status: int) -> str:
    if status in {401, 403}:
        return "auth"
    if status == 402:
        return "insufficient_credits"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limited"
    if 500 <= status <= 599:
        return "provider_error"
    return "http_error"


def _rule_dict(rule: RuleResult) -> dict[str, Any]:
    return {
        "rule_id": rule.rule_id,
        "required": rule.required,
        "status": rule.status.value,
        "reason": rule.reason,
        "left": rule.left,
        "right": rule.right,
    }


def _attempt_dict(item: AttemptRecord) -> dict[str, Any]:
    return {
        "workload_item_id": item.workload_item_id,
        "local_request_id": item.local_request_id,
        "started_at_utc": item.started_at_utc,
        "ended_at_utc": item.ended_at_utc,
        "latency_ms": item.latency_ms,
        "http_status": item.http_status,
        "error_class": item.error_class,
        "retry_count": item.retry_count,
        "generation_id": item.generation_id,
        "execution_gateway": item.execution_gateway,
        "requested_model": item.requested_model,
        "response_model": item.response_model,
        "reported_upstream_provider": item.reported_upstream_provider,
        "upstream_id": item.upstream_id,
        "service_tier": item.service_tier,
        "finish_reason": item.finish_reason,
        "prompt_tokens": item.prompt_tokens,
        "completion_tokens": item.completion_tokens,
        "total_tokens": item.total_tokens,
        "cached_input_tokens": item.cached_input_tokens,
        "cache_write_tokens": item.cache_write_tokens,
        "reasoning_tokens": item.reasoning_tokens,
        "response_cost": format_decimal(item.response_cost) if item.response_cost is not None else None,
        "reconstructed_cost": (
            format_decimal(item.reconstructed_cost) if item.reconstructed_cost is not None else None
        ),
        "generation_total_cost": (
            format_decimal(item.generation_total_cost) if item.generation_total_cost is not None else None
        ),
        "upstream_inference_cost": (
            format_decimal(item.generation_upstream_inference_cost)
            if item.generation_upstream_inference_cost is not None
            else None
        ),
        "response_upstream_inference_cost": (
            format_decimal(item.response_upstream_inference_cost)
            if item.response_upstream_inference_cost is not None
            else None
        ),
        "generation_upstream_inference_cost": (
            format_decimal(item.generation_upstream_inference_cost)
            if item.generation_upstream_inference_cost is not None
            else None
        ),
        "attempt_status": item.attempt_status,
        "charged": item.charged,
        "stop_reason": item.stop_reason,
        "raw_response_sha256": item.raw_response_sha256,
        "raw_generation_sha256": item.raw_generation_sha256,
    }


def _streamed_field_summary(
    chats: Sequence[Mapping[str, Any]],
    generations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_item = {
        str(item.get("workload_item_id")): item
        for item in generations
        if isinstance(item, Mapping) and item.get("workload_item_id") is not None
    }
    interpretations: list[StreamedFieldInterpretation] = []
    for chat in chats:
        if not isinstance(chat, Mapping):
            continue
        item_id = chat.get("workload_item_id")
        generation = by_item.get(str(item_id), {})
        data = generation.get("data") if isinstance(generation.get("data"), Mapping) else {}
        streamed_raw = data.get("streamed") if isinstance(data, Mapping) else None
        streamed = streamed_raw if isinstance(streamed_raw, bool) else None
        object_raw = chat.get("object")
        response_object = object_raw if isinstance(object_raw, str) else None
        interpretations.append(
            interpret_generation_streamed_field(
                requested_stream=False,
                response_object=response_object,
                generation_streamed=streamed,
            )
        )
    if any(
        item.classification is StreamedFieldClassification.PROVIDER_METADATA_CONTRACT_ANOMALY
        for item in interpretations
    ):
        chosen = next(
            item
            for item in interpretations
            if item.classification is StreamedFieldClassification.PROVIDER_METADATA_CONTRACT_ANOMALY
        )
    elif interpretations:
        chosen = interpretations[0]
    else:
        chosen = interpret_generation_streamed_field(
            requested_stream=False,
            response_object=None,
            generation_streamed=None,
        )
    return {
        "classification": chosen.classification.value,
        "proves_client_visible_streaming": False,
        "requested_stream": False,
        "reason": chosen.reason,
        "item_count": len(interpretations),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    parsed = load_json_preserving_decimals(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise OpenRouterBillingError(f"{path} must contain a JSON object")
    return parsed


__all__ = [
    "issue_chat_completion",
    "poll_generation_metadata",
    "run_openrouter_billing_experiment",
    "validate_experiment_spec",
]
