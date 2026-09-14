from __future__ import annotations

from decimal import Decimal

import pytest

from inference_engine.benchmarking.openrouter_billing import (
    EXECUTION_GATEWAY,
    REQUESTED_MODEL,
    EvidenceClass,
    ExperimentClassification,
    FrozenTariff,
    KeyReadinessStatus,
    OpenRouterBillingError,
    ResponseUsage,
    SpendPolicy,
    allowed_upstream_provider,
    assess_key_readiness,
    budget_allows_next_request,
    chat_completion_body,
    chat_completion_headers,
    classify_experiment,
    conservative_full_run_cost_usd,
    conservative_request_cost_usd,
    env_key_presence,
    extract_response_usage,
    generate_workload_items,
    key_usage_delta,
    load_json_preserving_decimals,
    reconcile_reconstructed_vs_response_r2,
    reconcile_response_vs_generation_cost_r3,
    reconcile_sum_vs_key_delta,
    reconcile_tokens_r1,
    reconstruct_openrouter_cost,
    render_workload_jsonl,
    require_openrouter_gateway,
    sanitize_chat_completion_payload,
    sanitize_generation_payload,
    sanitize_key_payload,
)
from inference_engine.domain.cost.pricing import DEFAULT_PRICING, PRICING_TABLE_VERSION

EXPERIMENT_ID = "openrouter-billing-reconciliation-6b-20260911T220449Z"


def _tariff() -> FrozenTariff:
    return FrozenTariff(
        provider="openrouter",
        model=REQUESTED_MODEL,
        pricing_table_version="openrouter-gpt-4o-mini-2024-07-18-2026-09-11",
        pricing_observed_at="2026-09-11T22:02:53.267052+00:00",
        source_url="https://openrouter.ai/api/v1/models/openai/gpt-4o-mini-2024-07-18/endpoints",
        input_per_token=Decimal("0.00000015"),
        cached_input_per_token=Decimal("0.000000075"),
        output_per_token=Decimal("0.0000006"),
        cache_write_per_token=None,
    )


def _spend(**overrides: object) -> SpendPolicy:
    per = conservative_request_cost_usd(prompt_bytes=3000, max_output_tokens=8, tariff=_tariff())
    payload: dict[str, object] = {
        "user_authorized_maximum_usd": Decimal("0.20"),
        "provider_side_maximum_usd": Decimal("0.05"),
        "software_guard_usd": Decimal("0.025"),
        "planned_request_count": 50,
        "prompt_byte_bound": 3000,
        "max_output_tokens": 8,
        "conservative_per_request_usd": per,
        "conservative_full_run_usd": per * 50,
    }
    payload.update(overrides)
    if "planned_request_count" in overrides and "conservative_full_run_usd" not in overrides:
        count = int(payload["planned_request_count"])  # type: ignore[arg-type]
        payload["conservative_full_run_usd"] = payload["conservative_per_request_usd"] * count  # type: ignore[operator]
    return SpendPolicy(**payload)  # type: ignore[arg-type]


def test_execution_gateway_cannot_be_direct_openai() -> None:
    with pytest.raises(OpenRouterBillingError, match="openrouter"):
        require_openrouter_gateway("openai")
    require_openrouter_gateway(EXECUTION_GATEWAY)


def test_production_pricing_table_is_not_silently_openrouter() -> None:
    assert PRICING_TABLE_VERSION.startswith("openai-standard-")
    assert all(provider != "openrouter" for provider, _model in DEFAULT_PRICING)


def test_frozen_tariff_rejects_openai_provider_identity() -> None:
    with pytest.raises(OpenRouterBillingError, match="openrouter"):
        FrozenTariff(
            provider="openai",
            model=REQUESTED_MODEL,
            pricing_table_version="x",
            pricing_observed_at="2026-09-11T22:02:53.267052+00:00",
            source_url="https://example.invalid",
            input_per_token=Decimal("0.00000015"),
            cached_input_per_token=Decimal("0.000000075"),
            output_per_token=Decimal("0.0000006"),
            cache_write_per_token=None,
        )


def test_chat_body_pins_openai_without_fallbacks_or_cache_header() -> None:
    body = chat_completion_body(
        model=REQUESTED_MODEL,
        prompt="hello",
        max_tokens=8,
        temperature=0,
        provider_only=["openai"],
        provider_order=["openai"],
        allow_fallbacks=False,
    )
    headers = chat_completion_headers("test-key")
    assert body["provider"] == {"only": ["openai"], "order": ["openai"], "allow_fallbacks": False}
    assert body["stream"] is False
    assert "X-OpenRouter-Cache" not in headers
    assert "Authorization" in headers


def test_chat_body_rejects_fallbacks_and_non_openai_order() -> None:
    with pytest.raises(OpenRouterBillingError, match="fallbacks"):
        chat_completion_body(
            model=REQUESTED_MODEL,
            prompt="hello",
            max_tokens=8,
            temperature=0,
            provider_only=["openai"],
            provider_order=["openai"],
            allow_fallbacks=True,
        )
    with pytest.raises(OpenRouterBillingError, match="provider only/order"):
        chat_completion_body(
            model=REQUESTED_MODEL,
            prompt="hello",
            max_tokens=8,
            temperature=0,
            provider_only=["azure"],
            provider_order=["openai"],
            allow_fallbacks=False,
        )


def test_response_cost_parsed_from_decimal_json_text() -> None:
    payload = load_json_preserving_decimals(
        """
        {
          "id": "gen-1",
          "model": "openai/gpt-4o-mini-2024-07-18",
          "provider": "OpenAI",
          "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cost": 0.0000027,
            "prompt_tokens_details": {"cached_tokens": 0}
          },
          "choices": [{"finish_reason": "stop", "message": {"content": "secret"}}]
        }
        """
    )
    assert isinstance(payload, dict)
    usage = extract_response_usage(payload)
    assert usage is not None
    assert usage.cost == Decimal("0.0000027")
    assert usage.cached_input_tokens == 0
    sanitized = sanitize_chat_completion_payload(payload)
    assert sanitized["choices"][0].get("message") is None
    assert "secret" not in str(sanitized)


def test_missing_cached_tokens_are_not_coerced_to_zero() -> None:
    payload = load_json_preserving_decimals(
        '{"usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11, "cost": 0.1}}'
    )
    assert isinstance(payload, dict)
    usage = extract_response_usage(payload)
    assert usage is not None
    assert usage.cached_input_tokens is None
    assert usage.cache_write_tokens is None
    assert (
        reconstruct_openrouter_cost(
            prompt_tokens=10,
            completion_tokens=1,
            cached_input_tokens=None,
            cache_write_tokens=None,
            tariff=_tariff(),
        )
        is None
    )


def test_generation_cost_parsed_from_decimal_json_text() -> None:
    payload = load_json_preserving_decimals(
        '{"data": {"id": "gen-1", "total_cost": 0.0000027, "upstream_inference_cost": 0.0000021, "creator_user_id": "user_secret"}}'
    )
    assert isinstance(payload, dict)
    data = payload["data"]
    assert isinstance(data, dict)
    assert data["total_cost"] == Decimal("0.0000027")
    sanitized = sanitize_generation_payload(payload)
    assert "creator_user_id" not in sanitized["data"]
    assert sanitized["data"]["total_cost"] == "0.0000027"


def test_reconstructed_cost_matches_uncached_tariff() -> None:
    amount = reconstruct_openrouter_cost(
        prompt_tokens=1000,
        completion_tokens=16,
        cached_input_tokens=0,
        cache_write_tokens=0,
        tariff=_tariff(),
    )
    assert amount == Decimal("1000") * Decimal("0.00000015") + Decimal("16") * Decimal("0.0000006")


def test_cached_token_reconstruction_uses_cached_rate() -> None:
    amount = reconstruct_openrouter_cost(
        prompt_tokens=1000,
        completion_tokens=10,
        cached_input_tokens=400,
        cache_write_tokens=0,
        tariff=_tariff(),
    )
    expected = (
        Decimal("600") * Decimal("0.00000015")
        + Decimal("400") * Decimal("0.000000075")
        + Decimal("10") * Decimal("0.0000006")
    )
    assert amount == expected
    match = reconcile_reconstructed_vs_response_r2(
        reconstructed=amount,
        response_cost=expected,
        cache_discount=Decimal("0"),
    )
    assert match.status is EvidenceClass.MATCH


def test_nonzero_cache_write_without_listed_rate_is_not_reconstructable() -> None:
    amount = reconstruct_openrouter_cost(
        prompt_tokens=100,
        completion_tokens=1,
        cached_input_tokens=0,
        cache_write_tokens=40,
        tariff=_tariff(),
    )
    assert amount is None


def test_r2_exact_match_and_mismatch() -> None:
    reconstructed = Decimal("0.0001")
    assert (
        reconcile_reconstructed_vs_response_r2(
            reconstructed=reconstructed,
            response_cost=Decimal("0.0001"),
            cache_discount=None,
        ).status
        is EvidenceClass.MATCH
    )
    assert (
        reconcile_reconstructed_vs_response_r2(
            reconstructed=reconstructed,
            response_cost=Decimal("0.0002"),
            cache_discount=None,
        ).status
        is EvidenceClass.MISMATCH
    )


def test_missing_cost_is_inconclusive() -> None:
    assert (
        reconcile_reconstructed_vs_response_r2(
            reconstructed=Decimal("0.1"),
            response_cost=None,
            cache_discount=None,
        ).status
        is EvidenceClass.INCONCLUSIVE
    )
    assert (
        reconcile_response_vs_generation_cost_r3(
            response_cost=None,
            generation_total_cost=Decimal("0.1"),
        ).status
        is EvidenceClass.INCONCLUSIVE
    )


def test_nonzero_cache_discount_without_equality_is_not_comparable() -> None:
    result = reconcile_reconstructed_vs_response_r2(
        reconstructed=Decimal("0.0002"),
        response_cost=Decimal("0.0001"),
        cache_discount=Decimal("0.00005"),
    )
    assert result.status is EvidenceClass.NOT_COMPARABLE


def test_r3_response_versus_generation_cost() -> None:
    assert (
        reconcile_response_vs_generation_cost_r3(
            response_cost=Decimal("0.0000027"),
            generation_total_cost=Decimal("0.0000027"),
        ).status
        is EvidenceClass.MATCH
    )
    assert (
        reconcile_response_vs_generation_cost_r3(
            response_cost=Decimal("0.0000027"),
            generation_total_cost=Decimal("0.0000030"),
        ).status
        is EvidenceClass.MISMATCH
    )


def test_key_usage_delta_and_other_traffic_inconclusive() -> None:
    delta = key_usage_delta(pre_usage=Decimal("1.00"), post_usage=Decimal("1.01"))
    assert delta == Decimal("0.01")
    other = reconcile_sum_vs_key_delta(
        rule_id="R4",
        amount_sum=Decimal("0.01"),
        key_delta=Decimal("0.02"),
        settled=True,
        other_traffic_suspected=True,
    )
    assert other.status is EvidenceClass.INCONCLUSIVE
    mismatch = reconcile_sum_vs_key_delta(
        rule_id="R4",
        amount_sum=Decimal("0.01"),
        key_delta=Decimal("0.02"),
        settled=True,
        other_traffic_suspected=False,
    )
    assert mismatch.status is EvidenceClass.MISMATCH
    unsettled = reconcile_sum_vs_key_delta(
        rule_id="R4",
        amount_sum=Decimal("0.01"),
        key_delta=Decimal("0.009"),
        settled=False,
        other_traffic_suspected=False,
    )
    assert unsettled.status is EvidenceClass.INCONCLUSIVE


def test_r1_token_equality_ignores_divergent_native_fields() -> None:
    usage = ResponseUsage(
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        cached_input_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        cost=Decimal("0.0000027"),
        upstream_inference_cost=None,
        is_byok=False,
    )
    match = reconcile_tokens_r1(
        response=usage,
        generation={
            "tokens_prompt": 10,
            "tokens_completion": 2,
            "native_tokens_prompt": 12,
            "native_tokens_cached": 0,
        },
    )
    assert match.status is EvidenceClass.MATCH
    mismatch = reconcile_tokens_r1(
        response=usage,
        generation={"tokens_prompt": 11, "tokens_completion": 2},
    )
    assert mismatch.status is EvidenceClass.MISMATCH


def test_null_or_excessive_key_limit_blocks_inference() -> None:
    spend = _spend()
    null_limit = assess_key_readiness({"data": {"limit": None, "limit_remaining": 1, "usage": 0}}, spend=spend)
    assert null_limit.status is KeyReadinessStatus.READY_AFTER_USER_ACTION
    assert null_limit.ready is False
    high = assess_key_readiness(
        {"data": {"limit": Decimal("0.20"), "limit_remaining": Decimal("0.20"), "usage": Decimal("0")}},
        spend=spend,
    )
    assert high.ready is False
    assert high.limit == Decimal("0.20")
    slightly_high = assess_key_readiness(
        {"data": {"limit": Decimal("0.06"), "limit_remaining": Decimal("0.06"), "usage": Decimal("0")}},
        spend=spend,
    )
    assert slightly_high.ready is False
    ready = assess_key_readiness(
        {
            "data": {
                "limit": Decimal("0.05"),
                "limit_remaining": Decimal("0.05"),
                "usage": Decimal("0"),
            }
        },
        spend=spend,
    )
    assert ready.ready is True


def test_contaminated_nonzero_usage_blocks_inference() -> None:
    result = assess_key_readiness(
        {
            "data": {
                "limit": Decimal("0.05"),
                "limit_remaining": Decimal("0.04"),
                "usage": Decimal("0.01"),
            }
        },
        spend=_spend(),
    )
    assert result.ready is False
    assert "contaminated" in (result.user_action or "")


def test_insufficient_remaining_limit_blocks_inference() -> None:
    spend = _spend()
    result = assess_key_readiness(
        {
            "data": {
                "limit": Decimal("0.05"),
                "limit_remaining": Decimal("0.01"),
                "usage": Decimal("0"),
            }
        },
        spend=spend,
    )
    assert result.ready is False
    assert "limit_remaining" in (result.user_action or "")


def test_software_guard_is_strictly_below_0_025() -> None:
    spend = _spend()
    allowed, _reason = budget_allows_next_request(Decimal("0"), spend.conservative_per_request_usd, spend)
    assert allowed is True
    blocked, reason = budget_allows_next_request(Decimal("0.0246"), spend.conservative_per_request_usd, spend)
    assert blocked is False
    assert "software spend guard" in reason
    at_guard, at_reason = budget_allows_next_request(Decimal("0.025"), Decimal("0.0000001"), spend)
    assert at_guard is False
    assert "software spend guard" in at_reason


def test_request_size_budget_from_frozen_tariff() -> None:
    full = conservative_full_run_cost_usd(
        prompt_byte_bound=3000,
        request_count=50,
        max_output_tokens=8,
        tariff=_tariff(),
    )
    assert full == Decimal("0.02274")
    assert full < Decimal("0.025")


def test_key_sanitizer_removes_identifiers_and_secrets() -> None:
    sanitized = sanitize_key_payload(
        {
            "data": {
                "creator_user_id": "user_secret",
                "label": "sk-or-v1-example",
                "usage": Decimal("0"),
                "limit": Decimal("0.05"),
                "limit_remaining": Decimal("0.05"),
            }
        }
    )
    dumped = str(sanitized)
    assert "creator_user_id" not in sanitized["data"]
    assert "label" not in sanitized["data"]
    assert "sk-or-" not in dumped
    assert "user_secret" not in dumped
    assert sanitized["data"]["limit"] == "0.05"


def test_env_presence_does_not_echo_key() -> None:
    assert env_key_presence({}) == "ABSENT"
    assert env_key_presence({"OPENROUTER_API_KEY": "secret-value"}) == "PRESENT"


def test_upstream_provider_allows_openai_only() -> None:
    assert allowed_upstream_provider("OpenAI") is True
    assert allowed_upstream_provider("openai") is True
    assert allowed_upstream_provider("Azure") is False


def test_experiment_classification_requires_comparable_agreement() -> None:
    match = reconcile_response_vs_generation_cost_r3(
        response_cost=Decimal("1"),
        generation_total_cost=Decimal("1"),
    )
    mismatch = reconcile_response_vs_generation_cost_r3(
        response_cost=Decimal("1"),
        generation_total_cost=Decimal("2"),
    )
    missing = reconcile_response_vs_generation_cost_r3(
        response_cost=None,
        generation_total_cost=Decimal("1"),
    )
    assert (
        classify_experiment(
            [match, match, match, match, match],
            blocked=False,
            completed_requests=50,
            planned_requests=50,
            successful_paid_requests=50,
        )
        is ExperimentClassification.VALIDATED_MATCH
    )
    assert (
        classify_experiment(
            [match, mismatch],
            blocked=False,
            completed_requests=50,
            planned_requests=50,
            successful_paid_requests=50,
        )
        is ExperimentClassification.VALIDATED_MISMATCH
    )
    assert (
        classify_experiment(
            [match, missing],
            blocked=False,
            completed_requests=50,
            planned_requests=50,
            successful_paid_requests=50,
        )
        is ExperimentClassification.INCONCLUSIVE
    )
    assert (
        classify_experiment(
            [match, match, match, match, match],
            blocked=False,
            completed_requests=1,
            planned_requests=50,
            successful_paid_requests=1,
            protocol_violation=True,
        )
        is ExperimentClassification.INCONCLUSIVE
    )
    assert (
        classify_experiment(
            [],
            blocked=True,
            completed_requests=0,
            planned_requests=50,
            successful_paid_requests=0,
        )
        is ExperimentClassification.BLOCKED
    )


def test_workload_is_unique_and_byte_bounded() -> None:
    items = generate_workload_items(EXPERIMENT_ID)
    assert len(items) == 50
    ids = [item["id"] for item in items]
    prompts = [str(item["prompt"]) for item in items]
    assert len(set(ids)) == 50
    assert len(set(prompts)) == 50
    for prompt in prompts:
        assert len(prompt.encode("utf-8")) <= 3000
    rendered = render_workload_jsonl(EXPERIMENT_ID)
    assert rendered.count("\n") == 50


def test_negative_key_delta_fails_closed() -> None:
    with pytest.raises(OpenRouterBillingError, match="lower than the pre-run"):
        key_usage_delta(pre_usage=Decimal("2"), post_usage=Decimal("1"))
