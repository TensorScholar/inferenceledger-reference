from __future__ import annotations

import argparse
import json
from datetime import date

import pytest

import scripts.run_benchmark as benchmark_script
from inference_engine.benchmarking.harness import summarize_traces
from inference_engine.benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from inference_engine.cli import _run_smoke
from inference_engine.domain.cost.pricing import PricingQuote
from inference_engine.domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
    ProviderAttempt,
)
from inference_engine.domain.models.routing import (
    ModelConfig,
    ModelTier,
    RoutingDecision,
    RoutingStrategy,
)
from inference_engine.domain.routing.policy import PolicyRouter
from inference_engine.infrastructure.telemetry.request_log import RequestTrace


@pytest.mark.asyncio
async def test_smoke_cli_requires_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = argparse.Namespace(
        provider="openai",
        model="gpt-4o-mini",
        prompt="hello",
        base_url=None,
        timeout_seconds=30.0,
        max_tokens=16,
        temperature=0.0,
        log_path="unused.jsonl",
    )

    exit_code = await _run_smoke(args)

    assert exit_code == 2


@pytest.mark.asyncio
async def test_benchmark_budget_violation_skips_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        '{"id":"one","prompt":"hello","eval":{"type":"contains_all","required":["hello"]}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeRouter:
        async def route(self, request):
            model = ModelConfig(
                id="test-model",
                name="Test Model",
                tier=ModelTier.STANDARD,
                max_context_length=4096,
            )
            observed_at = date(2026, 9, 3)
            input_tokens = request.estimated_input_tokens
            output_tokens = request.parameters.max_tokens
            total_tokens = input_tokens + output_tokens
            rate_per_million = 0.02 * 1_000_000 / total_tokens
            quote = PricingQuote(
                amount_usd=0.02,
                provider="openai",
                model="test-model",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=0,
                input_per_million=rate_per_million,
                output_per_million=rate_per_million,
                cached_input_per_million=None,
                pricing_record_id=f"openai:test-model:{observed_at.isoformat()}",
                pricing_table_version="test-budget-v1",
                pricing_observed_at=observed_at,
                pricing_source_url="https://pricing.example/test-model",
            )
            return RoutingDecision(
                request_id=request.id,
                selected_model=model,
                strategy=RoutingStrategy.SINGLE_MODEL,
                complexity_estimate=None,
                estimated_cost=quote.amount_usd,
                estimated_latency_ms=100,
                estimated_quality_score=0.7,
                decision_reason="fake expensive route",
                fallback_models=[],
                cost_quote=quote,
                considered_models=["test-model"],
            )

    class FailingBackend:
        def __init__(self, *args, **kwargs):
            raise AssertionError("provider backend should not be constructed")

    monkeypatch.setattr(benchmark_script, "_build_router", lambda _args: FakeRouter())
    monkeypatch.setattr(benchmark_script, "OpenAIBackend", FailingBackend)
    args = argparse.Namespace(
        provider="openai",
        model="test-model",
        strategy="single_model",
        economy_model="test-model",
        standard_model="test-model",
        premium_model="test-model",
        workload=str(workload_path),
        base_url=None,
        timeout_seconds=30.0,
        max_tokens=16,
        temperature=0.0,
        ledger_path=str(tmp_path / "ledger.jsonl"),
        route_ledger_path=str(tmp_path / "routes.jsonl"),
        sqlite_ledger_path=str(tmp_path / "ledger.sqlite3"),
        report_path=str(tmp_path / "report.json"),
        run_id="budget-test",
        max_estimated_cost_usd=0.001,
    )

    exit_code = await benchmark_script._run(args)

    assert exit_code == 1
    ledger_raw = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    ledger_row = json.loads(ledger_raw)
    assert ledger_row["error_type"] == "budget_violation"
    assert ledger_row["provider_attempt_count"] == 0
    assert ledger_row["provider_retry_count"] == 0
    assert ledger_row["estimated_cost_usd"] == 0.0
    assert ledger_row["cost_evidence_complete"] is True
    route_row = json.loads((tmp_path / "routes.jsonl").read_text(encoding="utf-8"))
    assert "fake expensive route" in route_row["decision_reason"]
    assert route_row["cost_evidence_complete"] is True
    assert route_row["cost_quote"]["pricing_record_id"] == "openai:test-model:2026-09-03"


def test_benchmark_build_router_supports_policy_strategy() -> None:
    args = argparse.Namespace(
        strategy="policy",
        model="gpt-4o-mini",
        economy_model="gpt-4o-mini",
        standard_model="gpt-4o-mini",
        premium_model="gpt-4o",
        max_estimated_cost_usd=0.002,
        policy_latency_slo_ms=800,
        policy_min_quality_score=0.70,
        policy_cost_weight=0.55,
        policy_latency_weight=0.25,
        policy_quality_weight=0.20,
    )

    router = benchmark_script._build_router(args)

    assert isinstance(router, PolicyRouter)
    assert router.config.max_estimated_cost_usd == 0.002
    assert router.config.latency_slo_ms == 800
    assert router.config.min_quality_score == 0.70


def test_custom_benchmark_endpoint_requires_explicit_provider_identities() -> None:
    args = argparse.Namespace(base_url="https://provider.example/v1")

    with pytest.raises(ValueError, match="execution-provider"):
        benchmark_script._resolve_execution_provider(args)
    with pytest.raises(ValueError, match="pricing-provider"):
        benchmark_script._resolve_pricing_provider(args)


def _priced_attempt(
    *,
    attempt_index: int,
    outcome: AttemptOutcome,
    cost: float,
    error_type: str | None = None,
    status_code: int | None = None,
) -> ProviderAttempt:
    return ProviderAttempt(
        attempt_index=attempt_index,
        provider="test-provider",
        model="test-model",
        outcome=outcome,
        latency_ms=20,
        prompt_tokens=4 if attempt_index == 1 else 6,
        completion_tokens=2 if attempt_index == 1 else 3,
        total_tokens=6 if attempt_index == 1 else 9,
        cached_tokens=0,
        calculated_cost_usd=cost,
        cost_evidence=CostEvidenceKind.CALCULATED_FROM_USAGE,
        pricing_table_version="test",
        pricing_record_id="test-provider:test-model:2026-09-03",
        pricing_observed_at="2026-09-03",
        pricing_source_url="https://pricing.example/test-model",
        error_type=error_type,
        status_code=status_code,
    )


def test_benchmark_usage_summary_cli_writes_provider_usage_json(tmp_path) -> None:
    ledger = SQLiteBenchmarkLedger(tmp_path / "ledger.sqlite3")
    attempts = (
        _priced_attempt(
            attempt_index=1,
            outcome=AttemptOutcome.FAILED,
            cost=0.0004,
            error_type="server_error",
            status_code=500,
        ),
        _priced_attempt(
            attempt_index=2,
            outcome=AttemptOutcome.SUCCEEDED,
            cost=0.0006,
        ),
    )
    trace = RequestTrace(
        request_id="request-1",
        provider="test-provider",
        model="test-model",
        latency_ms=123,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.001,
        pricing_table_version="test",
        cache_hit=False,
        error_type=None,
        error_message=None,
        timestamp="2026-01-01T00:00:00+00:00",
        provider_attempts=attempts,
    )
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="test-provider",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=[trace],
    )
    ledger.record_run(run_id="run-1", report=report, traces=[trace])

    output_path = tmp_path / "usage.json"
    exit_code = benchmark_script._usage_summary(
        argparse.Namespace(
            sqlite_ledger_path=str(tmp_path / "ledger.sqlite3"),
            run_id="run-1",
            output_path=str(output_path),
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["run_id"] == "run-1"
    assert payload["estimated_cost_usd"] == 0.001
    assert payload["cost_evidence_complete"] is True
    assert payload["provider_attempt_count"] == 2
    assert payload["provider_retry_count"] == 1
    assert payload["cost_by_model"] == {"test-model": 0.001}
