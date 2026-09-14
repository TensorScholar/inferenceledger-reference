from __future__ import annotations

import argparse
import json
from datetime import date

import pytest

import scripts.run_benchmark as benchmark_script
from inference_engine.benchmarking.context_store import SQLiteBenchmarkContextStore
from inference_engine.domain.cost.pricing import PricingQuote
from inference_engine.domain.models.routing import (
    ModelConfig,
    ModelTier,
    RoutingDecision,
    RoutingStrategy,
)


@pytest.mark.asyncio
async def test_benchmark_run_captures_tags_and_writes_segment_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        '{"id":"item-1","prompt":"hello","tags":{"task":"qa","risk":"high"}}\n',
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
                pricing_table_version="test-v1",
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
                decision_reason="force budget rejection",
                fallback_models=[],
                cost_quote=quote,
                considered_models=["test-model"],
            )

    class FailingBackend:
        def __init__(self, *args, **kwargs):
            raise AssertionError("provider backend should not be constructed")

    monkeypatch.setattr(benchmark_script, "_build_router", lambda _args: FakeRouter())
    monkeypatch.setattr(benchmark_script, "OpenAIBackend", FailingBackend)
    sqlite_path = tmp_path / "ledger.sqlite3"
    segment_path = tmp_path / "segments.json"
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
        sqlite_ledger_path=str(sqlite_path),
        report_path=str(tmp_path / "report.json"),
        segment_report_path=str(segment_path),
        run_id="segment-test",
        max_estimated_cost_usd=0.001,
    )

    exit_code = await benchmark_script._run(args)

    assert exit_code == 1
    contexts = SQLiteBenchmarkContextStore(sqlite_path).get_contexts("segment-test")
    assert len(contexts) == 1
    assert contexts[0].workload_item_id == "item-1"
    assert contexts[0].tags_dict() == {"risk": "high", "task": "qa"}

    payload = json.loads(segment_path.read_text(encoding="utf-8"))
    assert payload["available"] is True
    assert payload["request_count"] == 1
    assert payload["segment_count"] == 2
    task = next(
        segment
        for segment in payload["segments"]
        if segment["tag_key"] == "task" and segment["tag_value"] == "qa"
    )
    assert task["request_count"] == 1
    assert task["failure_count"] == 1
    assert task["estimated_cost_usd"] == 0.0
    assert task["cost_evidence_complete"] is True
    assert task["cost_per_success_usd"] is None
