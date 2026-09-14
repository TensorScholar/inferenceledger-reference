from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from inference_engine.benchmarking.harness import summarize_traces
from inference_engine.benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from inference_engine.domain.cost.pricing import PricingQuote
from inference_engine.infrastructure.telemetry.request_log import RequestTrace, RouteTrace


def _quote() -> PricingQuote:
    observed_at = date(2026, 9, 3)
    return PricingQuote(
        amount_usd=0.001,
        provider="openai",
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        cached_input_tokens=0,
        input_per_million=50.0,
        output_per_million=100.0,
        cached_input_per_million=None,
        pricing_record_id=f"openai:test-model:{observed_at.isoformat()}",
        pricing_table_version="test-route-v1",
        pricing_observed_at=observed_at,
        pricing_source_url="https://pricing.example/test-model",
    )


def test_sqlite_route_reader_rejects_tampered_pricing_rates(tmp_path) -> None:
    path = tmp_path / "benchmarks.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    trace = RequestTrace(
        request_id="request-1",
        provider="openai",
        model="test-model",
        latency_ms=100,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.001,
        pricing_table_version="test",
        cache_hit=False,
        error_type=None,
        error_message=None,
        timestamp="2026-09-03T00:00:00+00:00",
        cost_evidence_complete=True,
    )
    quote = _quote()
    route = RouteTrace(
        request_id="request-1",
        strategy="single_model",
        selected_model="test-model",
        estimated_cost_usd=quote.amount_usd,
        estimated_latency_ms=250,
        decision_reason="single model",
        considered_models=["test-model"],
        fallback_models=[],
        max_estimated_cost_usd=0.01,
        budget_violation=False,
        budget_violation_reason=None,
        timestamp="2026-09-03T00:00:00+00:00",
        cost_evidence_complete=True,
        cost_quote=quote,
    )
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=[trace],
        route_traces=[route],
    )
    ledger.record_run(run_id="run-1", report=report, traces=[trace], route_traces=[route])

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE benchmark_route_cost_evidence
            SET input_per_million = 500.0
            WHERE run_id = 'run-1' AND request_id = 'request-1'
            """
        )

    with pytest.raises(ValueError, match="token and rate assumptions"):
        ledger.get_routes("run-1")
