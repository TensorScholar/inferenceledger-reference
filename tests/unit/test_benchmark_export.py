from __future__ import annotations

import argparse
import json
from datetime import date

from inference_engine.benchmarking.export import export_run_json, export_run_markdown
from inference_engine.benchmarking.harness import summarize_traces
from inference_engine.benchmarking.sqlite_ledger import ProviderUsageSummary, SQLiteBenchmarkLedger
from inference_engine.domain.cost.pricing import PricingQuote
from inference_engine.infrastructure.telemetry.request_log import RequestTrace, RouteTrace
from scripts.run_benchmark import _export


def _trace() -> RequestTrace:
    return RequestTrace(
        request_id="request-1",
        provider="openai",
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
        quality_passed=True,
        quality_score=1.0,
        quality_reason="passed",
        eval_type="exact_match",
        provider_attempt_count=1,
        provider_retry_count=0,
    )


def _route_quote() -> PricingQuote:
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


def _route() -> RouteTrace:
    quote = _route_quote()
    return RouteTrace(
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
        timestamp="2026-01-01T00:00:00+00:00",
        cost_evidence_complete=True,
        cost_quote=quote,
    )


def test_export_run_json_writes_report_traces_routes_and_reconciliation(tmp_path) -> None:
    traces = [_trace()]
    routes = [_route()]
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=traces,
        route_traces=routes,
    )
    output_path = tmp_path / "run.json"

    export_run_json(
        run_id="run-1",
        report=report,
        traces=traces,
        routes=routes,
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-1"
    assert payload["report"]["request_count"] == 1
    assert payload["traces"][0]["request_id"] == "request-1"
    assert payload["routes"][0]["selected_model"] == "test-model"
    assert payload["routes"][0]["cost_evidence_complete"] is True
    assert payload["routes"][0]["cost_quote"]["pricing_observed_at"] == "2026-09-03"
    assert payload["routes"][0]["cost_quote"]["input_per_million"] == 50.0
    reconciliation = payload["cost_reconciliation"]
    assert reconciliation["paired_request_count"] == 1
    assert reconciliation["comparable_request_count"] == 1
    assert reconciliation["comparable_coverage"] == 1.0
    assert reconciliation["comparable_cost_delta_usd"] == 0.0
    assert reconciliation["matched_rate"] == 1.0
    assert reconciliation["request_reconciliations"][0]["status"] == "comparable_success"


def test_export_run_markdown_writes_reconciliation_summary(tmp_path) -> None:
    traces = [_trace()]
    routes = [_route()]
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=traces,
        route_traces=routes,
    )
    output_path = tmp_path / "run.md"

    export_run_markdown(
        run_id="run-1",
        report=report,
        traces=traces,
        routes=routes,
        output_path=output_path,
        provider_usage_summary=ProviderUsageSummary(
            run_id="run-1",
            request_count=1,
            success_count=1,
            failure_count=0,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated_cost_usd=0.001,
            cost_evidence_complete=True,
            provider_attempt_count=1,
            provider_retry_count=0,
            cost_by_model={"test-model": 0.001},
            tokens_by_model={"test-model": 15},
        ),
    )

    raw = output_path.read_text(encoding="utf-8")
    assert "# Benchmark Run `run-1`" in raw
    assert "## Model Distribution" in raw
    assert "## Observed Latency By Model" in raw
    assert "## Route Reason Distribution" in raw
    assert "## Provider Usage Summary" in raw
    assert "| `test-model` | $0.00100000 |" in raw
    assert "| `test-model` | 15 |" in raw
    assert "- Provider attempts: 1" in raw
    assert "- Provider retries: 0" in raw
    assert "## Route Estimate vs Observed Execution" in raw
    assert "- Comparable coverage: 100.00%" in raw
    assert "- Matched rate: 100.00%" in raw
    assert "`comparable_success`" in raw
    assert "## Route Decisions" in raw
    assert "Pricing evidence for `request-1`" in raw
    assert "`openai:test-model:2026-09-03`" in raw
    assert "input=10, output=5, cached_input=0" in raw
    assert "## Limitations" in raw


def test_export_cli_writes_both_formats_with_reconciliation(tmp_path) -> None:
    ledger = SQLiteBenchmarkLedger(tmp_path / "ledger.sqlite3")
    traces = [_trace()]
    routes = [_route()]
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=traces,
        route_traces=routes,
    )
    ledger.record_run(run_id="run-1", report=report, traces=traces, route_traces=routes)

    exit_code = _export(
        argparse.Namespace(
            sqlite_ledger_path=str(tmp_path / "ledger.sqlite3"),
            run_id="run-1",
            output_dir=str(tmp_path / "exports"),
            format="both",
        )
    )

    assert exit_code == 0
    json_path = tmp_path / "exports" / "run-1.json"
    markdown_path = tmp_path / "exports" / "run-1.md"
    assert json_path.exists()
    assert markdown_path.exists()
    exported = json.loads(json_path.read_text(encoding="utf-8"))
    assert exported["routes"][0]["cost_quote"]["pricing_record_id"] == (
        "openai:test-model:2026-09-03"
    )
    assert exported["cost_reconciliation"]["comparable_request_count"] == 1
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Provider Usage Summary" in markdown
    assert "## Route Estimate vs Observed Execution" in markdown
