from __future__ import annotations

import json
import sqlite3
from datetime import date

import pytest

from inference_engine.benchmarking.harness import summarize_traces
from inference_engine.benchmarking.reconciliation import (
    ReconciliationStatus,
    reconcile_request_cost,
)
from inference_engine.benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from inference_engine.domain.cost.pricing import PricingQuote
from inference_engine.domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
    ProviderAttempt,
)
from inference_engine.infrastructure.importers.litellm import (
    LITELLM_STANDARD_LOGGING_SOURCE,
    import_litellm_standard_logging_chain,
)
from inference_engine.infrastructure.telemetry.request_log import RouteTrace


def _payload(
    *,
    record_id: str = "record-1",
    trace_id: str = "trace-1",
    call_id: str = "call-1",
    provider: str = "openai",
    model: str = "model-a",
    status: str = "success",
    start_time: float = 100.0,
    end_time: float = 100.2,
    response_time: float = 0.2,
    response_cost: float | None = 0.0012,
    cache_hit: bool | None = False,
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    total_tokens: int = 120,
    error_information: object = None,
) -> dict[str, object]:
    return {
        "id": record_id,
        "trace_id": trace_id,
        "litellm_call_id": call_id,
        "custom_llm_provider": provider,
        "model": model,
        "status": status,
        "startTime": start_time,
        "endTime": end_time,
        "response_time": response_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "response_cost": response_cost,
        "response_cost_failure_debug_info": None,
        "cache_hit": cache_hit,
        "error_information": error_information,
        "error_str": None,
    }


def _route() -> RouteTrace:
    observed_at = date(2026, 9, 7)
    quote = PricingQuote(
        amount_usd=0.001,
        provider="openai",
        model="model-a",
        input_tokens=1_000,
        output_tokens=0,
        cached_input_tokens=0,
        input_per_million=1.0,
        output_per_million=1.0,
        cached_input_per_million=None,
        pricing_record_id=f"openai:model-a:{observed_at.isoformat()}",
        pricing_table_version="test-route-v1",
        pricing_observed_at=observed_at,
        pricing_source_url="https://pricing.example/model-a",
    )
    return RouteTrace(
        request_id="trace-1",
        strategy="policy",
        selected_model="model-a",
        estimated_cost_usd=quote.amount_usd,
        estimated_latency_ms=250,
        decision_reason="test",
        considered_models=["model-a"],
        fallback_models=["model-b"],
        max_estimated_cost_usd=None,
        budget_violation=False,
        budget_violation_reason=None,
        timestamp="2026-09-07T00:00:00+00:00",
        cost_evidence_complete=True,
        cost_quote=quote,
    )


def _record(ledger: SQLiteBenchmarkLedger, tmp_path, trace, route: RouteTrace | None = None) -> None:
    routes = [route] if route is not None else []
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="policy",
        provider=trace.provider,
        model=trace.model,
        ledger_path=tmp_path / "ledger.jsonl",
        traces=[trace],
        route_traces=routes,
    )
    ledger.record_run(run_id="run-1", report=report, traces=[trace], route_traces=routes)


def test_sqlite_round_trip_preserves_external_reported_cost_provenance(tmp_path) -> None:
    trace = import_litellm_standard_logging_chain([_payload()])
    ledger = SQLiteBenchmarkLedger(tmp_path / "ledger.sqlite3")
    _record(ledger, tmp_path, trace)

    restored = ledger.get_traces("run-1")[0]
    usage = ledger.get_provider_usage("run-1")[0]

    assert restored == trace
    assert usage.provider_attempts == trace.provider_attempts
    attempt = restored.provider_attempts[0]
    assert attempt.cost_evidence == CostEvidenceKind.REPORTED_BY_EXECUTION_STACK
    assert attempt.reported_cost_usd == pytest.approx(0.0012)
    assert attempt.calculated_cost_usd is None
    assert attempt.cost_source == LITELLM_STANDARD_LOGGING_SOURCE
    assert attempt.cost_source_record_id == "record-1"


def test_sqlite_round_trip_preserves_true_external_zero(tmp_path) -> None:
    trace = import_litellm_standard_logging_chain([_payload(response_cost=0.0)])
    ledger = SQLiteBenchmarkLedger(tmp_path / "ledger.sqlite3")
    _record(ledger, tmp_path, trace)

    restored = ledger.get_traces("run-1")[0]

    assert restored.cost_evidence_complete is True
    assert restored.estimated_cost_usd == 0.0
    assert restored.provider_attempts[0].reported_cost_usd == 0.0
    assert restored.provider_attempts[0].cost_is_known is True


def test_sqlite_round_trip_preserves_unknown_litellm_cache_state(tmp_path) -> None:
    trace = import_litellm_standard_logging_chain([_payload(cache_hit=None)])
    ledger = SQLiteBenchmarkLedger(tmp_path / "ledger.sqlite3")
    _record(ledger, tmp_path, trace)

    restored = ledger.get_traces("run-1")[0]
    usage = ledger.get_provider_usage("run-1")[0]

    assert restored.cache_hit is None
    assert usage.cache_hit is None


def test_sqlite_tampered_external_provenance_downgrades_to_unknown(tmp_path) -> None:
    trace = import_litellm_standard_logging_chain([_payload()])
    path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    _record(ledger, tmp_path, trace)

    with sqlite3.connect(path) as connection:
        raw_json = connection.execute(
            "SELECT provider_attempts_json FROM benchmark_traces WHERE run_id = 'run-1'"
        ).fetchone()[0]
        attempts = json.loads(raw_json)
        attempts[0]["cost_source_record_id"] = None
        tampered = json.dumps(attempts, sort_keys=True)
        for table in ("benchmark_traces", "benchmark_provider_usage"):
            connection.execute(
                f"""
                UPDATE {table}
                SET provider_attempts_json = ?, estimated_cost_usd = 0.0012,
                    cost_evidence_complete = 1
                WHERE run_id = 'run-1'
                """,
                (tampered,),
            )

    report = ledger.get_report("run-1")
    restored = ledger.get_traces("run-1")[0]
    usage = ledger.get_provider_usage("run-1")[0]

    assert report.cost_evidence_complete is False
    assert report.estimated_cost_usd is None
    assert restored.cost_evidence_complete is False
    assert restored.estimated_cost_usd is None
    assert usage.cost_evidence_complete is False
    assert usage.estimated_cost_usd is None
    for attempt in (restored.provider_attempts[0], usage.provider_attempts[0]):
        assert attempt.cost_evidence == CostEvidenceKind.UNKNOWN
        assert attempt.reported_cost_usd is None
        assert attempt.cost_source is None
        assert attempt.cost_source_record_id is None


def test_sqlite_restored_reported_cost_drives_retry_amplification_reconciliation(tmp_path) -> None:
    failed = _payload(
        record_id="failed-leg",
        call_id="call-a",
        provider="openai",
        model="model-a",
        status="failure",
        start_time=100.0,
        end_time=100.2,
        response_time=0.2,
        response_cost=0.0004,
        completion_tokens=0,
        total_tokens=100,
        error_information={
            "error_class": "RateLimitError",
            "error_message": "rate limited",
            "error_code": 429,
        },
    )
    succeeded = _payload(
        record_id="success-leg",
        call_id="call-b",
        provider="anthropic",
        model="model-b",
        start_time=100.3,
        end_time=100.7,
        response_time=0.4,
        response_cost=0.0016,
        prompt_tokens=105,
        completion_tokens=25,
        total_tokens=130,
    )
    trace = import_litellm_standard_logging_chain([failed, succeeded])
    route = _route()
    ledger = SQLiteBenchmarkLedger(tmp_path / "ledger.sqlite3")
    _record(ledger, tmp_path, trace, route)

    restored_trace = ledger.get_traces("run-1")[0]
    restored_route = ledger.get_routes("run-1")[0]
    result = reconcile_request_cost(route=restored_route, execution=restored_trace)

    assert result.status == ReconciliationStatus.COMPARABLE_SUCCESS
    assert result.observed_execution_cost_usd == pytest.approx(0.002)
    assert result.successful_final_attempt_cost_usd == pytest.approx(0.0016)
    assert result.non_final_attempt_cost_usd == pytest.approx(0.0004)
    assert result.retry_amplification_ratio == pytest.approx(1.25)
    assert result.execution_path == ("openai/model-a", "anthropic/model-b")
    assert result.execution_path_diverged is True


def test_provider_attempt_rejects_non_finite_external_amount() -> None:
    with pytest.raises(ValueError, match="reported_cost_usd must be finite"):
        ProviderAttempt(
            attempt_index=1,
            provider="openai",
            model="model-a",
            outcome=AttemptOutcome.SUCCEEDED,
            latency_ms=1,
            reported_cost_usd=float("nan"),
            cost_evidence=CostEvidenceKind.REPORTED_BY_EXECUTION_STACK,
            cost_source="litellm",
            cost_source_record_id="record-1",
        )


def test_sqlite_unknown_attempt_suppresses_stale_complete_aggregate(tmp_path) -> None:
    trace = import_litellm_standard_logging_chain([_payload()])
    path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    _record(ledger, tmp_path, trace)

    with sqlite3.connect(path) as connection:
        raw_json = connection.execute(
            "SELECT provider_attempts_json FROM benchmark_traces WHERE run_id = 'run-1'"
        ).fetchone()[0]
        attempts = json.loads(raw_json)
        attempts[0]["reported_cost_usd"] = None
        attempts[0]["cost_evidence"] = "unknown"
        attempts[0]["cost_source"] = None
        attempts[0]["cost_source_record_id"] = None
        unknown_json = json.dumps(attempts, sort_keys=True)
        for table in ("benchmark_traces", "benchmark_provider_usage"):
            connection.execute(
                f"""
                UPDATE {table}
                SET provider_attempts_json = ?, estimated_cost_usd = 0.0012,
                    cost_evidence_complete = 1
                WHERE run_id = 'run-1'
                """,
                (unknown_json,),
            )

    report = ledger.get_report("run-1")
    restored = ledger.get_traces("run-1")[0]
    usage = ledger.get_provider_usage("run-1")[0]

    assert report.cost_evidence_complete is False
    assert report.estimated_cost_usd is None
    assert restored.cost_evidence_complete is False
    assert restored.estimated_cost_usd is None
    assert restored.provider_attempts[0].cost_evidence == CostEvidenceKind.UNKNOWN
    assert usage.cost_evidence_complete is False
    assert usage.estimated_cost_usd is None
