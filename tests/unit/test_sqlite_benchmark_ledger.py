from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from inference_engine.benchmarking.harness import summarize_traces
from inference_engine.benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from inference_engine.domain.cost.pricing import PricingQuote
from inference_engine.domain.models.routing import (
    ModelConfig,
    ModelTier,
    RoutingDecision,
    RoutingStrategy,
)
from inference_engine.infrastructure.telemetry.request_log import RequestTrace, RouteTrace


def _trace(
    request_id: str = "request-1",
    *,
    model: str = "test-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    estimated_cost_usd: float = 0.001,
    provider_attempt_count: int = 1,
    provider_retry_count: int = 0,
    error_type: str | None = None,
    cost_evidence_complete: bool | None = None,
) -> RequestTrace:
    failed = error_type is not None
    ambiguous = failed or provider_retry_count > 0 or provider_attempt_count > 1
    complete = not ambiguous if cost_evidence_complete is None else cost_evidence_complete
    return RequestTrace(
        request_id=request_id,
        provider="openai",
        model=model,
        latency_ms=123,
        prompt_tokens=0 if failed else prompt_tokens,
        completion_tokens=0 if failed else completion_tokens,
        total_tokens=0 if failed else prompt_tokens + completion_tokens,
        estimated_cost_usd=estimated_cost_usd if complete else None,
        pricing_table_version="test",
        cache_hit=False,
        error_type=error_type,
        error_message="rate limited" if failed else None,
        timestamp="2026-01-01T00:00:00+00:00",
        provider_attempt_count=provider_attempt_count,
        provider_retry_count=provider_retry_count,
        cost_evidence_complete=complete,
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


def _route(request_id: str = "request-1") -> RouteTrace:
    quote = _route_quote()
    return RouteTrace(
        request_id=request_id,
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


def test_sqlite_benchmark_ledger_records_report_and_traces(tmp_path) -> None:
    ledger = SQLiteBenchmarkLedger(tmp_path / "benchmarks.sqlite3")
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

    stored_report = ledger.get_report("run-1")
    stored_traces = ledger.get_traces("run-1")
    stored_routes = ledger.get_routes("run-1")
    usage = ledger.get_provider_usage("run-1")
    usage_summary = ledger.get_provider_usage_summary("run-1")
    assert stored_report.request_count == 1
    assert stored_report.route_count == 1
    assert stored_report.estimated_cost_usd == pytest.approx(0.001)
    assert stored_traces == traces
    assert stored_routes == routes
    assert len(usage) == 1
    assert usage[0].request_id == "request-1"
    assert usage[0].estimated_cost_usd == pytest.approx(0.001)
    assert usage_summary.request_count == 1
    assert usage_summary.success_count == 1
    assert usage_summary.estimated_cost_usd == pytest.approx(0.001)
    assert usage_summary.provider_attempt_count == 1
    assert usage_summary.provider_retry_count == 0
    assert usage_summary.cost_by_model == {"test-model": pytest.approx(0.001)}
    assert usage_summary.tokens_by_model == {"test-model": 15}


def test_sqlite_route_cost_evidence_round_trips_reconstructable_quote(tmp_path) -> None:
    path = tmp_path / "benchmarks.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    trace = _trace()
    route = _route()
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

    stored = ledger.get_routes("run-1")[0]
    assert stored == route
    assert stored.cost_quote is not None
    assert stored.cost_quote.reconstructed_amount_usd == pytest.approx(stored.estimated_cost_usd)
    with sqlite3.connect(path) as connection:
        evidence = connection.execute(
            """
            SELECT input_tokens, output_tokens, input_per_million, output_per_million,
                   pricing_record_id, pricing_source_url
            FROM benchmark_route_cost_evidence
            WHERE run_id = 'run-1' AND request_id = 'request-1'
            """
        ).fetchone()
    assert evidence == (
        10,
        5,
        50.0,
        100.0,
        "openai:test-model:2026-09-03",
        "https://pricing.example/test-model",
    )


def test_sqlite_v6_legacy_route_without_pricing_evidence_reads_as_unknown(tmp_path) -> None:
    path = tmp_path / "benchmarks.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    trace = _trace()
    route = _route()
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
            "DELETE FROM benchmark_route_cost_evidence WHERE run_id = 'run-1'"
        )
        connection.execute(
            "UPDATE ledger_metadata SET value = '5' WHERE key = 'schema_version'"
        )
        raw_cost = connection.execute(
            "SELECT estimated_cost_usd FROM benchmark_routes WHERE run_id = 'run-1'"
        ).fetchone()

    stored = ledger.get_routes("run-1")[0]
    assert raw_cost == (0.001,)
    assert stored.estimated_cost_usd is None
    assert stored.cost_evidence_complete is False
    assert stored.cost_quote is None


def test_sqlite_rejects_new_route_write_without_reconstructable_quote(tmp_path) -> None:
    model = ModelConfig(
        id="test-model",
        name="Test Model",
        tier=ModelTier.STANDARD,
        max_context_length=4096,
    )
    decision = RoutingDecision(
        request_id=_uuid_for_test(),
        selected_model=model,
        strategy=RoutingStrategy.SINGLE_MODEL,
        complexity_estimate=None,
        estimated_cost=0.001,
        estimated_latency_ms=250,
        estimated_quality_score=0.7,
        decision_reason="legacy route",
    )
    legacy_route = RouteTrace.from_decision(decision)
    trace = _trace(request_id=str(decision.request_id))
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=[trace],
        route_traces=[legacy_route],
    )
    ledger = SQLiteBenchmarkLedger(tmp_path / "benchmarks.sqlite3")

    with pytest.raises(ValueError, match="complete, reconstructable pricing evidence"):
        ledger.record_run(
            run_id="run-1",
            report=report,
            traces=[trace],
            route_traces=[legacy_route],
        )


def _uuid_for_test():
    from uuid import UUID

    return UUID("00000000-0000-0000-0000-000000000001")


def test_sqlite_benchmark_ledger_replaces_existing_run(tmp_path) -> None:
    ledger = SQLiteBenchmarkLedger(tmp_path / "benchmarks.sqlite3")
    first_trace = _trace("request-1")
    second_trace = _trace("request-2")
    first_report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=[first_trace],
    )
    second_report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=[second_trace],
    )

    ledger.record_run(run_id="run-1", report=first_report, traces=[first_trace])
    ledger.record_run(run_id="run-1", report=second_report, traces=[second_trace])

    assert ledger.get_traces("run-1") == [second_trace]
    assert [usage.request_id for usage in ledger.get_provider_usage("run-1")] == ["request-2"]


def test_sqlite_benchmark_ledger_summarizes_unknown_cost_conservatively(tmp_path) -> None:
    ledger = SQLiteBenchmarkLedger(tmp_path / "benchmarks.sqlite3")
    traces = [
        _trace(
            "request-1",
            model="economy",
            prompt_tokens=10,
            completion_tokens=5,
            estimated_cost_usd=0.001,
        ),
        _trace(
            "request-2",
            model="premium",
            prompt_tokens=20,
            completion_tokens=10,
            estimated_cost_usd=0.006,
            provider_attempt_count=2,
            provider_retry_count=1,
        ),
        _trace(
            "request-3",
            model="premium",
            provider_attempt_count=2,
            provider_retry_count=1,
            error_type="rate_limit",
        ),
    ]
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="policy",
        provider="openai",
        model="economy,premium",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=traces,
    )

    ledger.record_run(run_id="run-1", report=report, traces=traces)

    summary = ledger.get_provider_usage_summary("run-1")
    assert summary.request_count == 3
    assert summary.success_count == 2
    assert summary.failure_count == 1
    assert summary.prompt_tokens == 30
    assert summary.completion_tokens == 15
    assert summary.total_tokens == 45
    assert summary.estimated_cost_usd is None
    assert summary.cost_evidence_complete is False
    assert summary.provider_attempt_count == 5
    assert summary.provider_retry_count == 2
    assert summary.cost_by_model == {
        "economy": pytest.approx(0.001),
        "premium": None,
    }
    assert summary.tokens_by_model == {"economy": 15, "premium": 30}


def test_sqlite_unknown_cost_is_stored_as_sql_null(tmp_path) -> None:
    path = tmp_path / "benchmarks.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    trace = _trace(
        "request-1",
        provider_attempt_count=2,
        provider_retry_count=1,
    )
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=[trace],
    )

    ledger.record_run(run_id="run-1", report=report, traces=[trace])

    with sqlite3.connect(path) as connection:
        trace_cost = connection.execute(
            "SELECT estimated_cost_usd FROM benchmark_traces WHERE run_id = 'run-1'"
        ).fetchone()
        usage_cost = connection.execute(
            "SELECT estimated_cost_usd FROM benchmark_provider_usage WHERE run_id = 'run-1'"
        ).fetchone()
    assert trace_cost == (None,)
    assert usage_cost == (None,)


def test_sqlite_benchmark_ledger_backfills_usage_for_existing_trace_table(tmp_path) -> None:
    path = tmp_path / "benchmarks.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE benchmark_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                workload_path TEXT NOT NULL,
                strategy TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE benchmark_traces (
                run_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                estimated_cost_usd REAL NOT NULL,
                pricing_table_version TEXT NOT NULL,
                cache_hit INTEGER NOT NULL,
                error_type TEXT,
                error_message TEXT,
                quality_passed INTEGER,
                quality_score REAL,
                quality_reason TEXT,
                eval_type TEXT,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (run_id, request_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO benchmark_runs (
                run_id, created_at, workload_path, strategy, provider, model, report_json
            )
            VALUES ('run-1', '2026-01-01T00:00:00+00:00', 'workload.jsonl',
                    'single_model', 'openai', 'test-model', '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO benchmark_traces (
                run_id,
                request_id,
                provider,
                model,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                estimated_cost_usd,
                pricing_table_version,
                cache_hit,
                error_type,
                error_message,
                quality_passed,
                quality_score,
                quality_reason,
                eval_type,
                timestamp
            )
            VALUES ('run-1', 'request-1', 'openai', 'test-model', 100, 10, 5, 15,
                    0.001, 'test', 0, NULL, NULL, NULL, NULL, NULL, NULL,
                    '2026-01-01T00:00:00+00:00')
            """
        )

    ledger = SQLiteBenchmarkLedger(path)

    usage = ledger.get_provider_usage("run-1")

    assert len(usage) == 1
    assert usage[0].request_id == "request-1"
    assert usage[0].provider_attempt_count == 1
    assert usage[0].provider_retry_count == 0
    assert usage[0].estimated_cost_usd == pytest.approx(0.001)

    with sqlite3.connect(path) as connection:
        notnull = connection.execute("PRAGMA table_info(benchmark_traces)").fetchall()[8][3]
    assert notnull == 0


def test_sqlite_migration_downgrades_legacy_failed_zero_cost_to_null(tmp_path) -> None:
    path = tmp_path / "benchmarks.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE benchmark_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                workload_path TEXT NOT NULL,
                strategy TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE benchmark_traces (
                run_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                estimated_cost_usd REAL NOT NULL,
                pricing_table_version TEXT NOT NULL,
                cache_hit INTEGER NOT NULL,
                error_type TEXT,
                error_message TEXT,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (run_id, request_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO benchmark_runs (
                run_id, created_at, workload_path, strategy, provider, model, report_json
            )
            VALUES ('run-1', '2026-01-01T00:00:00+00:00', 'workload.jsonl',
                    'single_model', 'openai', 'test-model', '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO benchmark_traces (
                run_id, request_id, provider, model, latency_ms, prompt_tokens,
                completion_tokens, total_tokens, estimated_cost_usd, pricing_table_version,
                cache_hit, error_type, error_message, timestamp
            )
            VALUES ('run-1', 'request-1', 'openai', 'test-model', 100, 0,
                    0, 0, 0.0, 'legacy', 0, 'rate_limit', 'rate limited',
                    '2026-01-01T00:00:00+00:00')
            """
        )

    ledger = SQLiteBenchmarkLedger(path)
    trace = ledger.get_traces("run-1")[0]

    assert trace.cost_evidence_complete is False
    assert trace.estimated_cost_usd is None
    with sqlite3.connect(path) as connection:
        raw_cost = connection.execute(
            "SELECT estimated_cost_usd FROM benchmark_traces WHERE run_id = 'run-1'"
        ).fetchone()
    assert raw_cost == (None,)


def test_sqlite_benchmark_ledger_unknown_run_fails(tmp_path) -> None:
    ledger = SQLiteBenchmarkLedger(tmp_path / "benchmarks.sqlite3")

    with pytest.raises(KeyError, match="Unknown benchmark run_id"):
        ledger.get_report("missing")
