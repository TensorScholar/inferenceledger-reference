from __future__ import annotations

import json
import sqlite3

import pytest

from inference_engine.benchmarking.harness import summarize_traces
from inference_engine.benchmarking.sqlite_ledger import SCHEMA_VERSION, SQLiteBenchmarkLedger
from inference_engine.domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
    ProviderAttempt,
)
from inference_engine.infrastructure.telemetry.request_log import RequestTrace


def _priced_attempt() -> ProviderAttempt:
    return ProviderAttempt(
        attempt_index=1,
        provider="test-provider",
        model="test-model",
        outcome=AttemptOutcome.SUCCEEDED,
        latency_ms=25,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cached_tokens=0,
        calculated_cost_usd=0.001,
        cost_evidence=CostEvidenceKind.CALCULATED_FROM_USAGE,
        pricing_table_version="test",
        pricing_record_id="test-provider:test-model:2026-09-03",
        pricing_observed_at="2026-09-03",
        pricing_source_url="https://pricing.example/test-model",
    )


def _trace() -> RequestTrace:
    attempt = _priced_attempt()
    return RequestTrace(
        request_id="request-1",
        provider="test-provider",
        model="test-model",
        latency_ms=25,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.001,
        cost_evidence_complete=True,
        pricing_table_version="test",
        cache_hit=False,
        error_type=None,
        error_message=None,
        timestamp="2026-09-03T00:00:00+00:00",
        provider_attempts=(attempt,),
    )


def _record_run(ledger: SQLiteBenchmarkLedger, tmp_path) -> None:
    trace = _trace()
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="test-provider",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=[trace],
    )
    ledger.record_run(run_id="run-1", report=report, traces=[trace])


def test_sqlite_round_trip_preserves_attempt_pricing_provenance(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    _record_run(ledger, tmp_path)

    trace = ledger.get_traces("run-1")[0]
    usage = ledger.get_provider_usage("run-1")[0]

    assert trace.provider_attempts == (_priced_attempt(),)
    assert usage.provider_attempts == (_priced_attempt(),)
    assert trace.estimated_cost_usd == pytest.approx(0.001)
    assert trace.cost_evidence_complete is True
    with sqlite3.connect(path) as connection:
        schema_version = connection.execute(
            "SELECT value FROM ledger_metadata WHERE key = 'schema_version'"
        ).fetchone()
    assert schema_version == (str(SCHEMA_VERSION),)
    assert SCHEMA_VERSION == 7


def test_sqlite_migration_invalidates_pre_provenance_calculated_attempt(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    _record_run(ledger, tmp_path)

    legacy_attempt = {
        "attempt_index": 1,
        "provider": "test-provider",
        "model": "test-model",
        "outcome": "succeeded",
        "latency_ms": 25,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cached_tokens": 0,
        "calculated_cost_usd": 0.001,
        "cost_evidence": "calculated_from_usage",
        "pricing_table_version": "pr4-legacy",
        "error_type": None,
        "status_code": None,
    }
    legacy_json = json.dumps([legacy_attempt], sort_keys=True)
    with sqlite3.connect(path) as connection:
        for table in ("benchmark_traces", "benchmark_provider_usage"):
            connection.execute(
                f"""
                UPDATE {table}
                SET provider_attempts_json = ?,
                    estimated_cost_usd = 0.001,
                    cost_evidence_complete = 1,
                    pricing_table_version = 'pr4-legacy'
                WHERE run_id = 'run-1'
                """,
                (legacy_json,),
            )
        connection.execute(
            "UPDATE ledger_metadata SET value = '4' WHERE key = 'schema_version'"
        )

    stored_report = ledger.get_report("run-1")
    stored_trace = ledger.get_traces("run-1")[0]

    assert stored_report.estimated_cost_usd is None
    assert stored_report.cost_evidence_complete is False
    assert stored_trace.estimated_cost_usd is None
    assert stored_trace.cost_evidence_complete is False
    assert stored_trace.provider_attempts[0].cost_evidence == CostEvidenceKind.UNKNOWN
    assert stored_trace.provider_attempts[0].calculated_cost_usd is None
    assert stored_trace.provider_attempts[0].pricing_table_version is None
    assert stored_trace.provider_attempts[0].pricing_record_id is None
    assert stored_trace.provider_attempts[0].pricing_source_url is None

    with sqlite3.connect(path) as connection:
        for table in ("benchmark_traces", "benchmark_provider_usage"):
            row = connection.execute(
                f"""
                SELECT estimated_cost_usd, cost_evidence_complete, provider_attempts_json
                FROM {table}
                WHERE run_id = 'run-1'
                """
            ).fetchone()
            assert row is not None
            assert row[0] is None
            assert row[1] == 0
            raw_attempt = json.loads(row[2])[0]
            assert raw_attempt["calculated_cost_usd"] is None
            assert raw_attempt["cost_evidence"] == "unknown"
            assert raw_attempt["pricing_table_version"] is None
