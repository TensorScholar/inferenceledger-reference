from __future__ import annotations

import sqlite3

import pytest

from inference_engine.benchmarking.context_store import (
    CONTEXT_SCHEMA_VERSION,
    SQLiteBenchmarkContextStore,
)
from inference_engine.benchmarking.harness import summarize_traces
from inference_engine.benchmarking.segmentation import BenchmarkRequestContext
from inference_engine.benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from inference_engine.infrastructure.telemetry.request_log import RequestTrace


def _trace(request_id: str) -> RequestTrace:
    return RequestTrace(
        request_id=request_id,
        provider="openai",
        model="model-a",
        latency_ms=10,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.001,
        pricing_table_version="test-v1",
        cache_hit=False,
        error_type=None,
        error_message=None,
        timestamp="2026-09-03T00:00:00+00:00",
        provider_attempt_count=1,
        provider_retry_count=0,
        cost_evidence_complete=True,
    )


def _record_run(
    ledger: SQLiteBenchmarkLedger,
    tmp_path,
    *,
    run_id: str,
    traces: list[RequestTrace],
) -> None:
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="model-a",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=traces,
    )
    ledger.record_run(run_id=run_id, report=report, traces=traces)


def test_context_store_round_trips_normalized_workload_identity_and_tags(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    _record_run(ledger, tmp_path, run_id="run-1", traces=[_trace("request-1")])
    store = SQLiteBenchmarkContextStore(path)
    context = BenchmarkRequestContext.from_tags(
        request_id="request-1",
        workload_item_id="item-1",
        tags={"risk": "high", "task": "code"},
    )

    store.record_contexts(run_id="run-1", contexts=[context])

    restored = store.get_contexts("run-1")
    assert restored == [context]
    assert restored[0].tags == (("risk", "high"), ("task", "code"))
    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT value FROM ledger_metadata WHERE key = 'benchmark_context_schema_version'"
        ).fetchone()
    assert version is not None
    assert version[0] == str(CONTEXT_SCHEMA_VERSION)


def test_context_store_requires_exact_trace_coverage(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    _record_run(
        ledger,
        tmp_path,
        run_id="run-1",
        traces=[_trace("request-1"), _trace("request-2")],
    )
    store = SQLiteBenchmarkContextStore(path)

    with pytest.raises(ValueError, match="cover exactly"):
        store.record_contexts(
            run_id="run-1",
            contexts=[
                BenchmarkRequestContext.from_tags(
                    request_id="request-1",
                    workload_item_id="item-1",
                    tags={"task": "qa"},
                )
            ],
        )


def test_context_store_rejects_duplicate_workload_item_identity(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    _record_run(
        ledger,
        tmp_path,
        run_id="run-1",
        traces=[_trace("request-1"), _trace("request-2")],
    )
    store = SQLiteBenchmarkContextStore(path)

    with pytest.raises(ValueError, match="duplicate benchmark workload_item_id"):
        store.record_contexts(
            run_id="run-1",
            contexts=[
                BenchmarkRequestContext.from_tags(
                    request_id="request-1",
                    workload_item_id="item-1",
                    tags={"task": "qa"},
                ),
                BenchmarkRequestContext.from_tags(
                    request_id="request-2",
                    workload_item_id="item-1",
                    tags={"task": "qa"},
                ),
            ],
        )


def test_legacy_run_without_context_remains_readable_as_unsegmented(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    _record_run(ledger, tmp_path, run_id="legacy", traces=[_trace("request-1")])

    assert SQLiteBenchmarkContextStore(path).get_contexts("legacy") == []


def test_replacing_run_cascades_stale_context_rows(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(path)
    _record_run(ledger, tmp_path, run_id="run-1", traces=[_trace("request-1")])
    store = SQLiteBenchmarkContextStore(path)
    store.record_contexts(
        run_id="run-1",
        contexts=[
            BenchmarkRequestContext.from_tags(
                request_id="request-1",
                workload_item_id="item-1",
                tags={"task": "qa"},
            )
        ],
    )

    _record_run(ledger, tmp_path, run_id="run-1", traces=[_trace("request-2")])

    assert store.get_contexts("run-1") == []
