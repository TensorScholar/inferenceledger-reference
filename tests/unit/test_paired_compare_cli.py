from __future__ import annotations

import argparse
import json

import scripts.compare_paired as paired_cli
from inference_engine.benchmarking.context_store import SQLiteBenchmarkContextStore
from inference_engine.benchmarking.harness import BenchmarkReport
from inference_engine.benchmarking.segmentation import BenchmarkRequestContext
from inference_engine.benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from inference_engine.infrastructure.telemetry.request_log import RequestTrace


def _report(*, provider: str) -> BenchmarkReport:
    return BenchmarkReport(
        workload_path="benchmarks/workloads/paired-test.jsonl",
        workload_sha256="paired-test-sha",
        strategy="policy",
        provider=provider,
        model="profile",
        request_count=5,
        success_count=5,
        failure_count=0,
        error_rate=0.0,
        latency_p50_ms=100,
        latency_p95_ms=120,
        prompt_tokens=50,
        completion_tokens=25,
        total_tokens=75,
        estimated_cost_usd=0.005,
        cost_evidence_complete=True,
        provider_attempt_count=5,
        provider_retry_count=0,
        route_count=0,
        budget_violation_count=0,
        model_distribution={"model": 5},
        route_reason_distribution={},
        observed_latency_ms_by_model={"model": {"count": 5, "p50": 100, "p95": 120}},
        quality_count=5,
        quality_pass_count=5,
        quality_pass_rate=1.0,
        quality_score_avg=1.0,
        ledger_path="ledger.jsonl",
        limitations=[],
    )


def _trace(*, request_id: str, latency_ms: int, cost: float) -> RequestTrace:
    return RequestTrace(
        request_id=request_id,
        provider="provider",
        model="model",
        latency_ms=latency_ms,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost_usd=cost,
        pricing_table_version="test-v1",
        cache_hit=False,
        error_type=None,
        error_message=None,
        timestamp="2026-09-04T00:00:00+00:00",
        quality_passed=True,
        quality_score=1.0,
        provider_attempt_count=1,
        provider_retry_count=0,
        cost_evidence_complete=True,
    )


def _contexts(prefix: str) -> list[BenchmarkRequestContext]:
    return [
        BenchmarkRequestContext.from_tags(
            request_id=f"{prefix}-{index}",
            workload_item_id=f"item-{index}",
            tags={"task": "qa"},
        )
        for index in range(5)
    ]


def test_stored_paired_cli_reconstructs_json_without_provider_execution(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(ledger_path)
    context_store = SQLiteBenchmarkContextStore(ledger_path)

    baseline_contexts = _contexts("baseline")
    candidate_contexts = _contexts("candidate")
    baseline_traces = [
        _trace(
            request_id=context.request_id,
            latency_ms=100 + index * 3,
            cost=0.001 + index * 0.00001,
        )
        for index, context in enumerate(baseline_contexts)
    ]
    candidate_traces = [
        _trace(
            request_id=context.request_id,
            latency_ms=95 + index * 2,
            cost=0.0009 + index * 0.000012,
        )
        for index, context in enumerate(candidate_contexts)
    ]

    ledger.record_run(
        run_id="baseline",
        report=_report(provider="openai"),
        traces=baseline_traces,
    )
    context_store.record_contexts(run_id="baseline", contexts=baseline_contexts)
    ledger.record_run(
        run_id="candidate",
        report=_report(provider="other-provider"),
        traces=candidate_traces,
    )
    context_store.record_contexts(run_id="candidate", contexts=candidate_contexts)

    output_path = tmp_path / "paired.json"
    exit_code = paired_cli._run(
        argparse.Namespace(
            sqlite_ledger_path=str(ledger_path),
            baseline_run_id="baseline",
            candidate_run_id="candidate",
            output_path=str(output_path),
            confidence_level=0.95,
            bootstrap_iterations=1_000,
            minimum_samples=5,
            seed=13,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["available"] is True
    assert payload["baseline_provider"] == "openai"
    assert payload["candidate_provider"] == "other-provider"
    assert payload["workload_item_count"] == 5
    assert payload["execution_cost_usd"]["eligible_pair_count"] == 5
    assert payload["execution_cost_usd"]["estimate"]["status"] == "sufficient"
    assert payload["successful_latency_ms"]["estimate"]["status"] == "sufficient"
    assert payload["tail_latency_inference_supported"] is False
    assert len(payload["segments"]) == 1
