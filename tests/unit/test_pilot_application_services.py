from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_engine.application.services.benchmarking.migration_decision_service import (
    _replace_cost_classification_rule,
)
from inference_engine.application.services.benchmarking.run_evidence_service import (
    persist_benchmark_run,
)
from inference_engine.benchmarking.context_store import SQLiteBenchmarkContextStore
from inference_engine.benchmarking.segmentation import BenchmarkRequestContext
from inference_engine.benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from inference_engine.domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
    ProviderAttempt,
)
from inference_engine.infrastructure.telemetry.request_log import RequestTrace


def _trace(
    request_id: str,
    *,
    provider: str = "openai",
    model: str = "test-model",
    cost_kind: CostEvidenceKind = CostEvidenceKind.CALCULATED_FROM_USAGE,
    cost_source: str | None = None,
) -> RequestTrace:
    is_calculated = cost_kind == CostEvidenceKind.CALCULATED_FROM_USAGE
    is_reported = cost_kind == CostEvidenceKind.REPORTED_BY_EXECUTION_STACK
    pricing_date = "2026-09-13"
    attempt = ProviderAttempt(
        attempt_index=1,
        provider=provider,
        model=model,
        outcome=AttemptOutcome.SUCCEEDED,
        latency_ms=100,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        calculated_cost_usd=0.01 if is_calculated else None,
        reported_cost_usd=0.01 if is_reported else None,
        cost_evidence=cost_kind,
        pricing_table_version="test-pricing" if is_calculated else None,
        pricing_record_id=f"{provider}:{model}:{pricing_date}" if is_calculated else None,
        pricing_observed_at=pricing_date if is_calculated else None,
        pricing_source_url="https://example.test/pricing" if is_calculated else None,
        cost_source=cost_source if is_reported else None,
        cost_source_record_id="source-1" if is_reported else None,
    )
    return RequestTrace(
        request_id=request_id,
        provider=provider,
        model=model,
        latency_ms=100,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.01,
        pricing_table_version=(
            "test-pricing" if is_calculated else "external_reported:test"
        ),
        cache_hit=False,
        error_type=None,
        error_message=None,
        timestamp="2026-09-13T00:00:00+00:00",
        quality_passed=True,
        quality_score=1.0,
        quality_reason="fixture",
        eval_type="exact_match",
        provider_attempt_count=1,
        provider_retry_count=0,
        cost_evidence_complete=True,
        provider_attempts=(attempt,),
    )


def test_persist_benchmark_run_is_single_report_segment_and_storage_owner(tmp_path: Path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        '{"id":"item-1","prompt":"hello","tags":{"segment":"critical"}}\n',
        encoding="utf-8",
    )
    trace = _trace("request-1")
    context = BenchmarkRequestContext.from_tags(
        request_id="request-1",
        workload_item_id="item-1",
        tags={"segment": "critical"},
    )
    sqlite_path = tmp_path / "ledger.sqlite3"
    report_path = tmp_path / "report.json"
    segment_path = tmp_path / "segments.json"

    persisted = persist_benchmark_run(
        run_id="baseline",
        workload_path=workload_path,
        strategy="controlled_replay",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "requests.jsonl",
        sqlite_ledger_path=sqlite_path,
        traces=[trace],
        request_contexts=[context],
        report_path=report_path,
        segment_report_path=segment_path,
        additional_limitations=("pilot fixture",),
    )

    stored_report = SQLiteBenchmarkLedger(sqlite_path).get_report("baseline")
    stored_contexts = SQLiteBenchmarkContextStore(sqlite_path).get_contexts("baseline")
    assert persisted.report == stored_report
    assert persisted.segments.segment_count == 1
    assert stored_contexts == [context]
    assert "pilot fixture" in persisted.report.limitations
    assert json.loads(report_path.read_text(encoding="utf-8"))["provider"] == "openai"
    assert json.loads(segment_path.read_text(encoding="utf-8"))["segment_count"] == 1


def test_persist_benchmark_run_rejects_context_trace_drift(tmp_path: Path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text('{"id":"item-1","prompt":"hello"}\n', encoding="utf-8")
    context = BenchmarkRequestContext.from_tags(
        request_id="different-request",
        workload_item_id="item-1",
        tags={},
    )

    with pytest.raises(ValueError, match="exact request-context/trace coverage"):
        persist_benchmark_run(
            run_id="baseline",
            workload_path=workload_path,
            strategy="controlled_replay",
            provider="openai",
            model="test-model",
            ledger_path=tmp_path / "requests.jsonl",
            sqlite_ledger_path=tmp_path / "ledger.sqlite3",
            traces=[_trace("request-1")],
            request_contexts=[context],
        )


def test_authoritative_decision_cost_rule_uses_observed_attempt_metadata() -> None:
    payload: dict[str, object] = {
        "cost_evidence": {"classification_rule": "legacy LiteLLM-specific rule"}
    }
    baseline = _trace("baseline", provider="openai")
    candidate = _trace(
        "candidate",
        provider="ollama",
        cost_kind=CostEvidenceKind.REPORTED_BY_EXECUTION_STACK,
        cost_source="litellm_standard_logging_payload",
    )

    _replace_cost_classification_rule(
        payload,
        baseline_traces=[baseline],
        candidate_traces=[candidate],
    )

    cost_evidence = payload["cost_evidence"]
    assert isinstance(cost_evidence, dict)
    rule = str(cost_evidence["classification_rule"])
    assert CostEvidenceKind.CALCULATED_FROM_USAGE.value in rule
    assert CostEvidenceKind.REPORTED_BY_EXECUTION_STACK.value in rule
    assert "litellm_standard_logging_payload" in rule
    assert "final-invoice truth" in rule
    assert "legacy LiteLLM-specific rule" not in rule
