from __future__ import annotations

import json
from argparse import Namespace

import pytest

from inference_engine.benchmarking.harness import (
    compare_reports,
    load_workload,
    summarize_traces,
    write_comparison,
    write_report,
)
from inference_engine.infrastructure.telemetry.request_log import RequestTrace
from scripts.run_benchmark import _compare


def _trace(
    request_id: str,
    *,
    model: str = "test-model",
    latency_ms: int = 100,
    estimated_cost_usd: float = 0.01,
    quality_passed: bool | None = True,
    quality_score: float | None = 1.0,
    error_type: str | None = None,
    provider_attempt_count: int = 1,
    provider_retry_count: int = 0,
    cost_evidence_complete: bool | None = None,
) -> RequestTrace:
    failed = error_type is not None
    complete = not failed if cost_evidence_complete is None else cost_evidence_complete
    return RequestTrace(
        request_id=request_id,
        provider="openai",
        model=model,
        latency_ms=latency_ms,
        prompt_tokens=0 if failed else 10,
        completion_tokens=0 if failed else 5,
        total_tokens=0 if failed else 15,
        estimated_cost_usd=estimated_cost_usd if complete else None,
        cost_evidence_complete=complete,
        pricing_table_version="test",
        cache_hit=False,
        error_type=error_type,
        error_message="rate limited" if failed else None,
        timestamp=f"2026-01-01T00:00:0{request_id}+00:00",
        quality_passed=None if failed else quality_passed,
        quality_score=None if failed else quality_score,
        quality_reason=None if failed else "passed",
        eval_type=None if failed else "exact_match",
        provider_attempt_count=provider_attempt_count,
        provider_retry_count=provider_retry_count,
    )


def test_load_workload_parses_jsonl(tmp_path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        '{"id":"one","prompt":"hello","tags":{"task":"smoke"},'
        '"eval":{"type":"exact_match","expected":"ok"}}\n',
        encoding="utf-8",
    )

    workload = load_workload(workload_path)

    assert len(workload) == 1
    assert workload[0].id == "one"
    assert workload[0].prompt == "hello"
    assert workload[0].tags == {"task": "smoke"}
    assert workload[0].eval_spec is not None


def test_load_workload_rejects_empty_prompt(tmp_path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text('{"id":"one","prompt":""}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing string prompt"):
        load_workload(workload_path)


def test_load_workload_rejects_invalid_eval(tmp_path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        '{"id":"one","prompt":"hello","eval":{"type":"contains_all","required":"cost"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid eval"):
        load_workload(workload_path)


def test_summarize_traces_blocks_cost_when_failed_attempt_cost_is_unknown(tmp_path) -> None:
    traces = [
        _trace("1", latency_ms=100, estimated_cost_usd=0.01),
        _trace(
            "2",
            latency_ms=300,
            error_type="rate_limit",
            provider_attempt_count=2,
            provider_retry_count=1,
        ),
    ]

    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=traces,
    )

    assert report.request_count == 2
    assert report.success_count == 1
    assert report.failure_count == 1
    assert report.error_rate == 0.5
    assert report.latency_p50_ms == 100
    assert report.latency_p95_ms == 300
    assert report.prompt_tokens == 10
    assert report.completion_tokens == 5
    assert report.estimated_cost_usd is None
    assert report.cost_evidence_complete is False
    assert report.provider_attempt_count == 3
    assert report.provider_retry_count == 1
    assert report.model_distribution == {"test-model": 1}
    assert report.observed_latency_ms_by_model == {"test-model": {"count": 1, "p50": 100, "p95": 100}}
    assert report.route_reason_distribution == {}
    assert report.quality_count == 1
    assert report.quality_pass_count == 1
    assert report.quality_pass_rate == 1.0
    assert report.quality_score_avg == 1.0
    assert any("unknown cost" in limitation for limitation in report.limitations)


def test_summarize_traces_does_not_claim_pricing_table_for_execution_stack_cost(tmp_path) -> None:
    from inference_engine.domain.models.execution import (
        AttemptOutcome,
        CostEvidenceKind,
        ProviderAttempt,
    )

    traces = [
        RequestTrace(
            request_id="ext-1",
            provider="ollama",
            model="gemma3:270m",
            latency_ms=100,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated_cost_usd=0.0,
            pricing_table_version="external_reported:litellm_standard_logging_payload",
            cache_hit=None,
            error_type=None,
            error_message=None,
            timestamp="2026-09-11T00:00:00+00:00",
            cost_evidence_complete=True,
            provider_attempts=(
                ProviderAttempt(
                    attempt_index=1,
                    provider="ollama",
                    model="gemma3:270m",
                    outcome=AttemptOutcome.SUCCEEDED,
                    latency_ms=100,
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    reported_cost_usd=0.0,
                    cost_evidence=CostEvidenceKind.REPORTED_BY_EXECUTION_STACK,
                    cost_source="litellm_standard_logging_payload",
                    cost_source_record_id="record-1",
                ),
            ),
        )
    ]

    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="litellm_baseline",
        provider="ollama",
        model="gemma3:270m",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=traces,
    )

    assert report.estimated_cost_usd == 0.0
    assert any("execution stack" in item for item in report.limitations)
    assert not any("pricing table" in item for item in report.limitations)


def test_summarize_empty_run_has_known_zero_cost(tmp_path) -> None:
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=[],
    )

    assert report.estimated_cost_usd == 0.0
    assert report.cost_evidence_complete is True


def test_write_report_outputs_json(tmp_path) -> None:
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=[],
    )
    report_path = tmp_path / "report.json"

    write_report(report, report_path)

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    assert raw["request_count"] == 0
    assert raw["cost_evidence_complete"] is True
    assert raw["limitations"]


def test_compare_reports_marks_matching_runs_comparable(tmp_path) -> None:
    baseline = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="baseline-model",
        ledger_path=tmp_path / "baseline.jsonl",
        traces=[_trace("1", model="baseline-model", latency_ms=200, estimated_cost_usd=0.04)],
    )
    candidate = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="rule_based",
        provider="openai",
        model="candidate-model",
        ledger_path=tmp_path / "candidate.jsonl",
        traces=[_trace("2", model="candidate-model", latency_ms=150, estimated_cost_usd=0.02)],
    )

    comparison = compare_reports(
        baseline_run_id="baseline",
        baseline=baseline,
        candidate_run_id="candidate",
        candidate=candidate,
    )

    assert comparison.comparable is True
    assert comparison.cost_evidence_complete is True
    assert comparison.cost_delta_usd == pytest.approx(-0.02)
    assert comparison.cost_delta_percent == pytest.approx(-50.0)
    assert comparison.latency_p95_delta_ms == -50
    assert comparison.quality_pass_rate_delta == 0.0


def test_compare_reports_rejects_incomplete_candidate_cost_evidence(tmp_path) -> None:
    baseline = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="baseline-model",
        ledger_path=tmp_path / "baseline.jsonl",
        traces=[_trace("1", model="baseline-model", estimated_cost_usd=0.04)],
    )
    candidate = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="rule_based",
        provider="openai",
        model="candidate-model",
        ledger_path=tmp_path / "candidate.jsonl",
        traces=[
            _trace(
                "2",
                model="candidate-model",
                estimated_cost_usd=0.02,
                provider_attempt_count=2,
                provider_retry_count=1,
                cost_evidence_complete=False,
            )
        ],
    )

    comparison = compare_reports(
        baseline_run_id="baseline",
        baseline=baseline,
        candidate_run_id="candidate",
        candidate=candidate,
    )

    assert comparison.comparable is False
    assert comparison.cost_evidence_complete is False
    assert comparison.candidate_cost_usd is None
    assert comparison.cost_delta_usd is None
    assert comparison.cost_delta_percent is None
    assert any("incomplete provider-attempt cost evidence" in item for item in comparison.limitations)


def test_compare_reports_rejects_quality_regression(tmp_path) -> None:
    baseline = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="baseline-model",
        ledger_path=tmp_path / "baseline.jsonl",
        traces=[_trace("1", model="baseline-model", quality_passed=True, quality_score=1.0)],
    )
    candidate = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="rule_based",
        provider="openai",
        model="candidate-model",
        ledger_path=tmp_path / "candidate.jsonl",
        traces=[_trace("2", model="candidate-model", quality_passed=False, quality_score=0.0)],
    )

    comparison = compare_reports(
        baseline_run_id="baseline",
        baseline=baseline,
        candidate_run_id="candidate",
        candidate=candidate,
    )

    assert comparison.comparable is False
    assert any("quality pass rate" in limitation for limitation in comparison.limitations)


def test_compare_reports_rejects_different_request_counts(tmp_path) -> None:
    baseline = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="baseline-model",
        ledger_path=tmp_path / "baseline.jsonl",
        traces=[_trace("1", model="baseline-model")],
    )
    candidate = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="rule_based",
        provider="openai",
        model="candidate-model",
        ledger_path=tmp_path / "candidate.jsonl",
        traces=[],
    )

    comparison = compare_reports(
        baseline_run_id="baseline",
        baseline=baseline,
        candidate_run_id="candidate",
        candidate=candidate,
    )

    assert comparison.comparable is False
    assert any("request counts" in limitation for limitation in comparison.limitations)


def test_write_comparison_outputs_json(tmp_path) -> None:
    report = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="test-model",
        ledger_path=tmp_path / "ledger.jsonl",
        traces=[_trace("1")],
    )
    comparison = compare_reports(
        baseline_run_id="baseline",
        baseline=report,
        candidate_run_id="candidate",
        candidate=report,
    )
    comparison_path = tmp_path / "comparison.json"

    write_comparison(comparison, comparison_path)

    raw = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert raw["baseline_run_id"] == "baseline"
    assert raw["candidate_run_id"] == "candidate"
    assert raw["cost_evidence_complete"] is True
    assert raw["limitations"]


def test_compare_cli_writes_comparison_from_sqlite_ledger(tmp_path) -> None:
    from inference_engine.benchmarking.sqlite_ledger import SQLiteBenchmarkLedger

    ledger = SQLiteBenchmarkLedger(tmp_path / "ledger.sqlite3")
    baseline_trace = _trace("1", model="baseline-model", latency_ms=200, estimated_cost_usd=0.04)
    candidate_trace = _trace("2", model="candidate-model", latency_ms=100, estimated_cost_usd=0.02)
    baseline = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="single_model",
        provider="openai",
        model="baseline-model",
        ledger_path=tmp_path / "baseline.jsonl",
        traces=[baseline_trace],
    )
    candidate = summarize_traces(
        workload_path=tmp_path / "workload.jsonl",
        strategy="rule_based",
        provider="openai",
        model="candidate-model",
        ledger_path=tmp_path / "candidate.jsonl",
        traces=[candidate_trace],
    )
    ledger.record_run(run_id="baseline", report=baseline, traces=[baseline_trace])
    ledger.record_run(run_id="candidate", report=candidate, traces=[candidate_trace])
    comparison_path = tmp_path / "comparison.json"

    exit_code = _compare(
        Namespace(
            sqlite_ledger_path=str(tmp_path / "ledger.sqlite3"),
            baseline_run_id="baseline",
            candidate_run_id="candidate",
            comparison_path=str(comparison_path),
        )
    )

    raw = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert raw["comparable"] is True
    assert raw["cost_delta_usd"] == pytest.approx(-0.02)
