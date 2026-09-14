from __future__ import annotations

from dataclasses import replace

import pytest

from inference_engine.benchmarking.harness import BenchmarkReport
from inference_engine.benchmarking.paired_comparison import compare_paired_runs
from inference_engine.benchmarking.segmentation import BenchmarkRequestContext
from inference_engine.benchmarking.statistics import (
    PairedBootstrapConfig,
    StatisticalEvidenceStatus,
)
from inference_engine.infrastructure.telemetry.request_log import RequestTrace


def _report(*, request_count: int, provider: str, workload_sha256: str = "workload-sha") -> BenchmarkReport:
    return BenchmarkReport(
        workload_path="benchmarks/workloads/test.jsonl",
        workload_sha256=workload_sha256,
        strategy="policy",
        provider=provider,
        model="profile",
        request_count=request_count,
        success_count=request_count,
        failure_count=0,
        error_rate=0.0,
        latency_p50_ms=100,
        latency_p95_ms=130,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        estimated_cost_usd=0.03,
        cost_evidence_complete=True,
        provider_attempt_count=request_count,
        provider_retry_count=0,
        route_count=request_count,
        budget_violation_count=0,
        model_distribution={"model": request_count},
        route_reason_distribution={"test": request_count},
        observed_latency_ms_by_model={"model": {"count": request_count, "p50": 100, "p95": 130}},
        quality_count=request_count,
        quality_pass_count=request_count,
        quality_pass_rate=1.0,
        quality_score_avg=1.0,
        ledger_path="ledger.jsonl",
        limitations=[],
    )


def _context(*, prefix: str, index: int) -> BenchmarkRequestContext:
    return BenchmarkRequestContext.from_tags(
        request_id=f"{prefix}-request-{index}",
        workload_item_id=f"item-{index:03d}",
        tags={"task": "qa", "risk": "high" if index < 15 else "low"},
    )


def _trace(
    *,
    request_id: str,
    latency_ms: int,
    cost: float | None,
    failed: bool = False,
    quality_passed: bool | None = True,
) -> RequestTrace:
    if failed:
        return RequestTrace(
            request_id=request_id,
            provider="provider",
            model="model",
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            pricing_table_version="not_charged",
            cache_hit=False,
            error_type="budget_violation",
            error_message="rejected",
            timestamp="2026-09-04T00:00:00+00:00",
            quality_passed=None,
            provider_attempt_count=0,
            provider_retry_count=0,
            cost_evidence_complete=True,
        )
    return RequestTrace(
        request_id=request_id,
        provider="provider",
        model="model",
        latency_ms=latency_ms,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost_usd=cost,
        pricing_table_version="test-v1" if cost is not None else "unpriced",
        cache_hit=False,
        error_type=None,
        error_message=None,
        timestamp="2026-09-04T00:00:00+00:00",
        quality_passed=quality_passed,
        quality_score=1.0 if quality_passed else None,
        provider_attempt_count=1,
        provider_retry_count=0,
        cost_evidence_complete=cost is not None,
    )


def _paired_data(*, candidate_failures: set[int] | None = None):
    failures = candidate_failures or set()
    baseline_contexts = [_context(prefix="base", index=index) for index in range(30)]
    candidate_contexts = [_context(prefix="cand", index=index) for index in range(30)]
    baseline_traces = []
    candidate_traces = []
    for index in range(30):
        base_cost = 0.001 + index * 0.000001
        baseline_traces.append(
            _trace(
                request_id=f"base-request-{index}",
                latency_ms=100 + index,
                cost=base_cost,
                quality_passed=True,
            )
        )
        candidate_traces.append(
            _trace(
                request_id=f"cand-request-{index}",
                latency_ms=90 + index + (index % 3),
                cost=base_cost - (index % 3 + 1) * 0.000001,
                failed=index in failures,
                quality_passed=index % 7 != 0,
            )
        )
    return baseline_contexts, candidate_contexts, baseline_traces, candidate_traces


def _config() -> PairedBootstrapConfig:
    return PairedBootstrapConfig(
        confidence_level=0.95,
        bootstrap_iterations=1_000,
        minimum_samples=5,
        seed=31,
    )


def test_provider_migration_is_pairable_by_workload_identity_not_request_uuid_or_order() -> None:
    baseline_contexts, candidate_contexts, baseline_traces, candidate_traces = _paired_data()

    evidence = compare_paired_runs(
        baseline_run_id="baseline-a",
        candidate_run_id="candidate-a",
        baseline_report=_report(request_count=30, provider="openai"),
        candidate_report=_report(request_count=30, provider="other-provider"),
        baseline_contexts=list(reversed(baseline_contexts)),
        candidate_contexts=candidate_contexts,
        baseline_traces=baseline_traces[::2] + baseline_traces[1::2],
        candidate_traces=list(reversed(candidate_traces)),
        bootstrap_config=_config(),
    )

    assert evidence.available is True
    assert evidence.workload_item_count == 30
    assert evidence.baseline_provider == "openai"
    assert evidence.candidate_provider == "other-provider"
    assert evidence.execution_cost_usd is not None
    assert evidence.execution_cost_usd.eligible_pair_count == 30
    assert evidence.execution_cost_usd.pair_coverage == pytest.approx(1.0)
    assert evidence.execution_cost_usd.estimate.status == StatisticalEvidenceStatus.SUFFICIENT
    assert evidence.execution_cost_usd.estimate.observed_mean_difference is not None
    assert evidence.execution_cost_usd.estimate.observed_mean_difference < 0
    assert evidence.successful_latency_ms is not None
    assert evidence.successful_latency_ms.estimate.observed_mean_difference is not None
    assert evidence.successful_latency_ms.estimate.observed_mean_difference < 0
    assert len(evidence.segments) == 3
    assert evidence.tail_latency_inference_supported is False


def test_bootstrap_evidence_is_independent_of_opaque_run_ids() -> None:
    baseline_contexts, candidate_contexts, baseline_traces, candidate_traces = _paired_data()
    kwargs = {
        "baseline_report": _report(request_count=30, provider="openai"),
        "candidate_report": _report(request_count=30, provider="openai"),
        "baseline_contexts": baseline_contexts,
        "candidate_contexts": candidate_contexts,
        "baseline_traces": baseline_traces,
        "candidate_traces": candidate_traces,
        "bootstrap_config": _config(),
    }

    first = compare_paired_runs(
        baseline_run_id="run-one",
        candidate_run_id="run-two",
        **kwargs,
    )
    second = compare_paired_runs(
        baseline_run_id="opaque-a",
        candidate_run_id="opaque-b",
        **kwargs,
    )

    assert first.execution_cost_usd is not None
    assert second.execution_cost_usd is not None
    assert first.execution_cost_usd.estimate == second.execution_cost_usd.estimate
    assert first.successful_latency_ms is not None
    assert second.successful_latency_ms is not None
    assert first.successful_latency_ms.estimate == second.successful_latency_ms.estimate


def test_sparse_binary_discordance_suppresses_failure_interval() -> None:
    baseline_contexts, candidate_contexts, baseline_traces, candidate_traces = _paired_data(
        candidate_failures={0}
    )

    evidence = compare_paired_runs(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        baseline_report=_report(request_count=30, provider="openai"),
        candidate_report=_report(request_count=30, provider="openai"),
        baseline_contexts=baseline_contexts,
        candidate_contexts=candidate_contexts,
        baseline_traces=baseline_traces,
        candidate_traces=candidate_traces,
        bootstrap_config=_config(),
    )

    assert evidence.failure_rate is not None
    failure = evidence.failure_rate.estimate
    assert failure.changed_pair_count == 1
    assert failure.minimum_changed_pair_count == 10
    assert failure.observed_mean_difference == pytest.approx(1 / 30)
    assert failure.status == StatisticalEvidenceStatus.INSUFFICIENT_VARIATION
    assert failure.interval_available is False
    assert evidence.successful_latency_ms is not None
    assert evidence.successful_latency_ms.eligible_pair_count == 29
    assert evidence.successful_latency_ms.pair_coverage == pytest.approx(29 / 30)


def test_unknown_cost_reduces_only_cost_pair_coverage() -> None:
    baseline_contexts, candidate_contexts, baseline_traces, candidate_traces = _paired_data()
    candidate_traces[3] = replace(
        candidate_traces[3],
        estimated_cost_usd=None,
        cost_evidence_complete=False,
        pricing_table_version="unpriced",
    )

    evidence = compare_paired_runs(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        baseline_report=_report(request_count=30, provider="openai"),
        candidate_report=_report(request_count=30, provider="openai"),
        baseline_contexts=baseline_contexts,
        candidate_contexts=candidate_contexts,
        baseline_traces=baseline_traces,
        candidate_traces=candidate_traces,
        bootstrap_config=_config(),
    )

    assert evidence.execution_cost_usd is not None
    assert evidence.execution_cost_usd.eligible_pair_count == 29
    assert evidence.execution_cost_usd.pair_coverage == pytest.approx(29 / 30)
    assert evidence.failure_rate is not None
    assert evidence.failure_rate.eligible_pair_count == 30


def test_report_context_cardinality_mismatch_fails_closed_before_inference() -> None:
    baseline_contexts, candidate_contexts, baseline_traces, candidate_traces = _paired_data()

    evidence = compare_paired_runs(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        baseline_report=_report(request_count=29, provider="openai"),
        candidate_report=_report(request_count=30, provider="openai"),
        baseline_contexts=baseline_contexts,
        candidate_contexts=candidate_contexts,
        baseline_traces=baseline_traces,
        candidate_traces=candidate_traces,
        bootstrap_config=_config(),
    )

    assert evidence.available is False
    assert evidence.unavailable_reason is not None
    assert "request_count" in evidence.unavailable_reason


def test_workload_hash_or_tag_mismatch_fails_closed() -> None:
    baseline_contexts, candidate_contexts, baseline_traces, candidate_traces = _paired_data()

    hash_mismatch = compare_paired_runs(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        baseline_report=_report(request_count=30, provider="openai", workload_sha256="a"),
        candidate_report=_report(request_count=30, provider="openai", workload_sha256="b"),
        baseline_contexts=baseline_contexts,
        candidate_contexts=candidate_contexts,
        baseline_traces=baseline_traces,
        candidate_traces=candidate_traces,
        bootstrap_config=_config(),
    )
    assert hash_mismatch.available is False
    assert "SHA256" in (hash_mismatch.unavailable_reason or "")

    candidate_contexts[0] = BenchmarkRequestContext.from_tags(
        request_id=candidate_contexts[0].request_id,
        workload_item_id=candidate_contexts[0].workload_item_id,
        tags={"task": "different", "risk": "high"},
    )
    tag_mismatch = compare_paired_runs(
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        baseline_report=_report(request_count=30, provider="openai"),
        candidate_report=_report(request_count=30, provider="openai"),
        baseline_contexts=baseline_contexts,
        candidate_contexts=candidate_contexts,
        baseline_traces=baseline_traces,
        candidate_traces=candidate_traces,
        bootstrap_config=_config(),
    )
    assert tag_mismatch.available is False
    assert "tags differ" in (tag_mismatch.unavailable_reason or "")


def test_exact_context_trace_coverage_is_required() -> None:
    baseline_contexts, candidate_contexts, baseline_traces, candidate_traces = _paired_data()

    with pytest.raises(ValueError, match="exact context/trace coverage"):
        compare_paired_runs(
            baseline_run_id="baseline",
            candidate_run_id="candidate",
            baseline_report=_report(request_count=30, provider="openai"),
            candidate_report=_report(request_count=30, provider="openai"),
            baseline_contexts=baseline_contexts,
            candidate_contexts=candidate_contexts,
            baseline_traces=baseline_traces[:-1],
            candidate_traces=candidate_traces,
            bootstrap_config=_config(),
        )
