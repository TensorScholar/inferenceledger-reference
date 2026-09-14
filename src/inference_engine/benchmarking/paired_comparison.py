from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from ..infrastructure.telemetry.request_log import RequestTrace
from .harness import BenchmarkReport
from .segmentation import BenchmarkRequestContext, SegmentKey
from .statistics import (
    PairedBootstrapConfig,
    PairedMeanDifferenceEstimate,
    paired_mean_difference_bca,
)

_MINIMUM_BINARY_DISCORDANT_PAIRS = 10


@dataclass(frozen=True)
class PairedMetricEvidence:
    """One candidate-minus-baseline metric over an explicit paired population."""

    metric: str
    unit: str
    population: str
    lower_is_better: bool
    workload_item_count: int
    eligible_pair_count: int
    pair_coverage: float
    estimate: PairedMeanDifferenceEstimate


@dataclass(frozen=True)
class PairedSegmentStatisticalEvidence:
    """Paired central-statistical evidence for one shared workload segment."""

    tag_key: str
    tag_value: str
    workload_item_count: int
    execution_cost_usd: PairedMetricEvidence
    failure_rate: PairedMetricEvidence
    successful_latency_ms: PairedMetricEvidence
    accepted_outcome_rate: PairedMetricEvidence
    provider_attempt_count: PairedMetricEvidence
    provider_retry_count: PairedMetricEvidence


@dataclass(frozen=True)
class PairedRunStatisticalEvidence:
    """Reproducible workload-paired uncertainty evidence for one candidate change."""

    available: bool
    unavailable_reason: str | None
    baseline_run_id: str
    candidate_run_id: str
    workload_sha256: str | None
    workload_item_count: int
    baseline_provider: str
    candidate_provider: str
    execution_cost_usd: PairedMetricEvidence | None
    failure_rate: PairedMetricEvidence | None
    successful_latency_ms: PairedMetricEvidence | None
    accepted_outcome_rate: PairedMetricEvidence | None
    provider_attempt_count: PairedMetricEvidence | None
    provider_retry_count: PairedMetricEvidence | None
    segments: tuple[PairedSegmentStatisticalEvidence, ...]
    tail_latency_inference_supported: bool
    limitations: tuple[str, ...]


def write_paired_run_evidence(evidence: PairedRunStatisticalEvidence, path: Path) -> None:
    """Persist a deterministic JSON evidence artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class PairingAudit:
    """Explicit workload_item_id pairing inventory. Ambiguity is reported, never repaired."""

    baseline_count: int
    candidate_count: int
    matched_pair_count: int
    unmatched_baseline_item_ids: tuple[str, ...]
    unmatched_candidate_item_ids: tuple[str, ...]
    duplicate_baseline_item_ids: tuple[str, ...]
    duplicate_candidate_item_ids: tuple[str, ...]
    duplicate_baseline_request_ids: tuple[str, ...]
    duplicate_candidate_request_ids: tuple[str, ...]
    coverage: float
    ambiguous: bool
    fail_closed_reason: str | None


def audit_workload_pairing(
    *,
    baseline_contexts: list[BenchmarkRequestContext],
    candidate_contexts: list[BenchmarkRequestContext],
) -> PairingAudit:
    """Audit pairing by immutable workload_item_id, never request UUID or execution order."""
    baseline_item_ids = [context.workload_item_id for context in baseline_contexts]
    candidate_item_ids = [context.workload_item_id for context in candidate_contexts]
    baseline_request_ids = [context.request_id for context in baseline_contexts]
    candidate_request_ids = [context.request_id for context in candidate_contexts]

    duplicate_baseline_item_ids = _duplicate_values(baseline_item_ids)
    duplicate_candidate_item_ids = _duplicate_values(candidate_item_ids)
    duplicate_baseline_request_ids = _duplicate_values(baseline_request_ids)
    duplicate_candidate_request_ids = _duplicate_values(candidate_request_ids)

    baseline_set = set(baseline_item_ids)
    candidate_set = set(candidate_item_ids)
    unmatched_baseline = tuple(sorted(baseline_set - candidate_set))
    unmatched_candidate = tuple(sorted(candidate_set - baseline_set))
    matched = tuple(sorted(baseline_set & candidate_set))
    union_count = len(baseline_set | candidate_set)
    coverage = (len(matched) / union_count) if union_count else 0.0

    reasons: list[str] = []
    if duplicate_baseline_item_ids:
        reasons.append(f"duplicate baseline workload_item_id values: {list(duplicate_baseline_item_ids)}")
    if duplicate_candidate_item_ids:
        reasons.append(
            f"duplicate candidate workload_item_id values: {list(duplicate_candidate_item_ids)}"
        )
    if duplicate_baseline_request_ids:
        reasons.append(f"duplicate baseline request_id values: {list(duplicate_baseline_request_ids)}")
    if duplicate_candidate_request_ids:
        reasons.append(
            f"duplicate candidate request_id values: {list(duplicate_candidate_request_ids)}"
        )
    if unmatched_baseline or unmatched_candidate:
        reasons.append(
            "unmatched workload_item_id values; "
            f"baseline_only={list(unmatched_baseline)}, candidate_only={list(unmatched_candidate)}"
        )
    if len(baseline_contexts) != len(baseline_set):
        reasons.append("baseline pairing cannot proceed while duplicate workload item ids exist")
    if len(candidate_contexts) != len(candidate_set):
        reasons.append("candidate pairing cannot proceed while duplicate workload item ids exist")

    fail_closed_reason = None if not reasons else "; ".join(reasons)
    return PairingAudit(
        baseline_count=len(baseline_contexts),
        candidate_count=len(candidate_contexts),
        matched_pair_count=len(matched),
        unmatched_baseline_item_ids=unmatched_baseline,
        unmatched_candidate_item_ids=unmatched_candidate,
        duplicate_baseline_item_ids=duplicate_baseline_item_ids,
        duplicate_candidate_item_ids=duplicate_candidate_item_ids,
        duplicate_baseline_request_ids=duplicate_baseline_request_ids,
        duplicate_candidate_request_ids=duplicate_candidate_request_ids,
        coverage=coverage,
        ambiguous=fail_closed_reason is not None,
        fail_closed_reason=fail_closed_reason,
    )


def write_pairing_audit(audit: PairingAudit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(audit), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _duplicate_values(values: list[str]) -> tuple[str, ...]:
    counts = Counter(values)
    return tuple(sorted(value for value, count in counts.items() if count > 1))


@dataclass(frozen=True)
class _PairedObservation:
    workload_item_id: str
    tags: tuple[tuple[str, str], ...]
    baseline: RequestTrace
    candidate: RequestTrace


def compare_paired_runs(
    *,
    baseline_run_id: str,
    candidate_run_id: str,
    baseline_report: BenchmarkReport,
    candidate_report: BenchmarkReport,
    baseline_contexts: list[BenchmarkRequestContext],
    candidate_contexts: list[BenchmarkRequestContext],
    baseline_traces: list[RequestTrace],
    candidate_traces: list[RequestTrace],
    bootstrap_config: PairedBootstrapConfig | None = None,
) -> PairedRunStatisticalEvidence:
    """Compare two runs by immutable workload item identity, never UUID or list position.

    Provider identity may differ so model/provider migrations remain valid use cases. Workload
    identity is strict: hash, item set, tags, report cardinality, and context/trace coverage must
    agree before any paired statistical claim is produced.
    """
    config = bootstrap_config or PairedBootstrapConfig()
    unavailable = _validate_workload_identity(
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        baseline_contexts=baseline_contexts,
        candidate_contexts=candidate_contexts,
    )
    if unavailable is not None:
        return _unavailable_evidence(
            reason=unavailable,
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
        )

    observations = _pair_observations(
        baseline_contexts=baseline_contexts,
        candidate_contexts=candidate_contexts,
        baseline_traces=baseline_traces,
        candidate_traces=candidate_traces,
    )
    workload_sha = baseline_report.workload_sha256
    assert workload_sha is not None
    overall = _metric_bundle(
        observations=observations,
        scope="all",
        config=config,
        seed_namespace=f"{workload_sha}|all",
    )

    segment_members: dict[SegmentKey, list[_PairedObservation]] = {}
    for observation in observations:
        for key, value in observation.tags:
            segment_members.setdefault(SegmentKey(key, value), []).append(observation)
    segments = tuple(
        PairedSegmentStatisticalEvidence(
            tag_key=segment_key.tag_key,
            tag_value=segment_key.tag_value,
            workload_item_count=len(members),
            **_metric_bundle(
                observations=members,
                scope=f"segment:{segment_key.tag_key}={segment_key.tag_value}",
                config=config,
                seed_namespace=(
                    f"{workload_sha}|{segment_key.tag_key}={segment_key.tag_value}"
                ),
            ),
        )
        for segment_key, members in sorted(segment_members.items())
    )

    limitations = (
        "Intervals are candidate-minus-baseline central mean-difference uncertainty estimates under empirical workload-item resampling; they are not production guarantees.",
        "Successful-latency evidence is conditioned on workload items that succeeded in both runs; failure-rate evidence must be interpreted alongside it.",
        "Accepted-outcome evidence is emitted only where request-level acceptance is determinable in both runs; incomplete quality evidence reduces pair coverage.",
        "Failure-rate and accepted-outcome intervals use paired BCa risk-difference approximation and require at least 10 discordant pairs; they are not exact McNemar-compatible intervals.",
        "Samples with zero empirical paired-difference variance retain the observed effect but suppress confidence intervals rather than claiming degenerate certainty.",
        "High-tail latency inference, including p95/p99 confidence intervals, is intentionally not produced by this central paired bootstrap primitive.",
        "Configured sample and discordant-pair floors are conservative product policies, not universal statistical power guarantees.",
    )
    return PairedRunStatisticalEvidence(
        available=True,
        unavailable_reason=None,
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        workload_sha256=workload_sha,
        workload_item_count=len(observations),
        baseline_provider=baseline_report.provider,
        candidate_provider=candidate_report.provider,
        execution_cost_usd=overall["execution_cost_usd"],
        failure_rate=overall["failure_rate"],
        successful_latency_ms=overall["successful_latency_ms"],
        accepted_outcome_rate=overall["accepted_outcome_rate"],
        provider_attempt_count=overall["provider_attempt_count"],
        provider_retry_count=overall["provider_retry_count"],
        segments=segments,
        tail_latency_inference_supported=False,
        limitations=limitations,
    )


def _validate_workload_identity(
    *,
    baseline_report: BenchmarkReport,
    candidate_report: BenchmarkReport,
    baseline_contexts: list[BenchmarkRequestContext],
    candidate_contexts: list[BenchmarkRequestContext],
) -> str | None:
    if baseline_report.workload_sha256 is None or candidate_report.workload_sha256 is None:
        return "paired inference requires workload SHA256 evidence for both runs"
    if baseline_report.workload_sha256 != candidate_report.workload_sha256:
        return "baseline and candidate workload SHA256 values differ"
    if not baseline_contexts or not candidate_contexts:
        return "paired inference requires benchmark request context for both runs"
    if baseline_report.request_count != len(baseline_contexts):
        return (
            "baseline report request_count does not match benchmark context cardinality; "
            f"report={baseline_report.request_count}, contexts={len(baseline_contexts)}"
        )
    if candidate_report.request_count != len(candidate_contexts):
        return (
            "candidate report request_count does not match benchmark context cardinality; "
            f"report={candidate_report.request_count}, contexts={len(candidate_contexts)}"
        )

    baseline_by_item = _contexts_by_workload_item(baseline_contexts, kind="baseline")
    candidate_by_item = _contexts_by_workload_item(candidate_contexts, kind="candidate")
    baseline_ids = set(baseline_by_item)
    candidate_ids = set(candidate_by_item)
    if baseline_ids != candidate_ids:
        return (
            "baseline and candidate workload item sets differ; "
            f"baseline_only={sorted(baseline_ids - candidate_ids)}, "
            f"candidate_only={sorted(candidate_ids - baseline_ids)}"
        )
    mismatched_tags = sorted(
        item_id
        for item_id in baseline_ids
        if baseline_by_item[item_id].tags != candidate_by_item[item_id].tags
    )
    if mismatched_tags:
        return f"baseline and candidate workload tags differ for items: {mismatched_tags}"
    return None


def _pair_observations(
    *,
    baseline_contexts: list[BenchmarkRequestContext],
    candidate_contexts: list[BenchmarkRequestContext],
    baseline_traces: list[RequestTrace],
    candidate_traces: list[RequestTrace],
) -> list[_PairedObservation]:
    baseline_by_item = _contexts_by_workload_item(baseline_contexts, kind="baseline")
    candidate_by_item = _contexts_by_workload_item(candidate_contexts, kind="candidate")
    baseline_traces_by_id = _traces_by_request_id(baseline_traces, kind="baseline")
    candidate_traces_by_id = _traces_by_request_id(candidate_traces, kind="candidate")

    _require_context_trace_coverage(
        contexts=baseline_contexts,
        traces_by_id=baseline_traces_by_id,
        kind="baseline",
    )
    _require_context_trace_coverage(
        contexts=candidate_contexts,
        traces_by_id=candidate_traces_by_id,
        kind="candidate",
    )

    return [
        _PairedObservation(
            workload_item_id=item_id,
            tags=baseline_by_item[item_id].tags,
            baseline=baseline_traces_by_id[baseline_by_item[item_id].request_id],
            candidate=candidate_traces_by_id[candidate_by_item[item_id].request_id],
        )
        for item_id in sorted(baseline_by_item)
    ]


def _metric_bundle(
    *,
    observations: list[_PairedObservation],
    scope: str,
    config: PairedBootstrapConfig,
    seed_namespace: str,
) -> dict[str, PairedMetricEvidence]:
    return {
        "execution_cost_usd": _metric_evidence(
            observations=observations,
            metric="execution_cost_usd",
            unit="usd_per_request",
            population=f"{scope}: requests with complete execution-cost evidence in both runs",
            lower_is_better=True,
            values=_execution_cost_values,
            config=config,
            seed_namespace=seed_namespace,
        ),
        "failure_rate": _metric_evidence(
            observations=observations,
            metric="failure_rate",
            unit="rate",
            population=f"{scope}: all paired workload items",
            lower_is_better=True,
            values=lambda observation: (
                1.0 if observation.baseline.error_type is not None else 0.0,
                1.0 if observation.candidate.error_type is not None else 0.0,
            ),
            config=config,
            seed_namespace=seed_namespace,
            minimum_changed_pairs=_MINIMUM_BINARY_DISCORDANT_PAIRS,
        ),
        "successful_latency_ms": _metric_evidence(
            observations=observations,
            metric="successful_latency_ms",
            unit="milliseconds",
            population=f"{scope}: workload items successful in both runs",
            lower_is_better=True,
            values=_successful_latency_values,
            config=config,
            seed_namespace=seed_namespace,
        ),
        "accepted_outcome_rate": _metric_evidence(
            observations=observations,
            metric="accepted_outcome_rate",
            unit="rate",
            population=f"{scope}: workload items with determinable acceptance in both runs",
            lower_is_better=False,
            values=_accepted_outcome_values,
            config=config,
            seed_namespace=seed_namespace,
            minimum_changed_pairs=_MINIMUM_BINARY_DISCORDANT_PAIRS,
        ),
        "provider_attempt_count": _metric_evidence(
            observations=observations,
            metric="provider_attempt_count",
            unit="attempts_per_request",
            population=f"{scope}: all paired workload items",
            lower_is_better=True,
            values=lambda observation: (
                float(observation.baseline.provider_attempt_count),
                float(observation.candidate.provider_attempt_count),
            ),
            config=config,
            seed_namespace=seed_namespace,
        ),
        "provider_retry_count": _metric_evidence(
            observations=observations,
            metric="provider_retry_count",
            unit="retries_per_request",
            population=f"{scope}: all paired workload items",
            lower_is_better=True,
            values=lambda observation: (
                float(observation.baseline.provider_retry_count),
                float(observation.candidate.provider_retry_count),
            ),
            config=config,
            seed_namespace=seed_namespace,
        ),
    }


def _metric_evidence(
    *,
    observations: list[_PairedObservation],
    metric: str,
    unit: str,
    population: str,
    lower_is_better: bool,
    values: Callable[[_PairedObservation], tuple[float, float] | None],
    config: PairedBootstrapConfig,
    seed_namespace: str,
    minimum_changed_pairs: int = 0,
) -> PairedMetricEvidence:
    baseline: list[float] = []
    candidate: list[float] = []
    for observation in observations:
        resolved = values(observation)
        if resolved is None:
            continue
        baseline_value, candidate_value = resolved
        baseline.append(baseline_value)
        candidate.append(candidate_value)

    metric_config = replace(
        config,
        minimum_changed_pairs=max(config.minimum_changed_pairs, minimum_changed_pairs),
        seed=_derived_seed(config.seed, f"{seed_namespace}|{metric}"),
    )
    estimate = paired_mean_difference_bca(
        baseline=baseline,
        candidate=candidate,
        config=metric_config,
    )
    total = len(observations)
    return PairedMetricEvidence(
        metric=metric,
        unit=unit,
        population=population,
        lower_is_better=lower_is_better,
        workload_item_count=total,
        eligible_pair_count=len(baseline),
        pair_coverage=len(baseline) / total if total else 0.0,
        estimate=estimate,
    )


def _execution_cost_values(
    observation: _PairedObservation,
) -> tuple[float, float] | None:
    baseline = observation.baseline
    candidate = observation.candidate
    if (
        not baseline.cost_evidence_complete
        or baseline.estimated_cost_usd is None
        or not candidate.cost_evidence_complete
        or candidate.estimated_cost_usd is None
    ):
        return None
    return baseline.estimated_cost_usd, candidate.estimated_cost_usd


def _successful_latency_values(
    observation: _PairedObservation,
) -> tuple[float, float] | None:
    if observation.baseline.error_type is not None or observation.candidate.error_type is not None:
        return None
    return float(observation.baseline.latency_ms), float(observation.candidate.latency_ms)


def _accepted_outcome_values(
    observation: _PairedObservation,
) -> tuple[float, float] | None:
    baseline = _request_acceptance(observation.baseline)
    candidate = _request_acceptance(observation.candidate)
    if baseline is None or candidate is None:
        return None
    return float(baseline), float(candidate)


def _request_acceptance(trace: RequestTrace) -> int | None:
    if trace.error_type is not None:
        return 0
    if trace.quality_passed is None:
        return None
    return 1 if trace.quality_passed else 0


def _contexts_by_workload_item(
    contexts: list[BenchmarkRequestContext],
    *,
    kind: str,
) -> dict[str, BenchmarkRequestContext]:
    result: dict[str, BenchmarkRequestContext] = {}
    request_ids: set[str] = set()
    for context in contexts:
        if context.workload_item_id in result:
            raise ValueError(f"duplicate {kind} workload_item_id: {context.workload_item_id}")
        if context.request_id in request_ids:
            raise ValueError(f"duplicate {kind} context request_id: {context.request_id}")
        result[context.workload_item_id] = context
        request_ids.add(context.request_id)
    return result


def _traces_by_request_id(
    traces: list[RequestTrace],
    *,
    kind: str,
) -> dict[str, RequestTrace]:
    result: dict[str, RequestTrace] = {}
    for trace in traces:
        if trace.request_id in result:
            raise ValueError(f"duplicate {kind} trace request_id: {trace.request_id}")
        result[trace.request_id] = trace
    return result


def _require_context_trace_coverage(
    *,
    contexts: list[BenchmarkRequestContext],
    traces_by_id: dict[str, RequestTrace],
    kind: str,
) -> None:
    context_ids = {context.request_id for context in contexts}
    trace_ids = set(traces_by_id)
    if context_ids != trace_ids:
        raise ValueError(
            f"{kind} paired evidence requires exact context/trace coverage; "
            f"context_only={sorted(context_ids - trace_ids)}, "
            f"trace_only={sorted(trace_ids - context_ids)}"
        )


def _derived_seed(base_seed: int, namespace: str) -> int:
    digest = sha256(f"{base_seed}|{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _unavailable_evidence(
    *,
    reason: str,
    baseline_run_id: str,
    candidate_run_id: str,
    baseline_report: BenchmarkReport,
    candidate_report: BenchmarkReport,
) -> PairedRunStatisticalEvidence:
    return PairedRunStatisticalEvidence(
        available=False,
        unavailable_reason=reason,
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        workload_sha256=(
            baseline_report.workload_sha256
            if baseline_report.workload_sha256 == candidate_report.workload_sha256
            else None
        ),
        workload_item_count=0,
        baseline_provider=baseline_report.provider,
        candidate_provider=candidate_report.provider,
        execution_cost_usd=None,
        failure_rate=None,
        successful_latency_ms=None,
        accepted_outcome_rate=None,
        provider_attempt_count=None,
        provider_retry_count=None,
        segments=(),
        tail_latency_inference_supported=False,
        limitations=(reason,),
    )
