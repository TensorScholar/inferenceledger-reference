from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from ..infrastructure.telemetry.request_log import RequestTrace, RouteTrace
from .reconciliation import reconcile_run_costs

_LATENCY_PERCENTILE_METHOD = "empirical_nearest_rank_successful_requests"


@dataclass(frozen=True)
class BenchmarkRequestContext:
    """Benchmark-owned identity and tags for one executed workload item.

    Raw provider/request telemetry deliberately does not own workload segmentation metadata.
    Tags are stored as a sorted immutable tuple so evidence is deterministic and cannot be
    mutated after collection.
    """

    request_id: str
    workload_item_id: str
    tags: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not self.workload_item_id.strip():
            raise ValueError("workload_item_id must be non-empty")
        if tuple(sorted(self.tags)) != self.tags:
            raise ValueError("tags must be sorted for deterministic evidence")
        keys: set[str] = set()
        for key, value in self.tags:
            if not key.strip() or not value.strip():
                raise ValueError("tag keys and values must be non-empty")
            if key in keys:
                raise ValueError(f"duplicate tag key: {key}")
            keys.add(key)

    @classmethod
    def from_tags(
        cls,
        *,
        request_id: str,
        workload_item_id: str,
        tags: dict[str, str],
    ) -> BenchmarkRequestContext:
        return cls(
            request_id=request_id,
            workload_item_id=workload_item_id,
            tags=tuple(sorted(tags.items())),
        )

    def tags_dict(self) -> dict[str, str]:
        return dict(self.tags)


@dataclass(frozen=True, order=True)
class SegmentKey:
    tag_key: str
    tag_value: str


@dataclass(frozen=True)
class SegmentEvidence:
    """Evidence for one dynamic workload segment identified by one tag key/value pair."""

    tag_key: str
    tag_value: str
    request_count: int
    success_count: int
    failure_count: int
    error_rate: float
    latency_sample_count: int
    latency_p50_ms: int | None
    latency_p95_ms: int | None
    latency_p99_ms: int | None
    provider_attempt_count: int
    provider_retry_count: int
    execution_path_divergence_count: int
    execution_path_divergence_rate: float | None
    estimated_cost_usd: float | None
    cost_evidence_complete: bool
    cost_per_success_usd: float | None
    quality_count: int
    quality_coverage: float | None
    quality_pass_count: int
    quality_pass_rate: float | None
    cost_per_accepted_outcome_usd: float | None
    model_distribution: dict[str, int]


@dataclass(frozen=True)
class SegmentEvidenceSummary:
    """Derived segment evidence for one benchmark run.

    Segment request counts are intentionally non-additive because a request can have multiple
    tags and therefore belong to multiple segments.
    """

    available: bool
    unavailable_reason: str | None
    request_count: int
    tagged_request_count: int
    untagged_request_count: int
    segment_count: int
    latency_percentile_method: str
    segments: tuple[SegmentEvidence, ...]


def summarize_segments(
    *,
    request_contexts: list[BenchmarkRequestContext],
    traces: list[RequestTrace],
    routes: list[RouteTrace] | None = None,
) -> SegmentEvidenceSummary:
    """Build dynamic segment evidence without imposing a fixed workload taxonomy.

    Empty contexts are treated as legacy/unavailable evidence. Once contexts exist, coverage is
    strict: every trace must have exactly one context and every context must resolve to a trace.
    This prevents partial tagging from silently biasing segment-level conclusions.
    """
    if not request_contexts:
        return SegmentEvidenceSummary(
            available=False,
            unavailable_reason="benchmark request context was not recorded for this run",
            request_count=len(traces),
            tagged_request_count=0,
            untagged_request_count=len(traces),
            segment_count=0,
            latency_percentile_method=_LATENCY_PERCENTILE_METHOD,
            segments=(),
        )

    contexts_by_id = _unique_contexts(request_contexts)
    traces_by_id = _unique_traces(traces)
    context_ids = set(contexts_by_id)
    trace_ids = set(traces_by_id)
    if context_ids != trace_ids:
        missing_context = sorted(trace_ids - context_ids)
        missing_trace = sorted(context_ids - trace_ids)
        raise ValueError(
            "segment evidence requires exact request-context coverage; "
            f"missing_context={missing_context}, missing_trace={missing_trace}"
        )

    routes_by_id = _unique_routes(routes or [])
    unknown_route_ids = sorted(set(routes_by_id) - trace_ids)
    if unknown_route_ids:
        raise ValueError(
            "segment evidence routes must resolve to recorded traces; "
            f"unknown_route_ids={unknown_route_ids}"
        )

    segment_members: dict[SegmentKey, list[str]] = {}
    tagged_request_count = 0
    for context in request_contexts:
        if context.tags:
            tagged_request_count += 1
        for key, value in context.tags:
            segment_members.setdefault(SegmentKey(key, value), []).append(context.request_id)

    segments = tuple(
        _summarize_segment(
            key=key,
            request_ids=request_ids,
            traces_by_id=traces_by_id,
            routes_by_id=routes_by_id,
        )
        for key, request_ids in sorted(segment_members.items())
    )
    return SegmentEvidenceSummary(
        available=True,
        unavailable_reason=None,
        request_count=len(traces),
        tagged_request_count=tagged_request_count,
        untagged_request_count=len(traces) - tagged_request_count,
        segment_count=len(segments),
        latency_percentile_method=_LATENCY_PERCENTILE_METHOD,
        segments=segments,
    )


def empirical_nearest_rank(values: list[int], percentile: int) -> int:
    """Return the empirical nearest-rank percentile for a non-empty sample.

    This method is deliberately explicit and conservative for tail-SLO evidence: with small
    samples, p95/p99 tend toward the observed maximum rather than interpolating an unobserved
    latency. Statistical confidence/sufficiency is a separate decision-layer concern.
    """
    if not values:
        raise ValueError("percentile requires at least one value")
    if percentile < 0 or percentile > 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if percentile == 0:
        return ordered[0]
    rank = ceil((percentile / 100) * len(ordered))
    return ordered[max(rank - 1, 0)]


def _summarize_segment(
    *,
    key: SegmentKey,
    request_ids: list[str],
    traces_by_id: dict[str, RequestTrace],
    routes_by_id: dict[str, RouteTrace],
) -> SegmentEvidence:
    traces = [traces_by_id[request_id] for request_id in request_ids]
    segment_routes = [
        routes_by_id[request_id]
        for request_id in request_ids
        if request_id in routes_by_id
    ]
    success_traces = [trace for trace in traces if trace.error_type is None]
    failure_count = len(traces) - len(success_traces)
    successful_latencies = [trace.latency_ms for trace in success_traces]

    cost_evidence_complete = all(
        trace.cost_evidence_complete and trace.estimated_cost_usd is not None for trace in traces
    )
    total_cost = (
        sum(_required_cost(trace.estimated_cost_usd) for trace in traces)
        if cost_evidence_complete
        else None
    )
    cost_per_success = (
        total_cost / len(success_traces)
        if total_cost is not None and success_traces
        else None
    )

    quality_traces = [trace for trace in success_traces if trace.quality_passed is not None]
    quality_pass_count = sum(1 for trace in quality_traces if trace.quality_passed)
    quality_coverage = (
        len(quality_traces) / len(success_traces) if success_traces else None
    )
    quality_pass_rate = (
        quality_pass_count / len(quality_traces) if quality_traces else None
    )
    cost_per_accepted: float | None = None
    if (
        total_cost is not None
        and success_traces
        and len(quality_traces) == len(success_traces)
        and quality_pass_count > 0
    ):
        cost_per_accepted = total_cost / quality_pass_count

    reconciliation = reconcile_run_costs(routes=segment_routes, executions=traces)
    model_distribution: dict[str, int] = {}
    for trace in success_traces:
        model_distribution[trace.model] = model_distribution.get(trace.model, 0) + 1

    return SegmentEvidence(
        tag_key=key.tag_key,
        tag_value=key.tag_value,
        request_count=len(traces),
        success_count=len(success_traces),
        failure_count=failure_count,
        error_rate=failure_count / len(traces),
        latency_sample_count=len(successful_latencies),
        latency_p50_ms=(
            empirical_nearest_rank(successful_latencies, 50)
            if successful_latencies
            else None
        ),
        latency_p95_ms=(
            empirical_nearest_rank(successful_latencies, 95)
            if successful_latencies
            else None
        ),
        latency_p99_ms=(
            empirical_nearest_rank(successful_latencies, 99)
            if successful_latencies
            else None
        ),
        provider_attempt_count=sum(trace.provider_attempt_count for trace in traces),
        provider_retry_count=sum(trace.provider_retry_count for trace in traces),
        execution_path_divergence_count=reconciliation.execution_path_divergence_count,
        execution_path_divergence_rate=reconciliation.execution_path_divergence_rate,
        estimated_cost_usd=total_cost,
        cost_evidence_complete=cost_evidence_complete,
        cost_per_success_usd=cost_per_success,
        quality_count=len(quality_traces),
        quality_coverage=quality_coverage,
        quality_pass_count=quality_pass_count,
        quality_pass_rate=quality_pass_rate,
        cost_per_accepted_outcome_usd=cost_per_accepted,
        model_distribution=dict(sorted(model_distribution.items())),
    )


def _unique_contexts(
    contexts: list[BenchmarkRequestContext],
) -> dict[str, BenchmarkRequestContext]:
    result: dict[str, BenchmarkRequestContext] = {}
    workload_item_ids: set[str] = set()
    for context in contexts:
        if context.request_id in result:
            raise ValueError(f"duplicate benchmark context request_id: {context.request_id}")
        if context.workload_item_id in workload_item_ids:
            raise ValueError(
                f"duplicate benchmark workload_item_id: {context.workload_item_id}"
            )
        result[context.request_id] = context
        workload_item_ids.add(context.workload_item_id)
    return result


def _unique_traces(traces: list[RequestTrace]) -> dict[str, RequestTrace]:
    result: dict[str, RequestTrace] = {}
    for trace in traces:
        if trace.request_id in result:
            raise ValueError(f"duplicate trace request_id: {trace.request_id}")
        result[trace.request_id] = trace
    return result


def _unique_routes(routes: list[RouteTrace]) -> dict[str, RouteTrace]:
    result: dict[str, RouteTrace] = {}
    for route in routes:
        if route.request_id in result:
            raise ValueError(f"duplicate route request_id: {route.request_id}")
        result[route.request_id] = route
    return result


def _required_cost(value: float | None) -> float:
    if value is None:
        raise ValueError("complete segment cost evidence unexpectedly contains unknown cost")
    return value
