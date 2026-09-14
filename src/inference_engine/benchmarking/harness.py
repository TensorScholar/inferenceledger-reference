from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from ..domain.models.execution import CostEvidenceKind
from ..infrastructure.telemetry.request_log import RequestTrace, RouteTrace
from .eval import EvalSpec, parse_eval_spec


@dataclass(frozen=True)
class WorkloadItem:
    """One prompt in a replayable benchmark workload."""

    id: str
    prompt: str
    tags: dict[str, str]
    eval_spec: EvalSpec | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregate report generated from raw request trace records."""

    workload_path: str
    workload_sha256: str | None
    strategy: str
    provider: str
    model: str
    request_count: int
    success_count: int
    failure_count: int
    error_rate: float
    latency_p50_ms: int
    latency_p95_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    cost_evidence_complete: bool
    provider_attempt_count: int
    provider_retry_count: int
    route_count: int
    budget_violation_count: int
    model_distribution: dict[str, int]
    route_reason_distribution: dict[str, int]
    observed_latency_ms_by_model: dict[str, dict[str, int]]
    quality_count: int
    quality_pass_count: int
    quality_pass_rate: float | None
    quality_score_avg: float | None
    ledger_path: str
    limitations: list[str]


@dataclass(frozen=True)
class BenchmarkComparison:
    """Comparison between one baseline run and one candidate run."""

    baseline_run_id: str
    candidate_run_id: str
    workload_path: str
    baseline_strategy: str
    candidate_strategy: str
    baseline_cost_usd: float | None
    candidate_cost_usd: float | None
    cost_delta_usd: float | None
    cost_delta_percent: float | None
    baseline_latency_p95_ms: int
    candidate_latency_p95_ms: int
    latency_p95_delta_ms: int
    baseline_error_rate: float
    candidate_error_rate: float
    baseline_quality_pass_rate: float | None
    candidate_quality_pass_rate: float | None
    quality_pass_rate_delta: float | None
    cost_evidence_complete: bool
    comparable: bool
    limitations: list[str]


def load_workload(path: Path) -> list[WorkloadItem]:
    items: list[WorkloadItem] = []
    seen_item_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            item_id = raw.get("id")
            prompt = raw.get("prompt")
            tags = raw.get("tags", {})
            if not isinstance(item_id, str) or not item_id.strip():
                raise ValueError(f"{path}:{line_number} missing string id")
            if item_id in seen_item_ids:
                raise ValueError(f"{path}:{line_number} duplicate workload item id: {item_id}")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{path}:{line_number} missing string prompt")
            if not isinstance(tags, dict) or not all(
                isinstance(key, str)
                and isinstance(value, str)
                and bool(key.strip())
                and bool(value.strip())
                for key, value in tags.items()
            ):
                raise ValueError(
                    f"{path}:{line_number} tags must be an object of non-empty strings"
                )
            try:
                eval_spec = parse_eval_spec(raw.get("eval"))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} invalid eval: {exc}") from exc
            seen_item_ids.add(item_id)
            items.append(WorkloadItem(id=item_id, prompt=prompt, tags=tags, eval_spec=eval_spec))
    if not items:
        raise ValueError(f"{path} did not contain any workload items")
    return items


def summarize_traces(
    *,
    workload_path: Path,
    strategy: str,
    provider: str,
    model: str,
    ledger_path: Path,
    traces: list[RequestTrace],
    route_traces: list[RouteTrace] | None = None,
) -> BenchmarkReport:
    routes = route_traces or []
    latencies = sorted(trace.latency_ms for trace in traces)
    success_traces = [trace for trace in traces if trace.error_type is None]
    quality_traces = [trace for trace in success_traces if trace.quality_passed is not None]
    quality_pass_count = sum(1 for trace in quality_traces if trace.quality_passed)
    request_count = len(traces)
    failure_count = request_count - len(success_traces)
    cost_evidence_complete = all(
        trace.cost_evidence_complete and trace.estimated_cost_usd is not None for trace in traces
    )
    estimated_cost_usd = (
        sum(trace.estimated_cost_usd or 0.0 for trace in traces) if cost_evidence_complete else None
    )
    limitations = [
        _cost_origin_limitation(traces),
        "Quality scoring is deterministic and limited to workload-declared validators.",
        "Do not claim savings unless comparison quality remains acceptable.",
    ]
    if not cost_evidence_complete:
        limitations.append(
            "At least one executed provider attempt has unknown cost; aggregate execution cost and savings are not defensible."
        )

    return BenchmarkReport(
        workload_path=str(workload_path),
        workload_sha256=_file_sha256(workload_path),
        strategy=strategy,
        provider=provider,
        model=model,
        request_count=request_count,
        success_count=len(success_traces),
        failure_count=failure_count,
        error_rate=failure_count / request_count if request_count else 0.0,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        prompt_tokens=sum(trace.prompt_tokens for trace in success_traces),
        completion_tokens=sum(trace.completion_tokens for trace in success_traces),
        total_tokens=sum(trace.total_tokens for trace in success_traces),
        estimated_cost_usd=estimated_cost_usd,
        cost_evidence_complete=cost_evidence_complete,
        provider_attempt_count=sum(trace.provider_attempt_count for trace in traces),
        provider_retry_count=sum(trace.provider_retry_count for trace in traces),
        route_count=len(routes),
        budget_violation_count=sum(1 for route in routes if route.budget_violation),
        model_distribution=_count_by_model(success_traces),
        route_reason_distribution=_count_by_reason(routes),
        observed_latency_ms_by_model=_latency_profiles(success_traces),
        quality_count=len(quality_traces),
        quality_pass_count=quality_pass_count,
        quality_pass_rate=quality_pass_count / len(quality_traces) if quality_traces else None,
        quality_score_avg=sum(trace.quality_score or 0.0 for trace in quality_traces)
        / len(quality_traces)
        if quality_traces
        else None,
        ledger_path=str(ledger_path),
        limitations=limitations,
    )


def write_report(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2, sort_keys=True)
        handle.write("\n")


def compare_reports(
    *,
    baseline_run_id: str,
    baseline: BenchmarkReport,
    candidate_run_id: str,
    candidate: BenchmarkReport,
) -> BenchmarkComparison:
    same_workload_path = baseline.workload_path == candidate.workload_path
    same_workload_hash = (
        baseline.workload_sha256 == candidate.workload_sha256
        if baseline.workload_sha256 is not None and candidate.workload_sha256 is not None
        else True
    )
    same_request_count = baseline.request_count == candidate.request_count
    same_quality_coverage = baseline.quality_count == candidate.quality_count
    candidate_quality_ok = (
        candidate.quality_pass_rate is not None
        and baseline.quality_pass_rate is not None
        and candidate.quality_pass_rate >= baseline.quality_pass_rate
    )
    same_workload = same_workload_path and same_workload_hash and same_request_count
    cost_evidence_complete = (
        baseline.cost_evidence_complete
        and candidate.cost_evidence_complete
        and baseline.estimated_cost_usd is not None
        and candidate.estimated_cost_usd is not None
    )
    comparable = (
        same_workload
        and baseline.provider == candidate.provider
        and same_quality_coverage
        and candidate_quality_ok
        and cost_evidence_complete
    )

    cost_delta: float | None = None
    cost_delta_percent: float | None = None
    if cost_evidence_complete:
        assert baseline.estimated_cost_usd is not None
        assert candidate.estimated_cost_usd is not None
        cost_delta = candidate.estimated_cost_usd - baseline.estimated_cost_usd
        if baseline.estimated_cost_usd > 0:
            cost_delta_percent = (cost_delta / baseline.estimated_cost_usd) * 100

    limitations = [
        "Comparison uses stored run summaries from the local SQLite ledger.",
        "Comparison requires candidate quality pass rate to meet or exceed baseline before savings are defensible.",
    ]
    if not same_workload_path:
        limitations.append("Runs used different workload paths and are not comparable.")
    if not same_workload_hash:
        limitations.append("Runs used different workload hashes and are not comparable.")
    if not same_request_count:
        limitations.append("Runs used different request counts and are not comparable.")
    if baseline.provider != candidate.provider:
        limitations.append("Runs used different providers and are not comparable.")
    if not same_quality_coverage:
        limitations.append("Runs have different quality-evaluated request counts.")
    if baseline.quality_pass_rate is None or candidate.quality_pass_rate is None:
        limitations.append("Quality pass rate is unavailable; do not claim savings.")
    elif not candidate_quality_ok:
        limitations.append("Candidate quality pass rate is below baseline.")
    if not cost_evidence_complete:
        limitations.append(
            "Baseline or candidate has incomplete provider-attempt cost evidence; do not calculate or claim savings."
        )

    return BenchmarkComparison(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        workload_path=baseline.workload_path,
        baseline_strategy=baseline.strategy,
        candidate_strategy=candidate.strategy,
        baseline_cost_usd=baseline.estimated_cost_usd,
        candidate_cost_usd=candidate.estimated_cost_usd,
        cost_delta_usd=cost_delta,
        cost_delta_percent=cost_delta_percent,
        baseline_latency_p95_ms=baseline.latency_p95_ms,
        candidate_latency_p95_ms=candidate.latency_p95_ms,
        latency_p95_delta_ms=candidate.latency_p95_ms - baseline.latency_p95_ms,
        baseline_error_rate=baseline.error_rate,
        candidate_error_rate=candidate.error_rate,
        baseline_quality_pass_rate=baseline.quality_pass_rate,
        candidate_quality_pass_rate=candidate.quality_pass_rate,
        quality_pass_rate_delta=(
            candidate.quality_pass_rate - baseline.quality_pass_rate
            if candidate.quality_pass_rate is not None and baseline.quality_pass_rate is not None
            else None
        ),
        cost_evidence_complete=cost_evidence_complete,
        comparable=comparable,
        limitations=limitations,
    )


def write_comparison(comparison: BenchmarkComparison, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(comparison), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _percentile(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    index = round((percentile / 100) * (len(values) - 1))
    return values[index]


def _count_by_model(traces: list[RequestTrace]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trace in traces:
        counts[trace.model] = counts.get(trace.model, 0) + 1
    return dict(sorted(counts.items()))


def _count_by_reason(routes: list[RouteTrace]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for route in routes:
        counts[route.decision_reason] = counts.get(route.decision_reason, 0) + 1
    return dict(sorted(counts.items()))


def _latency_profiles(traces: list[RequestTrace]) -> dict[str, dict[str, int]]:
    latencies_by_model: dict[str, list[int]] = {}
    for trace in traces:
        latencies_by_model.setdefault(trace.model, []).append(trace.latency_ms)

    profiles: dict[str, dict[str, int]] = {}
    for model, latencies in sorted(latencies_by_model.items()):
        ordered = sorted(latencies)
        profiles[model] = {
            "count": len(ordered),
            "p50": _percentile(ordered, 50),
            "p95": _percentile(ordered, 95),
        }
    return profiles


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cost_origin_limitation(traces: list[RequestTrace]) -> str:
    kinds = {
        attempt.cost_evidence
        for trace in traces
        for attempt in trace.provider_attempts
    }
    if kinds == {CostEvidenceKind.REPORTED_BY_EXECUTION_STACK}:
        return (
            "Known cost is reported by the execution stack; it is not InferenceLedger "
            "pricing-table reconstruction and not provider invoice proof."
        )
    if kinds == {CostEvidenceKind.CALCULATED_FROM_USAGE} or not kinds:
        return (
            "Known cost is calculated from observed provider usage and the repository pricing table; "
            "it is not provider invoice proof."
        )
    return (
        "Attempt-level cost evidence is mixed, unknown, or incomplete; do not treat aggregate cost "
        "as a single pricing authority or as provider invoice proof."
    )
