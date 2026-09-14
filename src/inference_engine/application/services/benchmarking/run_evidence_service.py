from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ....benchmarking.context_store import SQLiteBenchmarkContextStore
from ....benchmarking.harness import BenchmarkReport, summarize_traces, write_report
from ....benchmarking.pilot_contract import AcquisitionSemantics
from ....benchmarking.pilot_evidence_store import SQLitePilotEvidenceStore
from ....benchmarking.segmentation import (
    BenchmarkRequestContext,
    SegmentEvidenceSummary,
    summarize_segments,
)
from ....benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from ....domain.models.economics import ExactRequestCostEvidence
from ....infrastructure.telemetry.request_log import RequestTrace, RouteTrace


@dataclass(frozen=True)
class PersistedBenchmarkRun:
    """Canonical stored evidence for one executed or imported benchmark arm."""

    report: BenchmarkReport
    segments: SegmentEvidenceSummary


def persist_benchmark_run(
    *,
    run_id: str,
    workload_path: Path,
    strategy: str,
    provider: str,
    model: str,
    ledger_path: Path,
    sqlite_ledger_path: Path,
    traces: Sequence[RequestTrace],
    request_contexts: Sequence[BenchmarkRequestContext],
    route_traces: Sequence[RouteTrace] = (),
    report_path: Path | None = None,
    segment_report_path: Path | None = None,
    additional_limitations: Sequence[str] = (),
    pilot_acquisition: AcquisitionSemantics | None = None,
    exact_cost_evidence: Sequence[ExactRequestCostEvidence] = (),
) -> PersistedBenchmarkRun:
    """Persist one run through the authoritative pilot evidence boundary.

    Execution adapters and importers may obtain evidence differently, but they must hand the
    resulting traces, route traces, and immutable workload contexts to this service. The service
    owns run summarization, segment summarization, SQLite persistence, and optional deterministic
    report artifacts so those meanings cannot drift between adapters.

    Prospective comparative-pilot runs additionally provide acquisition semantics plus exact
    decimal-string request/attempt cost evidence. Those pilot surfaces are persisted separately
    from legacy float columns; they are required together and must cover exactly the same requests.
    """
    trace_list = list(traces)
    context_list = list(request_contexts)
    route_list = list(route_traces)
    exact_list = list(exact_cost_evidence)
    _require_exact_context_trace_coverage(context_list, trace_list)
    _validate_optional_pilot_evidence(
        traces=trace_list,
        acquisition=pilot_acquisition,
        exact_costs=exact_list,
    )

    report = summarize_traces(
        workload_path=workload_path,
        strategy=strategy,
        provider=provider,
        model=model,
        ledger_path=ledger_path,
        traces=trace_list,
        route_traces=route_list,
    )
    if additional_limitations:
        report = replace(
            report,
            limitations=[*report.limitations, *[str(item) for item in additional_limitations]],
        )

    segments = summarize_segments(
        request_contexts=context_list,
        traces=trace_list,
        routes=route_list,
    )

    if report_path is not None:
        write_report(report, report_path)
    if segment_report_path is not None:
        segment_report_path.parent.mkdir(parents=True, exist_ok=True)
        segment_report_path.write_text(
            json.dumps(asdict(segments), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    SQLiteBenchmarkLedger(sqlite_ledger_path).record_run(
        run_id=run_id,
        report=report,
        traces=trace_list,
        route_traces=route_list,
    )
    SQLiteBenchmarkContextStore(sqlite_ledger_path).record_contexts(
        run_id=run_id,
        contexts=context_list,
    )
    if pilot_acquisition is not None:
        SQLitePilotEvidenceStore(sqlite_ledger_path).record_run(
            run_id=run_id,
            acquisition=pilot_acquisition,
            exact_costs=exact_list,
            expected_request_ids=[trace.request_id for trace in trace_list],
        )
    return PersistedBenchmarkRun(report=report, segments=segments)


def _validate_optional_pilot_evidence(
    *,
    traces: Sequence[RequestTrace],
    acquisition: AcquisitionSemantics | None,
    exact_costs: Sequence[ExactRequestCostEvidence],
) -> None:
    if acquisition is None and exact_costs:
        raise ValueError("pilot exact monetary evidence requires explicit acquisition semantics")
    if acquisition is not None and not exact_costs and traces:
        raise ValueError("pilot acquisition semantics require exact monetary evidence for every request")
    if acquisition is None:
        return

    exact_by_request = {item.request_id: item for item in exact_costs}
    if len(exact_by_request) != len(exact_costs):
        raise ValueError("pilot exact monetary evidence contains duplicate request ids")
    trace_ids = {trace.request_id for trace in traces}
    if set(exact_by_request) != trace_ids:
        raise ValueError(
            "pilot exact monetary evidence requires exact trace coverage; "
            f"missing={sorted(trace_ids - set(exact_by_request))}, "
            f"unexpected={sorted(set(exact_by_request) - trace_ids)}"
        )
    for trace in traces:
        exact = exact_by_request[trace.request_id]
        if len(exact.attempts) != trace.provider_attempt_count:
            raise ValueError(
                f"pilot exact attempt count disagrees with trace for request {trace.request_id}"
            )
        if trace.provider_attempts and len(trace.provider_attempts) != len(exact.attempts):
            raise ValueError(
                f"pilot exact attempt chain disagrees with canonical attempts for request {trace.request_id}"
            )


def _require_exact_context_trace_coverage(
    contexts: Sequence[BenchmarkRequestContext], traces: Sequence[RequestTrace]
) -> None:
    context_ids = [context.request_id for context in contexts]
    trace_ids = [trace.request_id for trace in traces]
    if len(context_ids) != len(set(context_ids)):
        raise ValueError("benchmark request contexts contain duplicate request_id values")
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("benchmark traces contain duplicate request_id values")
    if len(context_ids) != len(trace_ids) or set(context_ids) != set(trace_ids):
        raise ValueError(
            "benchmark run persistence requires exact request-context/trace coverage; "
            f"context_only={sorted(set(context_ids) - set(trace_ids))}, "
            f"trace_only={sorted(set(trace_ids) - set(context_ids))}"
        )
