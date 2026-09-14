from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from ..infrastructure.telemetry.request_log import RequestTrace, RouteTrace
from .harness import BenchmarkReport
from .reconciliation import reconcile_run_costs
from .sqlite_ledger import ProviderUsageSummary


def export_run_json(
    *,
    run_id: str,
    report: BenchmarkReport,
    traces: list[RequestTrace],
    routes: list[RouteTrace],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reconciliation = reconcile_run_costs(routes=routes, executions=traces)
    payload = {
        "run_id": run_id,
        "report": asdict(report),
        "cost_reconciliation": asdict(reconciliation),
        "traces": [asdict(trace) for trace in traces],
        "routes": [asdict(route) for route in routes],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def export_run_markdown(
    *,
    run_id: str,
    report: BenchmarkReport,
    traces: list[RequestTrace],
    routes: list[RouteTrace],
    output_path: Path,
    provider_usage_summary: ProviderUsageSummary | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reconciliation = reconcile_run_costs(routes=routes, executions=traces)
    lines = [
        f"# Benchmark Run `{run_id}`",
        "",
        "## Summary",
        "",
        f"- Workload: `{report.workload_path}`",
        f"- Workload SHA256: `{report.workload_sha256 or 'unavailable'}`",
        f"- Strategy: `{report.strategy}`",
        f"- Provider: `{report.provider}`",
        f"- Model profile: `{report.model}`",
        f"- Requests: {report.request_count}",
        f"- Successes: {report.success_count}",
        f"- Failures: {report.failure_count}",
        f"- Error rate: {_format_rate(report.error_rate)}",
        f"- Latency p50: {report.latency_p50_ms} ms",
        f"- Latency p95: {report.latency_p95_ms} ms",
        f"- Prompt tokens: {report.prompt_tokens}",
        f"- Completion tokens: {report.completion_tokens}",
        f"- Total tokens: {report.total_tokens}",
        f"- Execution cost: {_format_optional_cost(report.estimated_cost_usd)}",
        f"- Cost evidence complete: {report.cost_evidence_complete}",
        f"- Provider attempts: {report.provider_attempt_count}",
        f"- Provider retries: {report.provider_retry_count}",
        f"- Quality pass rate: {_format_optional_rate(report.quality_pass_rate)}",
        f"- Quality score average: {_format_optional_float(report.quality_score_avg)}",
        f"- Route decisions: {report.route_count}",
        f"- Budget violations: {report.budget_violation_count}",
        "",
        "## Model Distribution",
        "",
    ]
    if report.model_distribution:
        for model, count in report.model_distribution.items():
            lines.append(f"- `{model}`: {count}")
    else:
        lines.append("No successful model calls were recorded.")

    lines.extend(["", "## Observed Latency By Model", ""])
    if report.observed_latency_ms_by_model:
        lines.extend(["| Model | Count | p50 | p95 |", "| --- | ---: | ---: | ---: |"])
        for model, profile in report.observed_latency_ms_by_model.items():
            lines.append(
                f"| `{model}` | {profile['count']} | {profile['p50']} ms | {profile['p95']} ms |"
            )
    else:
        lines.append("No successful latency profiles were recorded.")

    lines.extend(["", "## Route Reason Distribution", ""])
    if report.route_reason_distribution:
        for reason, count in report.route_reason_distribution.items():
            lines.append(f"- {count} x {_escape_table(reason)}")
    else:
        lines.append("No route reasons were recorded.")

    lines.extend(["", "## Provider Usage Summary", ""])
    if provider_usage_summary is not None:
        lines.extend(
            [
                f"- Requests: {provider_usage_summary.request_count}",
                f"- Successes: {provider_usage_summary.success_count}",
                f"- Failures: {provider_usage_summary.failure_count}",
                f"- Prompt tokens: {provider_usage_summary.prompt_tokens}",
                f"- Completion tokens: {provider_usage_summary.completion_tokens}",
                f"- Total tokens: {provider_usage_summary.total_tokens}",
                f"- Execution cost: {_format_optional_cost(provider_usage_summary.estimated_cost_usd)}",
                f"- Cost evidence complete: {provider_usage_summary.cost_evidence_complete}",
                f"- Provider attempts: {provider_usage_summary.provider_attempt_count}",
                f"- Provider retries: {provider_usage_summary.provider_retry_count}",
                "",
                "### Cost By Model",
                "",
            ]
        )
        if provider_usage_summary.cost_by_model:
            lines.extend(["| Model | Cost |", "| --- | ---: |"])
            for model, cost in provider_usage_summary.cost_by_model.items():
                lines.append(f"| `{model}` | {_format_optional_cost(cost)} |")
        else:
            lines.append("No provider cost was recorded.")

        lines.extend(["", "### Tokens By Model", ""])
        if provider_usage_summary.tokens_by_model:
            lines.extend(["| Model | Tokens |", "| --- | ---: |"])
            for model, tokens in provider_usage_summary.tokens_by_model.items():
                lines.append(f"| `{model}` | {tokens} |")
        else:
            lines.append("No provider tokens were recorded.")
    else:
        lines.append("No provider usage summary was available for this export.")

    lines.extend(
        [
            "",
            "## Route Estimate vs Observed Execution",
            "",
            f"- Request universe: {reconciliation.request_count}",
            f"- Paired requests: {reconciliation.paired_request_count}",
            f"- Comparable requests: {reconciliation.comparable_request_count}",
            f"- Comparable coverage: {_format_rate(reconciliation.comparable_coverage)} (of request universe)",
            f"- Comparable successes: {reconciliation.comparable_success_count}",
            f"- Comparable failures: {reconciliation.comparable_failure_count}",
            f"- Not executed: {reconciliation.not_executed_count}",
            f"- Route cost incomplete: {reconciliation.route_cost_incomplete_count}",
            f"- Execution cost incomplete: {reconciliation.execution_cost_incomplete_count}",
            f"- Missing route: {reconciliation.missing_route_count}",
            f"- Missing execution: {reconciliation.missing_execution_count}",
            f"- Execution-path divergences among comparable requests: {reconciliation.execution_path_divergence_count}",
            f"- Execution-path divergence rate among comparable requests: {_format_optional_rate(reconciliation.execution_path_divergence_rate)}",
            f"- Comparable route estimate: {_format_optional_cost(reconciliation.comparable_route_estimated_cost_usd)}",
            f"- Comparable observed execution cost: {_format_optional_cost(reconciliation.comparable_observed_execution_cost_usd)}",
            f"- Observed minus route estimate: {_format_optional_signed_cost(reconciliation.comparable_cost_delta_usd)}",
            f"- Aggregate cost delta: {_format_optional_percent(reconciliation.comparable_cost_delta_percent)}",
            f"- Mean absolute cost deviation: {_format_optional_cost(reconciliation.mean_absolute_cost_deviation_usd)}",
            f"- Median absolute cost deviation: {_format_optional_cost(reconciliation.median_absolute_cost_deviation_usd)}",
            f"- p95 absolute cost deviation: {_format_optional_cost(reconciliation.p95_absolute_cost_deviation_usd)}",
            f"- Underestimation rate: {_format_optional_rate(reconciliation.underestimation_rate)}",
            f"- Overestimation rate: {_format_optional_rate(reconciliation.overestimation_rate)}",
            f"- Matched rate: {_format_optional_rate(reconciliation.matched_rate)}",
            f"- Retry-amplification eligible successful requests: {reconciliation.retry_amplification_eligible_request_count}",
            f"- Known non-final attempt cost within amplification-eligible requests: {_format_optional_cost(reconciliation.non_final_attempt_cost_usd)}",
            f"- Retry amplification share of amplification-eligible execution cost: {_format_optional_rate(reconciliation.retry_amplification_share)}",
            "",
        ]
    )
    if reconciliation.request_reconciliations:
        lines.extend(
            [
                "| Request | Status | Route model | Execution model | Route estimate | Observed execution | Delta | Relative delta | Attempts | Retries | Path diverged |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in reconciliation.request_reconciliations:
            lines.append(
                "| "
                f"`{item.request_id}` | "
                f"`{item.status.value}` | "
                f"`{item.route_model or 'n/a'}` | "
                f"`{item.execution_model or 'n/a'}` | "
                f"{_format_optional_cost(item.route_estimated_cost_usd)} | "
                f"{_format_optional_cost(item.observed_execution_cost_usd)} | "
                f"{_format_optional_signed_cost(item.cost_delta_usd)} | "
                f"{_format_optional_percent(item.relative_cost_delta_percent)} | "
                f"{item.provider_attempt_count} | "
                f"{item.provider_retry_count} | "
                f"{item.execution_path_diverged} |"
            )
    else:
        lines.append("No route/execution pairs were available for reconciliation.")

    lines.extend(["", "## Route Decisions", ""])
    if routes:
        lines.extend(
            [
                "| Request | Strategy | Selected model | Estimated route cost | Cost evidence | Budget violation | Reason |",
                "| --- | --- | --- | ---: | --- | --- | --- |",
            ]
        )
        for route in routes:
            lines.append(
                "| "
                f"`{route.request_id}` | "
                f"`{route.strategy}` | "
                f"`{route.selected_model}` | "
                f"{_format_optional_cost(route.estimated_cost_usd)} | "
                f"{route.cost_evidence_complete} | "
                f"{route.budget_violation} | "
                f"{_escape_table(route.decision_reason)} |"
            )
            if route.cost_quote is not None:
                quote = route.cost_quote
                lines.extend(
                    [
                        "",
                        f"Pricing evidence for `{route.request_id}`:",
                        f"- Pricing record: `{quote.pricing_record_id}`",
                        f"- Pricing table: `{quote.pricing_table_version}`",
                        f"- Pricing observed at: `{quote.pricing_observed_at.isoformat()}`",
                        f"- Pricing source: `{quote.pricing_source_url}`",
                        f"- Route token assumptions: input={quote.input_tokens}, output={quote.output_tokens}, cached_input={quote.cached_input_tokens}",
                        f"- Rate snapshot per 1M tokens: input={quote.input_per_million}, output={quote.output_per_million}, cached_input={quote.cached_input_per_million if quote.cached_input_per_million is not None else 'n/a'}",
                        "",
                    ]
                )
    else:
        lines.append("No route decisions were recorded.")

    lines.extend(["", "## Request Outcomes", ""])
    if traces:
        lines.extend(
            [
                "| Request | Model | Latency | Tokens | Execution cost | Cost complete | Attempts | Error | Quality |",
                "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
            ]
        )
        for trace in traces:
            quality = (
                "n/a"
                if trace.quality_passed is None
                else f"{trace.quality_passed} ({_format_optional_float(trace.quality_score)})"
            )
            lines.append(
                "| "
                f"`{trace.request_id}` | "
                f"`{trace.model}` | "
                f"{trace.latency_ms} ms | "
                f"{trace.total_tokens} | "
                f"{_format_optional_cost(trace.estimated_cost_usd)} | "
                f"{trace.cost_evidence_complete} | "
                f"{trace.provider_attempt_count} | "
                f"{trace.error_type or 'none'} | "
                f"{quality} |"
            )
    else:
        lines.append("No request traces were recorded.")

    lines.extend(["", "## Limitations", ""])
    for limitation in report.limitations:
        lines.append(f"- {limitation}")
    lines.append("- Route cost is defensible only when route cost evidence is complete.")
    lines.append(
        "- Cost-delta and deviation metrics use only paired requests with complete route and execution cost evidence and at least one provider attempt; comparable coverage and evidence-gap counts use the full logical request universe."
    )
    lines.append(
        "- Retry amplification is reported only for successful comparable executions whose final-attempt cost is reconstructable; its denominator is the execution cost of that eligible population, not every request in the run."
    )
    lines.append("- This export is evidence for one run only; compare runs before discussing cost deltas.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _json_default(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _format_rate(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_optional_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return _format_rate(value)


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_optional_cost(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"${value:.8f}"


def _format_optional_signed_cost(value: float | None) -> str:
    if value is None:
        return "unknown"
    sign = "+" if value > 0 else ""
    return f"{sign}${value:.8f}" if value >= 0 else f"-${abs(value):.8f}"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
