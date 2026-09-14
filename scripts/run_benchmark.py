from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from inference_engine.benchmarking.budget import BudgetViolation, enforce_estimated_cost_budget
from inference_engine.benchmarking.context_store import SQLiteBenchmarkContextStore
from inference_engine.benchmarking.eval import evaluate_text
from inference_engine.benchmarking.export import export_run_json, export_run_markdown
from inference_engine.benchmarking.harness import (
    compare_reports,
    load_workload,
    summarize_traces,
    write_comparison,
    write_report,
)
from inference_engine.benchmarking.segmentation import (
    BenchmarkRequestContext,
    summarize_segments,
)
from inference_engine.benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from inference_engine.domain.cost.routing_estimator import ProviderPricingCostEstimator
from inference_engine.domain.models.request import InferenceRequest, ModelParameters
from inference_engine.domain.models.routing import ModelConfig, ModelTier, RoutingStrategy
from inference_engine.domain.routing.baseline import BaselineRouter
from inference_engine.domain.routing.complexity import ComplexityEstimator
from inference_engine.domain.routing.policy import PolicyRouter, PolicyRouterConfig
from inference_engine.infrastructure.models.errors import ProviderError, classify_openai_error
from inference_engine.infrastructure.models.openai_backend import OpenAIBackend, RetryPolicy
from inference_engine.infrastructure.telemetry.request_log import (
    JsonlRequestLog,
    JsonlRouteLog,
    RequestTrace,
    RouteTrace,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_benchmark")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run one benchmark strategy")
    _add_run_arguments(run_parser)

    compare_parser = subparsers.add_parser("compare", help="Compare two stored benchmark runs")
    compare_parser.add_argument("--sqlite-ledger-path", default="reports/benchmarks/ledger.sqlite3")
    compare_parser.add_argument("--baseline-run-id", required=True)
    compare_parser.add_argument("--candidate-run-id", required=True)
    compare_parser.add_argument("--comparison-path", default="reports/benchmarks/latest-comparison.json")

    export_parser = subparsers.add_parser("export", help="Export one stored benchmark run")
    export_parser.add_argument("--sqlite-ledger-path", default="reports/benchmarks/ledger.sqlite3")
    export_parser.add_argument("--run-id", required=True)
    export_parser.add_argument("--output-dir", default="reports/benchmarks/exports")
    export_parser.add_argument("--format", choices=["json", "markdown", "both"], default="both")

    usage_parser = subparsers.add_parser("usage-summary", help="Summarize stored provider usage")
    usage_parser.add_argument("--sqlite-ledger-path", default="reports/benchmarks/ledger.sqlite3")
    usage_parser.add_argument("--run-id", required=True)
    usage_parser.add_argument("--output-path", default=None)

    _add_run_arguments(parser)
    parser.set_defaults(command="run")
    args = parser.parse_args()
    if args.command == "compare":
        return _compare(args)
    if args.command == "export":
        return _export(args)
    if args.command == "usage-summary":
        return _usage_summary(args)
    return asyncio.run(_run(args))


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workload", default="benchmarks/workloads/smoke.jsonl")
    parser.add_argument(
        "--provider",
        choices=["openai"],
        default="openai",
        help="Transport adapter. Billing and execution identity are configured separately.",
    )
    parser.add_argument("--execution-provider", default=None)
    parser.add_argument("--pricing-provider", default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--strategy",
        choices=["single_model", "rule_based", "policy"],
        default="single_model",
    )
    parser.add_argument("--economy-model", default="gpt-4o-mini")
    parser.add_argument("--standard-model", default="gpt-4o-mini")
    parser.add_argument("--premium-model", default="gpt-4o")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ledger-path", default="reports/benchmarks/latest-ledger.jsonl")
    parser.add_argument("--sqlite-ledger-path", default="reports/benchmarks/ledger.sqlite3")
    parser.add_argument("--report-path", default="reports/benchmarks/latest-report.json")
    parser.add_argument(
        "--segment-report-path",
        default="reports/benchmarks/latest-segments.json",
    )
    parser.add_argument("--route-ledger-path", default="reports/benchmarks/latest-routes.jsonl")
    parser.add_argument("--max-estimated-cost-usd", type=float, default=None)
    parser.add_argument("--policy-latency-slo-ms", type=int, default=None)
    parser.add_argument("--policy-min-quality-score", type=float, default=None)
    parser.add_argument("--policy-cost-weight", type=float, default=0.55)
    parser.add_argument("--policy-latency-weight", type=float, default=0.25)
    parser.add_argument("--policy-quality-weight", type=float, default=0.20)
    parser.add_argument("--run-id", default=None)


async def _run(args: argparse.Namespace) -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required for the openai-compatible benchmark transport")
        return 2

    execution_provider = _resolve_execution_provider(args)
    pricing_provider = _resolve_pricing_provider(args)
    if execution_provider != pricing_provider:
        raise ValueError(
            "benchmark execution_provider and pricing_provider must match until explicit billing mappings exist"
        )

    workload_path = Path(args.workload)
    ledger_path = Path(args.ledger_path)
    sqlite_ledger_path = Path(args.sqlite_ledger_path)
    report_path = Path(args.report_path)
    segment_report_path = Path(
        getattr(
            args,
            "segment_report_path",
            str(report_path.with_name(f"{report_path.stem}-segments.json")),
        )
    )
    run_id = args.run_id or f"{args.strategy}-{uuid4().hex[:12]}"
    workload = load_workload(workload_path)
    request_log = JsonlRequestLog(ledger_path)
    route_log = JsonlRouteLog(Path(args.route_ledger_path))
    sqlite_ledger = SQLiteBenchmarkLedger(sqlite_ledger_path)
    context_store = SQLiteBenchmarkContextStore(sqlite_ledger_path)
    router = _build_router(args)
    backends: dict[str, OpenAIBackend] = {}

    def backend_for(model_name: str) -> OpenAIBackend:
        if model_name not in backends:
            backends[model_name] = OpenAIBackend(
                api_key=api_key,
                model_name=model_name,
                base_url=args.base_url,
                timeout_seconds=args.timeout_seconds,
                retry_policy=RetryPolicy(max_attempts=2),
                provider_name=execution_provider,
                pricing_provider=pricing_provider,
            )
        return backends[model_name]

    report_model = (
        args.model
        if args.strategy == RoutingStrategy.SINGLE_MODEL.value
        else ",".join([args.economy_model, args.standard_model, args.premium_model])
    )

    traces: list[RequestTrace] = []
    route_traces: list[RouteTrace] = []
    request_contexts: list[BenchmarkRequestContext] = []
    for item in workload:
        request = InferenceRequest(
            prompt=item.prompt,
            parameters=ModelParameters(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ),
        )
        request_contexts.append(
            BenchmarkRequestContext.from_tags(
                request_id=str(request.id),
                workload_item_id=item.id,
                tags=item.tags,
            )
        )
        decision = await router.route(request)
        route_trace = RouteTrace.from_decision(
            decision,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
        )
        selected_model = decision.selected_model.id
        started = perf_counter()
        try:
            enforce_estimated_cost_budget(decision, args.max_estimated_cost_usd)
            route_log.append(route_trace)
            route_traces.append(route_trace)
            response = await backend_for(selected_model).infer(request)
            trace = RequestTrace.from_response(provider=execution_provider, response=response)
            eval_result = evaluate_text(response.text, item.eval_spec)
            if eval_result is not None:
                trace = replace(
                    trace,
                    quality_passed=eval_result.passed,
                    quality_score=eval_result.score,
                    quality_reason=eval_result.reason,
                    eval_type=eval_result.eval_type,
                )
        except BudgetViolation as exc:
            route_trace = RouteTrace.from_decision(
                decision,
                max_estimated_cost_usd=args.max_estimated_cost_usd,
                budget_violation_reason=exc.message,
            )
            route_log.append(route_trace)
            route_traces.append(route_trace)
            trace = RequestTrace(
                request_id=str(request.id),
                provider=execution_provider,
                model=selected_model,
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                pricing_table_version="not_charged",
                cache_hit=False,
                error_type="budget_violation",
                error_message=exc.message,
                timestamp=route_trace.timestamp,
                provider_attempt_count=0,
                provider_retry_count=0,
                cost_evidence_complete=True,
            )
        except ProviderError as exc:
            trace = RequestTrace.from_error(
                request_id=request.id,
                provider=execution_provider,
                model=selected_model,
                latency_ms=int((perf_counter() - started) * 1000),
                error=exc,
            )
        except Exception as exc:
            trace = RequestTrace.from_error(
                request_id=request.id,
                provider=execution_provider,
                model=selected_model,
                latency_ms=int((perf_counter() - started) * 1000),
                error=replace(classify_openai_error(exc), provider=execution_provider),
            )
        request_log.append(trace)
        traces.append(trace)

    report = summarize_traces(
        workload_path=workload_path,
        strategy=args.strategy,
        provider=execution_provider,
        model=report_model,
        ledger_path=ledger_path,
        traces=traces,
        route_traces=route_traces,
    )
    segment_evidence = summarize_segments(
        request_contexts=request_contexts,
        traces=traces,
        routes=route_traces,
    )
    write_report(report, report_path)
    segment_report_path.parent.mkdir(parents=True, exist_ok=True)
    segment_report_path.write_text(
        json.dumps(asdict(segment_evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sqlite_ledger.record_run(
        run_id=run_id,
        report=report,
        traces=traces,
        route_traces=route_traces,
    )
    context_store.record_contexts(run_id=run_id, contexts=request_contexts)
    print(
        " ".join(
            [
                f"run_id={run_id}",
                f"requests={report.request_count}",
                f"successes={report.success_count}",
                f"failures={report.failure_count}",
                f"budget_violations={report.budget_violation_count}",
                f"latency_p50_ms={report.latency_p50_ms}",
                f"latency_p95_ms={report.latency_p95_ms}",
                f"execution_cost_usd={_format_optional_cost_value(report.estimated_cost_usd)}",
                f"cost_evidence_complete={str(report.cost_evidence_complete).lower()}",
                f"segments={segment_evidence.segment_count}",
                f"report_path={report_path}",
                f"segment_report_path={segment_report_path}",
                f"sqlite_ledger_path={sqlite_ledger_path}",
            ]
        )
    )
    return 0 if report.failure_count == 0 else 1


def _compare(args: argparse.Namespace) -> int:
    sqlite_ledger = SQLiteBenchmarkLedger(Path(args.sqlite_ledger_path))
    baseline = sqlite_ledger.get_report(args.baseline_run_id)
    candidate = sqlite_ledger.get_report(args.candidate_run_id)
    comparison = compare_reports(
        baseline_run_id=args.baseline_run_id,
        baseline=baseline,
        candidate_run_id=args.candidate_run_id,
        candidate=candidate,
    )
    write_comparison(comparison, Path(args.comparison_path))
    print(
        " ".join(
            [
                f"comparable={str(comparison.comparable).lower()}",
                f"baseline_run_id={comparison.baseline_run_id}",
                f"candidate_run_id={comparison.candidate_run_id}",
                f"cost_delta_usd={_format_optional_cost_value(comparison.cost_delta_usd)}",
                f"cost_evidence_complete={str(comparison.cost_evidence_complete).lower()}",
                f"latency_p95_delta_ms={comparison.latency_p95_delta_ms}",
                f"comparison_path={args.comparison_path}",
            ]
        )
    )
    return 0 if comparison.comparable else 1


def _export(args: argparse.Namespace) -> int:
    sqlite_ledger_path = Path(args.sqlite_ledger_path)
    sqlite_ledger = SQLiteBenchmarkLedger(sqlite_ledger_path)
    report = sqlite_ledger.get_report(args.run_id)
    traces = sqlite_ledger.get_traces(args.run_id)
    routes = sqlite_ledger.get_routes(args.run_id)
    provider_usage_summary = sqlite_ledger.get_provider_usage_summary(args.run_id)
    output_dir = Path(args.output_dir)

    written: list[Path] = []
    if args.format in {"json", "both"}:
        path = output_dir / f"{args.run_id}.json"
        export_run_json(run_id=args.run_id, report=report, traces=traces, routes=routes, output_path=path)
        written.append(path)
    if args.format in {"markdown", "both"}:
        path = output_dir / f"{args.run_id}.md"
        export_run_markdown(
            run_id=args.run_id,
            report=report,
            traces=traces,
            routes=routes,
            output_path=path,
            provider_usage_summary=provider_usage_summary,
        )
        written.append(path)

    contexts = SQLiteBenchmarkContextStore(sqlite_ledger_path).get_contexts(args.run_id)
    segment_evidence = summarize_segments(
        request_contexts=contexts,
        traces=traces,
        routes=routes,
    )
    segment_path = output_dir / f"{args.run_id}.segments.json"
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    segment_path.write_text(
        json.dumps(asdict(segment_evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written.append(segment_path)

    print(" ".join([f"run_id={args.run_id}", *[f"written={path}" for path in written]]))
    return 0


def _usage_summary(args: argparse.Namespace) -> int:
    sqlite_ledger = SQLiteBenchmarkLedger(Path(args.sqlite_ledger_path))
    summary = sqlite_ledger.get_provider_usage_summary(args.run_id)
    payload = asdict(summary)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        " ".join(
            [
                f"run_id={summary.run_id}",
                f"requests={summary.request_count}",
                f"successes={summary.success_count}",
                f"failures={summary.failure_count}",
                f"total_tokens={summary.total_tokens}",
                f"execution_cost_usd={_format_optional_cost_value(summary.estimated_cost_usd)}",
                f"cost_evidence_complete={str(summary.cost_evidence_complete).lower()}",
                f"provider_attempts={summary.provider_attempt_count}",
                f"provider_retries={summary.provider_retry_count}",
            ]
        )
    )
    return 0


def _build_router(args: argparse.Namespace) -> BaselineRouter | PolicyRouter:
    strategy = RoutingStrategy(args.strategy)
    pricing_provider = _resolve_pricing_provider(args)
    cost_estimator = ProviderPricingCostEstimator(provider=pricing_provider)

    if strategy == RoutingStrategy.SINGLE_MODEL:
        model = _model_config(args.model, ModelTier.STANDARD, cost_estimator)
        return BaselineRouter(
            [model],
            ComplexityEstimator(),
            cost_estimator,
            mode=strategy,
            single_model_id=args.model,
        )

    models = [
        _model_config(args.economy_model, ModelTier.ECONOMY, cost_estimator),
        _model_config(args.standard_model, ModelTier.STANDARD, cost_estimator),
        _model_config(args.premium_model, ModelTier.PREMIUM, cost_estimator),
    ]
    if strategy == RoutingStrategy.POLICY:
        return PolicyRouter(
            models,
            ComplexityEstimator(),
            cost_estimator,
            PolicyRouterConfig(
                max_estimated_cost_usd=args.max_estimated_cost_usd,
                latency_slo_ms=args.policy_latency_slo_ms,
                min_quality_score=args.policy_min_quality_score,
                cost_weight=args.policy_cost_weight,
                latency_weight=args.policy_latency_weight,
                quality_weight=args.policy_quality_weight,
            ),
        )
    return BaselineRouter(
        models,
        ComplexityEstimator(),
        cost_estimator,
        mode=strategy,
        single_model_id=args.model if strategy == RoutingStrategy.SINGLE_MODEL else None,
    )


def _model_config(
    model_name: str,
    tier: ModelTier,
    cost_estimator: ProviderPricingCostEstimator,
) -> ModelConfig:
    cost_estimator.validate_model(model_name)
    return ModelConfig(
        id=model_name,
        name=model_name,
        tier=tier,
        max_context_length=128_000,
    )


def _resolve_execution_provider(args: argparse.Namespace) -> str:
    explicit = getattr(args, "execution_provider", None)
    if explicit:
        return str(explicit)
    if getattr(args, "base_url", None) is None:
        return "openai"
    raise ValueError(
        "custom --base-url benchmark requires --execution-provider; protocol compatibility is not provider identity"
    )


def _resolve_pricing_provider(args: argparse.Namespace) -> str:
    explicit = getattr(args, "pricing_provider", None)
    if explicit:
        return str(explicit)
    if getattr(args, "base_url", None) is None:
        return "openai"
    raise ValueError(
        "custom --base-url benchmark requires --pricing-provider; InferenceLedger will not guess billing identity"
    )


def _format_optional_cost_value(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.8f}"


if __name__ == "__main__":
    raise SystemExit(main())
