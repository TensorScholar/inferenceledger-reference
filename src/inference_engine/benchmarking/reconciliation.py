from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isclose
from statistics import median
from typing import TypeVar

from ..domain.models.execution import AttemptOutcome
from ..infrastructure.telemetry.request_log import RequestTrace, RouteTrace

TTrace = TypeVar("TTrace", RouteTrace, RequestTrace)


class ReconciliationStatus(StrEnum):
    """Whether one route/execution pair supports a defensible monetary comparison."""

    COMPARABLE_SUCCESS = "comparable_success"
    COMPARABLE_FAILURE = "comparable_failure"
    NOT_EXECUTED = "not_executed"
    ROUTE_COST_INCOMPLETE = "route_cost_incomplete"
    EXECUTION_COST_INCOMPLETE = "execution_cost_incomplete"
    MISSING_ROUTE = "missing_route"
    MISSING_EXECUTION = "missing_execution"


class CostDeviationDirection(StrEnum):
    """Direction of observed execution cost relative to the route estimate."""

    UNDERESTIMATED = "underestimated"
    OVERESTIMATED = "overestimated"
    MATCHED = "matched"
    NOT_COMPARABLE = "not_comparable"


@dataclass(frozen=True)
class RequestCostReconciliation:
    """Derived economic evidence for one routed request and its observed execution chain."""

    request_id: str
    status: ReconciliationStatus
    route_model: str | None
    execution_model: str | None
    execution_path: tuple[str, ...]
    execution_path_diverged: bool
    route_estimated_cost_usd: float | None
    observed_execution_cost_usd: float | None
    cost_delta_usd: float | None
    absolute_cost_deviation_usd: float | None
    relative_cost_delta_percent: float | None
    execution_to_route_ratio: float | None
    deviation_direction: CostDeviationDirection
    provider_attempt_count: int
    provider_retry_count: int
    successful_final_attempt_cost_usd: float | None
    non_final_attempt_cost_usd: float | None
    retry_amplification_ratio: float | None
    execution_succeeded: bool | None
    error_type: str | None

    @property
    def comparable(self) -> bool:
        return self.status in {
            ReconciliationStatus.COMPARABLE_SUCCESS,
            ReconciliationStatus.COMPARABLE_FAILURE,
        }


@dataclass(frozen=True)
class RunCostReconciliation:
    """Run-level estimate-versus-observed economics derived from request reconciliation evidence."""

    request_count: int
    paired_request_count: int
    comparable_request_count: int
    comparable_coverage: float
    comparable_success_count: int
    comparable_failure_count: int
    not_executed_count: int
    route_cost_incomplete_count: int
    execution_cost_incomplete_count: int
    missing_route_count: int
    missing_execution_count: int
    execution_path_divergence_count: int
    execution_path_divergence_rate: float | None
    comparable_route_estimated_cost_usd: float | None
    comparable_observed_execution_cost_usd: float | None
    comparable_cost_delta_usd: float | None
    comparable_cost_delta_percent: float | None
    mean_absolute_cost_deviation_usd: float | None
    median_absolute_cost_deviation_usd: float | None
    p95_absolute_cost_deviation_usd: float | None
    underestimation_rate: float | None
    overestimation_rate: float | None
    matched_rate: float | None
    retry_amplification_eligible_request_count: int
    non_final_attempt_cost_usd: float | None
    retry_amplification_share: float | None
    request_reconciliations: tuple[RequestCostReconciliation, ...]


def reconcile_request_cost(
    *,
    route: RouteTrace | None,
    execution: RequestTrace | None,
) -> RequestCostReconciliation:
    """Pair one route with one execution and fail closed when either monetary side is incomplete."""
    if route is None and execution is None:
        raise ValueError("route and execution cannot both be missing")

    if route is not None:
        request_id = route.request_id
    else:
        assert execution is not None
        request_id = execution.request_id

    if route is None:
        assert execution is not None
        return _incomplete_reconciliation(
            request_id=request_id,
            status=ReconciliationStatus.MISSING_ROUTE,
            route=None,
            execution=execution,
        )
    if execution is None:
        return _incomplete_reconciliation(
            request_id=request_id,
            status=ReconciliationStatus.MISSING_EXECUTION,
            route=route,
            execution=None,
        )
    if route.request_id != execution.request_id:
        raise ValueError("route and execution request_id must match")

    if not route.cost_evidence_complete or route.estimated_cost_usd is None or route.cost_quote is None:
        return _incomplete_reconciliation(
            request_id=request_id,
            status=ReconciliationStatus.ROUTE_COST_INCOMPLETE,
            route=route,
            execution=execution,
        )

    if execution.provider_attempt_count == 0:
        return _incomplete_reconciliation(
            request_id=request_id,
            status=ReconciliationStatus.NOT_EXECUTED,
            route=route,
            execution=execution,
        )

    if not execution.cost_evidence_complete or execution.estimated_cost_usd is None:
        return _incomplete_reconciliation(
            request_id=request_id,
            status=ReconciliationStatus.EXECUTION_COST_INCOMPLETE,
            route=route,
            execution=execution,
        )

    route_cost = route.estimated_cost_usd
    execution_cost = execution.estimated_cost_usd
    delta = execution_cost - route_cost
    absolute_deviation = abs(delta)
    relative_delta = delta / route_cost * 100 if route_cost > 0 else None
    ratio = execution_cost / route_cost if route_cost > 0 else None
    direction = _deviation_direction(route_cost, execution_cost)
    execution_succeeded = execution.error_type is None
    status = (
        ReconciliationStatus.COMPARABLE_SUCCESS
        if execution_succeeded
        else ReconciliationStatus.COMPARABLE_FAILURE
    )
    final_attempt_cost = _successful_final_attempt_cost(execution)
    non_final_attempt_cost = (
        execution_cost - final_attempt_cost if final_attempt_cost is not None else None
    )
    amplification_ratio = (
        execution_cost / final_attempt_cost
        if final_attempt_cost is not None and final_attempt_cost > 0
        else None
    )
    execution_path = _execution_path(execution)

    return RequestCostReconciliation(
        request_id=request_id,
        status=status,
        route_model=route.selected_model,
        execution_model=execution.model,
        execution_path=execution_path,
        execution_path_diverged=_path_diverged(route, execution, execution_path),
        route_estimated_cost_usd=route_cost,
        observed_execution_cost_usd=execution_cost,
        cost_delta_usd=delta,
        absolute_cost_deviation_usd=absolute_deviation,
        relative_cost_delta_percent=relative_delta,
        execution_to_route_ratio=ratio,
        deviation_direction=direction,
        provider_attempt_count=execution.provider_attempt_count,
        provider_retry_count=execution.provider_retry_count,
        successful_final_attempt_cost_usd=final_attempt_cost,
        non_final_attempt_cost_usd=non_final_attempt_cost,
        retry_amplification_ratio=amplification_ratio,
        execution_succeeded=execution_succeeded,
        error_type=execution.error_type,
    )


def reconcile_run_costs(
    *,
    routes: list[RouteTrace],
    executions: list[RequestTrace],
) -> RunCostReconciliation:
    """Reconcile a run by request id and summarize only comparable monetary evidence."""
    routes_by_id = _unique_by_request_id(routes, kind="route")
    executions_by_id = _unique_by_request_id(executions, kind="execution")
    request_ids = sorted(set(routes_by_id) | set(executions_by_id))
    reconciliations = tuple(
        reconcile_request_cost(
            route=routes_by_id.get(request_id),
            execution=executions_by_id.get(request_id),
        )
        for request_id in request_ids
    )
    comparable = [item for item in reconciliations if item.comparable]
    comparable_count = len(comparable)
    paired_count = sum(
        1
        for request_id in request_ids
        if request_id in routes_by_id and request_id in executions_by_id
    )
    coverage = comparable_count / len(request_ids) if request_ids else 0.0

    route_total: float | None = None
    execution_total: float | None = None
    delta_total: float | None = None
    delta_percent: float | None = None
    absolute_deviations: list[float] = []
    if comparable:
        route_total = sum(_required(item.route_estimated_cost_usd) for item in comparable)
        execution_total = sum(_required(item.observed_execution_cost_usd) for item in comparable)
        delta_total = execution_total - route_total
        if route_total > 0:
            delta_percent = delta_total / route_total * 100
        absolute_deviations = [
            _required(item.absolute_cost_deviation_usd) for item in comparable
        ]

    under = sum(
        1
        for item in comparable
        if item.deviation_direction == CostDeviationDirection.UNDERESTIMATED
    )
    over = sum(
        1
        for item in comparable
        if item.deviation_direction == CostDeviationDirection.OVERESTIMATED
    )
    matched = sum(
        1
        for item in comparable
        if item.deviation_direction == CostDeviationDirection.MATCHED
    )
    path_divergence_count = sum(1 for item in comparable if item.execution_path_diverged)

    amplification_eligible = [
        item
        for item in comparable
        if item.non_final_attempt_cost_usd is not None
        and item.observed_execution_cost_usd is not None
    ]
    non_final_total = (
        sum(_required(item.non_final_attempt_cost_usd) for item in amplification_eligible)
        if amplification_eligible
        else None
    )
    amplification_execution_total = (
        sum(_required(item.observed_execution_cost_usd) for item in amplification_eligible)
        if amplification_eligible
        else None
    )
    retry_share = (
        non_final_total / amplification_execution_total
        if non_final_total is not None
        and amplification_execution_total is not None
        and amplification_execution_total > 0
        else None
    )

    return RunCostReconciliation(
        request_count=len(request_ids),
        paired_request_count=paired_count,
        comparable_request_count=comparable_count,
        comparable_coverage=coverage,
        comparable_success_count=sum(
            1 for item in comparable if item.status == ReconciliationStatus.COMPARABLE_SUCCESS
        ),
        comparable_failure_count=sum(
            1 for item in comparable if item.status == ReconciliationStatus.COMPARABLE_FAILURE
        ),
        not_executed_count=_count_status(reconciliations, ReconciliationStatus.NOT_EXECUTED),
        route_cost_incomplete_count=_count_status(
            reconciliations, ReconciliationStatus.ROUTE_COST_INCOMPLETE
        ),
        execution_cost_incomplete_count=_count_status(
            reconciliations, ReconciliationStatus.EXECUTION_COST_INCOMPLETE
        ),
        missing_route_count=_count_status(reconciliations, ReconciliationStatus.MISSING_ROUTE),
        missing_execution_count=_count_status(
            reconciliations, ReconciliationStatus.MISSING_EXECUTION
        ),
        execution_path_divergence_count=path_divergence_count,
        execution_path_divergence_rate=(
            path_divergence_count / comparable_count if comparable_count else None
        ),
        comparable_route_estimated_cost_usd=route_total,
        comparable_observed_execution_cost_usd=execution_total,
        comparable_cost_delta_usd=delta_total,
        comparable_cost_delta_percent=delta_percent,
        mean_absolute_cost_deviation_usd=(
            sum(absolute_deviations) / len(absolute_deviations)
            if absolute_deviations
            else None
        ),
        median_absolute_cost_deviation_usd=(
            median(absolute_deviations) if absolute_deviations else None
        ),
        p95_absolute_cost_deviation_usd=(
            _percentile(absolute_deviations, 95) if absolute_deviations else None
        ),
        underestimation_rate=under / comparable_count if comparable_count else None,
        overestimation_rate=over / comparable_count if comparable_count else None,
        matched_rate=matched / comparable_count if comparable_count else None,
        retry_amplification_eligible_request_count=len(amplification_eligible),
        non_final_attempt_cost_usd=non_final_total,
        retry_amplification_share=retry_share,
        request_reconciliations=reconciliations,
    )


def _incomplete_reconciliation(
    *,
    request_id: str,
    status: ReconciliationStatus,
    route: RouteTrace | None,
    execution: RequestTrace | None,
) -> RequestCostReconciliation:
    execution_path = _execution_path(execution) if execution is not None else ()
    if execution is None or status == ReconciliationStatus.NOT_EXECUTED:
        execution_succeeded: bool | None = None
    else:
        execution_succeeded = execution.error_type is None
    return RequestCostReconciliation(
        request_id=request_id,
        status=status,
        route_model=route.selected_model if route is not None else None,
        execution_model=execution.model if execution is not None else None,
        execution_path=execution_path,
        execution_path_diverged=(
            _path_diverged(route, execution, execution_path)
            if route is not None and execution is not None
            else False
        ),
        route_estimated_cost_usd=(
            route.estimated_cost_usd
            if route is not None and route.cost_evidence_complete
            else None
        ),
        observed_execution_cost_usd=(
            execution.estimated_cost_usd
            if execution is not None and execution.cost_evidence_complete
            else None
        ),
        cost_delta_usd=None,
        absolute_cost_deviation_usd=None,
        relative_cost_delta_percent=None,
        execution_to_route_ratio=None,
        deviation_direction=CostDeviationDirection.NOT_COMPARABLE,
        provider_attempt_count=execution.provider_attempt_count if execution is not None else 0,
        provider_retry_count=execution.provider_retry_count if execution is not None else 0,
        successful_final_attempt_cost_usd=None,
        non_final_attempt_cost_usd=None,
        retry_amplification_ratio=None,
        execution_succeeded=execution_succeeded,
        error_type=execution.error_type if execution is not None else None,
    )


def _successful_final_attempt_cost(execution: RequestTrace) -> float | None:
    if execution.error_type is not None:
        return None
    if execution.provider_attempts:
        final_attempt = execution.provider_attempts[-1]
        if final_attempt.outcome != AttemptOutcome.SUCCEEDED or not final_attempt.cost_is_known:
            return None
        return final_attempt.cost_usd
    if (
        execution.provider_attempt_count == 1
        and execution.cost_evidence_complete
        and execution.estimated_cost_usd is not None
    ):
        return execution.estimated_cost_usd
    return None


def _execution_path(execution: RequestTrace) -> tuple[str, ...]:
    if execution.provider_attempts:
        return tuple(f"{attempt.provider}/{attempt.model}" for attempt in execution.provider_attempts)
    if execution.provider_attempt_count > 0:
        return (f"{execution.provider}/{execution.model}",)
    return ()


def _path_diverged(
    route: RouteTrace,
    execution: RequestTrace,
    execution_path: tuple[str, ...],
) -> bool:
    if execution.model != route.selected_model:
        return True
    quote = route.cost_quote
    if quote is None:
        return False
    expected = f"{quote.provider}/{route.selected_model}"
    return any(step != expected for step in execution_path)


def _deviation_direction(route_cost: float, execution_cost: float) -> CostDeviationDirection:
    if isclose(route_cost, execution_cost, rel_tol=1e-12, abs_tol=1e-12):
        return CostDeviationDirection.MATCHED
    if execution_cost > route_cost:
        return CostDeviationDirection.UNDERESTIMATED
    return CostDeviationDirection.OVERESTIMATED


def _unique_by_request_id(
    values: list[TTrace],
    *,
    kind: str,
) -> dict[str, TTrace]:
    result: dict[str, TTrace] = {}
    for value in values:
        if value.request_id in result:
            raise ValueError(f"duplicate {kind} request_id: {value.request_id}")
        result[value.request_id] = value
    return result


def _count_status(
    values: tuple[RequestCostReconciliation, ...],
    status: ReconciliationStatus,
) -> int:
    return sum(1 for item in values if item.status == status)


def _required(value: float | None) -> float:
    if value is None:
        raise ValueError("comparable reconciliation unexpectedly has missing monetary evidence")
    return value


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = round((percentile / 100) * (len(ordered) - 1))
    return ordered[index]
