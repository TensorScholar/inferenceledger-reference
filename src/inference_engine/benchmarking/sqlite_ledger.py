from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Any

from ..domain.cost.pricing import PricingQuote
from ..domain.models.execution import AttemptOutcome, CostEvidenceKind, ProviderAttempt
from ..infrastructure.telemetry.request_log import RequestTrace, RouteTrace
from ..utils.time import utc_now
from .harness import BenchmarkReport

SCHEMA_VERSION = 7


@dataclass(frozen=True)
class ProviderUsageRecord:
    """Queryable request-level provider usage with explicit cost completeness."""

    request_id: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    cost_evidence_complete: bool
    pricing_table_version: str
    cache_hit: bool | None
    provider_attempt_count: int
    provider_retry_count: int
    provider_attempts: tuple[ProviderAttempt, ...]
    error_type: str | None
    timestamp: str


@dataclass(frozen=True)
class ProviderUsageSummary:
    """Aggregated provider usage for one benchmark run."""

    run_id: str
    request_count: int
    success_count: int
    failure_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    cost_evidence_complete: bool
    provider_attempt_count: int
    provider_retry_count: int
    cost_by_model: dict[str, float | None]
    tokens_by_model: dict[str, int]


_TRACE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_traces (
    run_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    estimated_cost_usd REAL,
    cost_evidence_complete INTEGER NOT NULL DEFAULT 1,
    pricing_table_version TEXT NOT NULL,
    cache_hit INTEGER,
    error_type TEXT,
    error_message TEXT,
    quality_passed INTEGER,
    quality_score REAL,
    quality_reason TEXT,
    eval_type TEXT,
    provider_attempt_count INTEGER NOT NULL DEFAULT 1,
    provider_retry_count INTEGER NOT NULL DEFAULT 0,
    provider_attempts_json TEXT NOT NULL DEFAULT '[]',
    timestamp TEXT NOT NULL,
    PRIMARY KEY (run_id, request_id),
    FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id) ON DELETE CASCADE
)
"""

_USAGE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_provider_usage (
    run_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    estimated_cost_usd REAL,
    cost_evidence_complete INTEGER NOT NULL DEFAULT 1,
    pricing_table_version TEXT NOT NULL,
    cache_hit INTEGER,
    provider_attempt_count INTEGER NOT NULL,
    provider_retry_count INTEGER NOT NULL,
    provider_attempts_json TEXT NOT NULL DEFAULT '[]',
    error_type TEXT,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (run_id, request_id),
    FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id) ON DELETE CASCADE
)
"""

_ROUTE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_routes (
    run_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    selected_model TEXT NOT NULL,
    estimated_cost_usd REAL NOT NULL,
    estimated_latency_ms INTEGER NOT NULL,
    decision_reason TEXT NOT NULL,
    considered_models_json TEXT NOT NULL,
    fallback_models_json TEXT NOT NULL,
    max_estimated_cost_usd REAL,
    budget_violation INTEGER NOT NULL,
    budget_violation_reason TEXT,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (run_id, request_id),
    FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id) ON DELETE CASCADE
)
"""

_ROUTE_COST_EVIDENCE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_route_cost_evidence (
    run_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL,
    input_per_million REAL NOT NULL,
    output_per_million REAL NOT NULL,
    cached_input_per_million REAL,
    pricing_record_id TEXT NOT NULL,
    pricing_table_version TEXT NOT NULL,
    pricing_observed_at TEXT NOT NULL,
    pricing_source_url TEXT NOT NULL,
    PRIMARY KEY (run_id, request_id),
    FOREIGN KEY (run_id, request_id)
        REFERENCES benchmark_routes(run_id, request_id) ON DELETE CASCADE
)
"""


class SQLiteBenchmarkLedger:
    """Local evidence ledger with fail-closed economic provenance semantics."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    workload_path TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )
                """
            )

            connection.execute(_TRACE_TABLE_SQL)
            _ensure_columns(
                connection,
                table_name="benchmark_traces",
                columns={
                    "quality_passed": "INTEGER",
                    "quality_score": "REAL",
                    "quality_reason": "TEXT",
                    "eval_type": "TEXT",
                    "provider_attempt_count": "INTEGER NOT NULL DEFAULT 1",
                    "provider_retry_count": "INTEGER NOT NULL DEFAULT 0",
                    "cost_evidence_complete": "INTEGER NOT NULL DEFAULT 1",
                    "provider_attempts_json": "TEXT NOT NULL DEFAULT '[]'",
                },
            )
            if _column_is_not_null(connection, "benchmark_traces", "estimated_cost_usd"):
                _rebuild_trace_table_with_nullable_cost(connection)
            if _column_is_not_null(connection, "benchmark_traces", "cache_hit"):
                _rebuild_trace_table_with_nullable_cache_hit(connection)

            connection.execute(_USAGE_TABLE_SQL)
            _ensure_columns(
                connection,
                table_name="benchmark_provider_usage",
                columns={
                    "cost_evidence_complete": "INTEGER NOT NULL DEFAULT 1",
                    "provider_attempts_json": "TEXT NOT NULL DEFAULT '[]'",
                },
            )
            if _column_is_not_null(
                connection,
                "benchmark_provider_usage",
                "estimated_cost_usd",
            ):
                _rebuild_usage_table_with_nullable_cost(connection)
            if _column_is_not_null(connection, "benchmark_provider_usage", "cache_hit"):
                _rebuild_usage_table_with_nullable_cache_hit(connection)

            # Raw route history remains intact across schema upgrades. The v6 evidence table is the
            # trust boundary: a historical numeric route estimate without a matching evidence row is
            # retained for audit history but is never exposed as validated monetary evidence.
            connection.execute(_ROUTE_TABLE_SQL)
            connection.execute(_ROUTE_COST_EVIDENCE_TABLE_SQL)

            connection.execute(
                """
                INSERT OR IGNORE INTO benchmark_provider_usage (
                    run_id,
                    request_id,
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    cost_evidence_complete,
                    pricing_table_version,
                    cache_hit,
                    provider_attempt_count,
                    provider_retry_count,
                    provider_attempts_json,
                    error_type,
                    timestamp
                )
                SELECT
                    run_id,
                    request_id,
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    cost_evidence_complete,
                    pricing_table_version,
                    cache_hit,
                    provider_attempt_count,
                    provider_retry_count,
                    provider_attempts_json,
                    error_type,
                    timestamp
                FROM benchmark_traces
                """
            )
            _downgrade_ambiguous_legacy_costs(connection, "benchmark_traces")
            _downgrade_ambiguous_legacy_costs(connection, "benchmark_provider_usage")
            _normalize_stored_attempt_provenance(connection, "benchmark_traces")
            _normalize_stored_attempt_provenance(connection, "benchmark_provider_usage")
            connection.execute(
                """
                INSERT OR REPLACE INTO ledger_metadata (key, value)
                VALUES ('schema_version', ?)
                """,
                (str(SCHEMA_VERSION),),
            )

    def record_run(
        self,
        *,
        run_id: str,
        report: BenchmarkReport,
        traces: list[RequestTrace],
        route_traces: list[RouteTrace] | None = None,
    ) -> None:
        routes = list(route_traces or [])
        for route in routes:
            _require_complete_route_cost_evidence(route)

        self.initialize()
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                INSERT OR REPLACE INTO benchmark_runs (
                    run_id, created_at, workload_path, strategy, provider, model, report_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    utc_now().isoformat(),
                    report.workload_path,
                    report.strategy,
                    report.provider,
                    report.model,
                    json.dumps(asdict(report), sort_keys=True),
                ),
            )
            connection.execute("DELETE FROM benchmark_traces WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM benchmark_routes WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM benchmark_provider_usage WHERE run_id = ?", (run_id,))
            connection.executemany(
                """
                INSERT INTO benchmark_traces (
                    run_id,
                    request_id,
                    provider,
                    model,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    cost_evidence_complete,
                    pricing_table_version,
                    cache_hit,
                    error_type,
                    error_message,
                    quality_passed,
                    quality_score,
                    quality_reason,
                    eval_type,
                    provider_attempt_count,
                    provider_retry_count,
                    provider_attempts_json,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_trace_storage_row(run_id, trace) for trace in traces],
            )
            connection.executemany(
                """
                INSERT INTO benchmark_provider_usage (
                    run_id,
                    request_id,
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    cost_evidence_complete,
                    pricing_table_version,
                    cache_hit,
                    provider_attempt_count,
                    provider_retry_count,
                    provider_attempts_json,
                    error_type,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_usage_storage_row(run_id, trace) for trace in traces],
            )
            connection.executemany(
                """
                INSERT INTO benchmark_routes (
                    run_id,
                    request_id,
                    strategy,
                    selected_model,
                    estimated_cost_usd,
                    estimated_latency_ms,
                    decision_reason,
                    considered_models_json,
                    fallback_models_json,
                    max_estimated_cost_usd,
                    budget_violation,
                    budget_violation_reason,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_route_storage_row(run_id, route) for route in routes],
            )
            connection.executemany(
                """
                INSERT INTO benchmark_route_cost_evidence (
                    run_id,
                    request_id,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    input_per_million,
                    output_per_million,
                    cached_input_per_million,
                    pricing_record_id,
                    pricing_table_version,
                    pricing_observed_at,
                    pricing_source_url
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_route_evidence_storage_row(run_id, route) for route in routes],
            )

    def get_report(self, run_id: str) -> BenchmarkReport:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM benchmark_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            trace_cost_state = connection.execute(
                """
                SELECT
                    COUNT(*) AS trace_count,
                    SUM(
                        CASE
                            WHEN cost_evidence_complete = 0 OR estimated_cost_usd IS NULL THEN 1
                            ELSE 0
                        END
                    ) AS incomplete_cost_count
                FROM benchmark_traces
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown benchmark run_id: {run_id}")
        raw = json.loads(str(row["report_json"]))
        raw.setdefault("workload_sha256", None)
        raw.setdefault("route_count", 0)
        raw.setdefault("budget_violation_count", 0)
        raw.setdefault("model_distribution", {})
        raw.setdefault("route_reason_distribution", {})
        raw.setdefault("observed_latency_ms_by_model", {})
        raw.setdefault("provider_attempt_count", raw.get("request_count", 0))
        raw.setdefault("provider_retry_count", 0)
        raw.setdefault("quality_count", 0)
        raw.setdefault("quality_pass_count", 0)
        raw.setdefault("quality_pass_rate", None)
        raw.setdefault("quality_score_avg", None)

        trace_count = int(trace_cost_state["trace_count"]) if trace_cost_state is not None else 0
        incomplete_cost_count = (
            int(trace_cost_state["incomplete_cost_count"] or 0)
            if trace_cost_state is not None
            else 0
        )
        if trace_count > 0 and incomplete_cost_count > 0:
            raw["cost_evidence_complete"] = False
            raw["estimated_cost_usd"] = None
            limitations = list(raw.get("limitations", []))
            reconciliation_note = (
                "Stored trace evidence contains unknown execution cost; historical aggregate cost "
                "is suppressed."
            )
            if reconciliation_note not in limitations:
                limitations.append(reconciliation_note)
            raw["limitations"] = limitations
        elif "cost_evidence_complete" not in raw:
            complete = (
                int(raw.get("failure_count", 0)) == 0
                and int(raw.get("provider_retry_count", 0)) == 0
            )
            raw["cost_evidence_complete"] = complete
            if not complete:
                raw["estimated_cost_usd"] = None
        return BenchmarkReport(**raw)

    def get_traces(self, run_id: str) -> list[RequestTrace]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    request_id,
                    provider,
                    model,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    cost_evidence_complete,
                    pricing_table_version,
                    cache_hit,
                    error_type,
                    error_message,
                    quality_passed,
                    quality_score,
                    quality_reason,
                    eval_type,
                    provider_attempt_count,
                    provider_retry_count,
                    provider_attempts_json,
                    timestamp
                FROM benchmark_traces
                WHERE run_id = ?
                ORDER BY timestamp, request_id
                """,
                (run_id,),
            ).fetchall()
        return [_trace_from_row(row) for row in rows]

    def get_provider_usage(self, run_id: str) -> list[ProviderUsageRecord]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    request_id,
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    cost_evidence_complete,
                    pricing_table_version,
                    cache_hit,
                    provider_attempt_count,
                    provider_retry_count,
                    provider_attempts_json,
                    error_type,
                    timestamp
                FROM benchmark_provider_usage
                WHERE run_id = ?
                ORDER BY timestamp, request_id
                """,
                (run_id,),
            ).fetchall()
        return [_usage_from_row(row) for row in rows]

    def get_provider_usage_summary(self, run_id: str) -> ProviderUsageSummary:
        usage = self.get_provider_usage(run_id)
        if not usage:
            self.get_report(run_id)

        cost_evidence_complete = all(
            record.cost_evidence_complete and record.estimated_cost_usd is not None
            for record in usage
        )
        cost_by_model: dict[str, float | None] = {}
        tokens_by_model: dict[str, int] = {}
        for record in usage:
            tokens_by_model[record.model] = tokens_by_model.get(record.model, 0) + record.total_tokens
            if record.model not in cost_by_model:
                cost_by_model[record.model] = 0.0
            if not record.cost_evidence_complete or record.estimated_cost_usd is None:
                cost_by_model[record.model] = None
            elif cost_by_model[record.model] is not None:
                known_cost = cost_by_model[record.model]
                assert known_cost is not None
                cost_by_model[record.model] = known_cost + record.estimated_cost_usd

        return ProviderUsageSummary(
            run_id=run_id,
            request_count=len(usage),
            success_count=sum(1 for record in usage if record.error_type is None),
            failure_count=sum(1 for record in usage if record.error_type is not None),
            prompt_tokens=sum(record.prompt_tokens for record in usage),
            completion_tokens=sum(record.completion_tokens for record in usage),
            total_tokens=sum(record.total_tokens for record in usage),
            estimated_cost_usd=(
                sum(record.estimated_cost_usd or 0.0 for record in usage)
                if cost_evidence_complete
                else None
            ),
            cost_evidence_complete=cost_evidence_complete,
            provider_attempt_count=sum(record.provider_attempt_count for record in usage),
            provider_retry_count=sum(record.provider_retry_count for record in usage),
            cost_by_model=dict(sorted(cost_by_model.items())),
            tokens_by_model=dict(sorted(tokens_by_model.items())),
        )

    def get_routes(self, run_id: str) -> list[RouteTrace]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    r.request_id,
                    r.strategy,
                    r.selected_model,
                    r.estimated_cost_usd AS raw_estimated_cost_usd,
                    r.estimated_latency_ms,
                    r.decision_reason,
                    r.considered_models_json,
                    r.fallback_models_json,
                    r.max_estimated_cost_usd,
                    r.budget_violation,
                    r.budget_violation_reason,
                    r.timestamp,
                    e.provider AS pricing_provider,
                    e.model AS pricing_model,
                    e.input_tokens AS pricing_input_tokens,
                    e.output_tokens AS pricing_output_tokens,
                    e.cached_input_tokens AS pricing_cached_input_tokens,
                    e.input_per_million AS pricing_input_per_million,
                    e.output_per_million AS pricing_output_per_million,
                    e.cached_input_per_million AS pricing_cached_input_per_million,
                    e.pricing_record_id,
                    e.pricing_table_version,
                    e.pricing_observed_at,
                    e.pricing_source_url
                FROM benchmark_routes AS r
                LEFT JOIN benchmark_route_cost_evidence AS e
                  ON e.run_id = r.run_id AND e.request_id = r.request_id
                WHERE r.run_id = ?
                ORDER BY r.timestamp, r.request_id
                """,
                (run_id,),
            ).fetchall()
        return [_route_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


def _trace_storage_row(run_id: str, trace: RequestTrace) -> tuple[object, ...]:
    return (
        run_id,
        trace.request_id,
        trace.provider,
        trace.model,
        trace.latency_ms,
        trace.prompt_tokens,
        trace.completion_tokens,
        trace.total_tokens,
        trace.estimated_cost_usd,
        1 if trace.cost_evidence_complete else 0,
        trace.pricing_table_version,
        _optional_bool_to_int(trace.cache_hit),
        trace.error_type,
        trace.error_message,
        _optional_bool_to_int(trace.quality_passed),
        trace.quality_score,
        trace.quality_reason,
        trace.eval_type,
        trace.provider_attempt_count,
        trace.provider_retry_count,
        _attempts_json(trace.provider_attempts),
        trace.timestamp,
    )


def _usage_storage_row(run_id: str, trace: RequestTrace) -> tuple[object, ...]:
    return (
        run_id,
        trace.request_id,
        trace.provider,
        trace.model,
        trace.prompt_tokens,
        trace.completion_tokens,
        trace.total_tokens,
        trace.estimated_cost_usd,
        1 if trace.cost_evidence_complete else 0,
        trace.pricing_table_version,
        _optional_bool_to_int(trace.cache_hit),
        trace.provider_attempt_count,
        trace.provider_retry_count,
        _attempts_json(trace.provider_attempts),
        trace.error_type,
        trace.timestamp,
    )


def _route_storage_row(run_id: str, route: RouteTrace) -> tuple[object, ...]:
    assert route.estimated_cost_usd is not None
    return (
        run_id,
        route.request_id,
        route.strategy,
        route.selected_model,
        route.estimated_cost_usd,
        route.estimated_latency_ms,
        route.decision_reason,
        json.dumps(route.considered_models, sort_keys=True),
        json.dumps(route.fallback_models, sort_keys=True),
        route.max_estimated_cost_usd,
        1 if route.budget_violation else 0,
        route.budget_violation_reason,
        route.timestamp,
    )


def _route_evidence_storage_row(run_id: str, route: RouteTrace) -> tuple[object, ...]:
    quote = route.cost_quote
    assert quote is not None
    return (
        run_id,
        route.request_id,
        quote.provider,
        quote.model,
        quote.input_tokens,
        quote.output_tokens,
        quote.cached_input_tokens,
        quote.input_per_million,
        quote.output_per_million,
        quote.cached_input_per_million,
        quote.pricing_record_id,
        quote.pricing_table_version,
        quote.pricing_observed_at.isoformat(),
        quote.pricing_source_url,
    )


def _require_complete_route_cost_evidence(route: RouteTrace) -> None:
    if (
        not route.cost_evidence_complete
        or route.estimated_cost_usd is None
        or route.cost_quote is None
    ):
        raise ValueError(
            "new SQLite route writes require complete, reconstructable pricing evidence"
        )


def _route_from_row(row: sqlite3.Row) -> RouteTrace:
    request_id = str(row["request_id"])
    strategy = str(row["strategy"])
    selected_model = str(row["selected_model"])
    estimated_latency_ms = int(row["estimated_latency_ms"])
    decision_reason = str(row["decision_reason"])
    considered_models = [
        str(item) for item in json.loads(str(row["considered_models_json"]))
    ]
    fallback_models = [
        str(item) for item in json.loads(str(row["fallback_models_json"]))
    ]
    max_estimated_cost_usd = _optional_float(row["max_estimated_cost_usd"])
    budget_violation = bool(row["budget_violation"])
    budget_violation_reason = _optional_str(row["budget_violation_reason"])
    timestamp = str(row["timestamp"])

    if row["pricing_record_id"] is None:
        return RouteTrace(
            request_id=request_id,
            strategy=strategy,
            selected_model=selected_model,
            estimated_cost_usd=None,
            estimated_latency_ms=estimated_latency_ms,
            decision_reason=decision_reason,
            considered_models=considered_models,
            fallback_models=fallback_models,
            max_estimated_cost_usd=max_estimated_cost_usd,
            budget_violation=budget_violation,
            budget_violation_reason=budget_violation_reason,
            timestamp=timestamp,
            cost_evidence_complete=False,
            cost_quote=None,
        )

    raw_cost = row["raw_estimated_cost_usd"]
    if raw_cost is None:
        raise ValueError("route pricing evidence exists without stored route cost")
    quote = PricingQuote(
        amount_usd=float(raw_cost),
        provider=str(row["pricing_provider"]),
        model=str(row["pricing_model"]),
        input_tokens=int(row["pricing_input_tokens"]),
        output_tokens=int(row["pricing_output_tokens"]),
        cached_input_tokens=int(row["pricing_cached_input_tokens"]),
        input_per_million=float(row["pricing_input_per_million"]),
        output_per_million=float(row["pricing_output_per_million"]),
        cached_input_per_million=_optional_float(row["pricing_cached_input_per_million"]),
        pricing_record_id=str(row["pricing_record_id"]),
        pricing_table_version=str(row["pricing_table_version"]),
        pricing_observed_at=date.fromisoformat(str(row["pricing_observed_at"])),
        pricing_source_url=str(row["pricing_source_url"]),
    )
    return RouteTrace(
        request_id=request_id,
        strategy=strategy,
        selected_model=selected_model,
        estimated_cost_usd=quote.amount_usd,
        estimated_latency_ms=estimated_latency_ms,
        decision_reason=decision_reason,
        considered_models=considered_models,
        fallback_models=fallback_models,
        max_estimated_cost_usd=max_estimated_cost_usd,
        budget_violation=budget_violation,
        budget_violation_reason=budget_violation_reason,
        timestamp=timestamp,
        cost_evidence_complete=True,
        cost_quote=quote,
    )


def _trace_from_row(row: sqlite3.Row) -> RequestTrace:
    attempts = _attempts_from_json(str(row["provider_attempts_json"]))
    complete = bool(row["cost_evidence_complete"])
    if attempts and not all(attempt.cost_is_known for attempt in attempts):
        complete = False
    return RequestTrace(
        request_id=str(row["request_id"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        latency_ms=int(row["latency_ms"]),
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
        total_tokens=int(row["total_tokens"]),
        estimated_cost_usd=_known_cost_from_row(row, complete),
        cost_evidence_complete=complete,
        pricing_table_version=str(row["pricing_table_version"]),
        cache_hit=_optional_int_to_bool(row["cache_hit"]),
        error_type=_optional_str(row["error_type"]),
        error_message=_optional_str(row["error_message"]),
        timestamp=str(row["timestamp"]),
        quality_passed=_optional_int_to_bool(row["quality_passed"]),
        quality_score=_optional_float(row["quality_score"]),
        quality_reason=_optional_str(row["quality_reason"]),
        eval_type=_optional_str(row["eval_type"]),
        provider_attempt_count=int(row["provider_attempt_count"]),
        provider_retry_count=int(row["provider_retry_count"]),
        provider_attempts=attempts,
    )


def _usage_from_row(row: sqlite3.Row) -> ProviderUsageRecord:
    attempts = _attempts_from_json(str(row["provider_attempts_json"]))
    complete = bool(row["cost_evidence_complete"])
    if attempts and not all(attempt.cost_is_known for attempt in attempts):
        complete = False
    return ProviderUsageRecord(
        request_id=str(row["request_id"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
        total_tokens=int(row["total_tokens"]),
        estimated_cost_usd=_known_cost_from_row(row, complete),
        cost_evidence_complete=complete,
        pricing_table_version=str(row["pricing_table_version"]),
        cache_hit=_optional_int_to_bool(row["cache_hit"]),
        provider_attempt_count=int(row["provider_attempt_count"]),
        provider_retry_count=int(row["provider_retry_count"]),
        provider_attempts=attempts,
        error_type=_optional_str(row["error_type"]),
        timestamp=str(row["timestamp"]),
    )


def _known_cost_from_row(row: sqlite3.Row, complete: bool) -> float | None:
    value = row["estimated_cost_usd"]
    if not complete:
        return None
    if value is None:
        raise ValueError("complete stored cost evidence is missing estimated_cost_usd")
    return float(value)


def _attempts_json(attempts: tuple[ProviderAttempt, ...]) -> str:
    return json.dumps([asdict(attempt) for attempt in attempts], sort_keys=True)


def _attempts_from_json(raw: str) -> tuple[ProviderAttempt, ...]:
    values = json.loads(raw)
    if not isinstance(values, list):
        raise ValueError("provider_attempts_json must contain a list")
    return tuple(_attempt_from_dict(value) for value in values)


def _attempt_from_dict(raw: Any) -> ProviderAttempt:
    if not isinstance(raw, dict):
        raise ValueError("provider attempt must be an object")
    normalized, _ = _normalize_attempt_cost_provenance(raw)
    return ProviderAttempt(
        attempt_index=int(normalized["attempt_index"]),
        provider=str(normalized["provider"]),
        model=str(normalized["model"]),
        outcome=AttemptOutcome(str(normalized["outcome"])),
        latency_ms=int(normalized["latency_ms"]),
        prompt_tokens=_optional_int(normalized.get("prompt_tokens")),
        completion_tokens=_optional_int(normalized.get("completion_tokens")),
        total_tokens=_optional_int(normalized.get("total_tokens")),
        cached_tokens=_optional_int(normalized.get("cached_tokens")),
        calculated_cost_usd=_optional_float(normalized.get("calculated_cost_usd")),
        reported_cost_usd=_optional_float(normalized.get("reported_cost_usd")),
        cost_evidence=CostEvidenceKind(
            str(normalized.get("cost_evidence", CostEvidenceKind.UNKNOWN.value))
        ),
        pricing_table_version=_optional_str(normalized.get("pricing_table_version")),
        pricing_record_id=_optional_str(normalized.get("pricing_record_id")),
        pricing_observed_at=_optional_str(normalized.get("pricing_observed_at")),
        pricing_source_url=_optional_str(normalized.get("pricing_source_url")),
        cost_source=_optional_str(normalized.get("cost_source")),
        cost_source_record_id=_optional_str(normalized.get("cost_source_record_id")),
        error_type=_optional_str(normalized.get("error_type")),
        status_code=_optional_int(normalized.get("status_code")),
    )


def _normalize_attempt_cost_provenance(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    normalized = dict(raw)
    try:
        cost_evidence = CostEvidenceKind(
            str(normalized.get("cost_evidence", CostEvidenceKind.UNKNOWN.value))
        )
    except ValueError:
        cost_evidence = CostEvidenceKind.UNKNOWN
        malformed_kind = True
    else:
        malformed_kind = False

    calculated_cost_usd = _optional_float(normalized.get("calculated_cost_usd"))
    reported_cost_usd = _optional_float(normalized.get("reported_cost_usd"))
    pricing_provenance = tuple(
        _optional_str(normalized.get(field))
        for field in (
            "pricing_table_version",
            "pricing_record_id",
            "pricing_observed_at",
            "pricing_source_url",
        )
    )
    external_provenance = tuple(
        _optional_str(normalized.get(field))
        for field in ("cost_source", "cost_source_record_id")
    )

    malformed_calculated = cost_evidence == CostEvidenceKind.CALCULATED_FROM_USAGE and (
        calculated_cost_usd is None
        or not isfinite(calculated_cost_usd)
        or calculated_cost_usd < 0
        or reported_cost_usd is not None
        or any(value is None or not value.strip() for value in pricing_provenance)
        or any(value is not None for value in external_provenance)
    )
    malformed_reported = cost_evidence == CostEvidenceKind.REPORTED_BY_EXECUTION_STACK and (
        reported_cost_usd is None
        or not isfinite(reported_cost_usd)
        or reported_cost_usd < 0
        or calculated_cost_usd is not None
        or any(value is not None for value in pricing_provenance)
        or any(value is None or not value.strip() for value in external_provenance)
    )
    malformed_unknown = cost_evidence == CostEvidenceKind.UNKNOWN and (
        calculated_cost_usd is not None
        or reported_cost_usd is not None
        or any(value is not None for value in pricing_provenance)
        or any(value is not None for value in external_provenance)
    )
    malformed = malformed_kind or malformed_calculated or malformed_reported or malformed_unknown
    if not malformed:
        return normalized, False

    normalized["calculated_cost_usd"] = None
    normalized["reported_cost_usd"] = None
    normalized["cost_evidence"] = CostEvidenceKind.UNKNOWN.value
    normalized["pricing_table_version"] = None
    normalized["pricing_record_id"] = None
    normalized["pricing_observed_at"] = None
    normalized["pricing_source_url"] = None
    normalized["cost_source"] = None
    normalized["cost_source_record_id"] = None
    return normalized, True


def _downgrade_ambiguous_legacy_costs(connection: sqlite3.Connection, table_name: str) -> None:
    connection.execute(
        f"""
        UPDATE {table_name}
        SET cost_evidence_complete = 0,
            estimated_cost_usd = NULL
        WHERE provider_attempts_json = '[]'
          AND provider_attempt_count > 0
          AND (error_type IS NOT NULL OR provider_retry_count > 0 OR provider_attempt_count > 1)
        """
    )


def _normalize_stored_attempt_provenance(
    connection: sqlite3.Connection,
    table_name: str,
) -> None:
    rows = connection.execute(
        f"SELECT rowid, provider_attempts_json, estimated_cost_usd, cost_evidence_complete "
        f"FROM {table_name} WHERE provider_attempts_json != '[]'"
    ).fetchall()
    for row in rows:
        raw_attempts = json.loads(str(row["provider_attempts_json"]))
        if not isinstance(raw_attempts, list):
            raise ValueError(f"{table_name}.provider_attempts_json must contain a list")

        changed = False
        normalized_attempts: list[dict[str, Any]] = []
        for raw_attempt in raw_attempts:
            if not isinstance(raw_attempt, dict):
                raise ValueError(f"{table_name}.provider_attempts_json contains a non-object attempt")
            normalized, attempt_changed = _normalize_attempt_cost_provenance(raw_attempt)
            normalized_attempts.append(normalized)
            changed = changed or attempt_changed

        any_unknown_cost = any(
            CostEvidenceKind(
                str(attempt.get("cost_evidence", CostEvidenceKind.UNKNOWN.value))
            )
            == CostEvidenceKind.UNKNOWN
            for attempt in normalized_attempts
        )
        stale_complete_aggregate = any_unknown_cost and (
            bool(row["cost_evidence_complete"]) or row["estimated_cost_usd"] is not None
        )
        if changed or stale_complete_aggregate:
            connection.execute(
                f"""
                UPDATE {table_name}
                SET provider_attempts_json = ?,
                    cost_evidence_complete = 0,
                    estimated_cost_usd = NULL
                WHERE rowid = ?
                """,
                (json.dumps(normalized_attempts, sort_keys=True), int(row["rowid"])),
            )


def _column_is_not_null(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
) -> bool:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    for row in rows:
        if str(row["name"]) == column_name:
            return bool(row["notnull"])
    raise ValueError(f"{table_name}.{column_name} does not exist")


def _rebuild_trace_table_with_nullable_cost(connection: sqlite3.Connection) -> None:
    connection.execute("ALTER TABLE benchmark_traces RENAME TO benchmark_traces_legacy_cost")
    connection.execute(_TRACE_TABLE_SQL)
    connection.execute(
        """
        INSERT INTO benchmark_traces (
            run_id,
            request_id,
            provider,
            model,
            latency_ms,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            estimated_cost_usd,
            cost_evidence_complete,
            pricing_table_version,
            cache_hit,
            error_type,
            error_message,
            quality_passed,
            quality_score,
            quality_reason,
            eval_type,
            provider_attempt_count,
            provider_retry_count,
            provider_attempts_json,
            timestamp
        )
        SELECT
            run_id,
            request_id,
            provider,
            model,
            latency_ms,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            CASE WHEN cost_evidence_complete = 1 THEN estimated_cost_usd ELSE NULL END,
            cost_evidence_complete,
            pricing_table_version,
            cache_hit,
            error_type,
            error_message,
            quality_passed,
            quality_score,
            quality_reason,
            eval_type,
            provider_attempt_count,
            provider_retry_count,
            provider_attempts_json,
            timestamp
        FROM benchmark_traces_legacy_cost
        """
    )
    connection.execute("DROP TABLE benchmark_traces_legacy_cost")


def _rebuild_trace_table_with_nullable_cache_hit(connection: sqlite3.Connection) -> None:
    connection.execute("ALTER TABLE benchmark_traces RENAME TO benchmark_traces_legacy_cache_hit")
    connection.execute(_TRACE_TABLE_SQL)
    connection.execute(
        """
        INSERT INTO benchmark_traces (
            run_id,
            request_id,
            provider,
            model,
            latency_ms,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            estimated_cost_usd,
            cost_evidence_complete,
            pricing_table_version,
            cache_hit,
            error_type,
            error_message,
            quality_passed,
            quality_score,
            quality_reason,
            eval_type,
            provider_attempt_count,
            provider_retry_count,
            provider_attempts_json,
            timestamp
        )
        SELECT
            run_id,
            request_id,
            provider,
            model,
            latency_ms,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            estimated_cost_usd,
            cost_evidence_complete,
            pricing_table_version,
            cache_hit,
            error_type,
            error_message,
            quality_passed,
            quality_score,
            quality_reason,
            eval_type,
            provider_attempt_count,
            provider_retry_count,
            provider_attempts_json,
            timestamp
        FROM benchmark_traces_legacy_cache_hit
        """
    )
    connection.execute("DROP TABLE benchmark_traces_legacy_cache_hit")


def _rebuild_usage_table_with_nullable_cost(connection: sqlite3.Connection) -> None:
    connection.execute(
        "ALTER TABLE benchmark_provider_usage RENAME TO benchmark_provider_usage_legacy_cost"
    )
    connection.execute(_USAGE_TABLE_SQL)
    connection.execute(
        """
        INSERT INTO benchmark_provider_usage (
            run_id,
            request_id,
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            estimated_cost_usd,
            cost_evidence_complete,
            pricing_table_version,
            cache_hit,
            provider_attempt_count,
            provider_retry_count,
            provider_attempts_json,
            error_type,
            timestamp
        )
        SELECT
            run_id,
            request_id,
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            CASE WHEN cost_evidence_complete = 1 THEN estimated_cost_usd ELSE NULL END,
            cost_evidence_complete,
            pricing_table_version,
            cache_hit,
            provider_attempt_count,
            provider_retry_count,
            provider_attempts_json,
            error_type,
            timestamp
        FROM benchmark_provider_usage_legacy_cost
        """
    )
    connection.execute("DROP TABLE benchmark_provider_usage_legacy_cost")


def _rebuild_usage_table_with_nullable_cache_hit(connection: sqlite3.Connection) -> None:
    connection.execute(
        "ALTER TABLE benchmark_provider_usage RENAME TO benchmark_provider_usage_legacy_cache_hit"
    )
    connection.execute(_USAGE_TABLE_SQL)
    connection.execute(
        """
        INSERT INTO benchmark_provider_usage (
            run_id,
            request_id,
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            estimated_cost_usd,
            cost_evidence_complete,
            pricing_table_version,
            cache_hit,
            provider_attempt_count,
            provider_retry_count,
            provider_attempts_json,
            error_type,
            timestamp
        )
        SELECT
            run_id,
            request_id,
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            estimated_cost_usd,
            cost_evidence_complete,
            pricing_table_version,
            cache_hit,
            provider_attempt_count,
            provider_retry_count,
            provider_attempts_json,
            error_type,
            timestamp
        FROM benchmark_provider_usage_legacy_cache_hit
        """
    )
    connection.execute("DROP TABLE benchmark_provider_usage_legacy_cache_hit")


def _optional_bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _optional_int_to_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, int)):
        return int(value)
    if isinstance(value, (str, bytes, bytearray, float)):
        return int(value)
    raise TypeError(f"expected integer-compatible value, got {type(value).__name__}")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return float(value)
    if isinstance(value, (str, bytes, bytearray)):
        return float(value)
    raise TypeError(f"expected float-compatible value, got {type(value).__name__}")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _ensure_columns(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    columns: dict[str, str],
) -> None:
    existing = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    for column_name, column_type in columns.items():
        if column_name not in existing:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
