from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .segmentation import BenchmarkRequestContext

CONTEXT_SCHEMA_VERSION = 1

_CONTEXT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_request_contexts (
    run_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    workload_item_id TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    PRIMARY KEY (run_id, request_id),
    UNIQUE (run_id, workload_item_id),
    FOREIGN KEY (run_id, request_id)
        REFERENCES benchmark_traces(run_id, request_id) ON DELETE CASCADE
)
"""


class SQLiteBenchmarkContextStore:
    """Canonical persistence owner for workload identity and segmentation tags.

    Context is benchmark metadata, not provider telemetry. It shares the benchmark SQLite database
    while remaining a separate storage concern from execution/pricing evidence.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(_CONTEXT_TABLE_SQL)
            connection.execute(
                """
                INSERT OR REPLACE INTO ledger_metadata (key, value)
                VALUES ('benchmark_context_schema_version', ?)
                """,
                (str(CONTEXT_SCHEMA_VERSION),),
            )

    def record_contexts(
        self,
        *,
        run_id: str,
        contexts: list[BenchmarkRequestContext],
    ) -> None:
        self.initialize()
        contexts_by_id = _unique_contexts(contexts)
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            run_exists = connection.execute(
                "SELECT 1 FROM benchmark_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_exists is None:
                raise KeyError(f"Unknown benchmark run_id: {run_id}")

            trace_rows = connection.execute(
                "SELECT request_id FROM benchmark_traces WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            trace_ids = {str(row["request_id"]) for row in trace_rows}
            context_ids = set(contexts_by_id)
            if trace_ids != context_ids:
                raise ValueError(
                    "benchmark contexts must cover exactly the stored run traces; "
                    f"missing_context={sorted(trace_ids - context_ids)}, "
                    f"missing_trace={sorted(context_ids - trace_ids)}"
                )

            connection.execute(
                "DELETE FROM benchmark_request_contexts WHERE run_id = ?",
                (run_id,),
            )
            connection.executemany(
                """
                INSERT INTO benchmark_request_contexts (
                    run_id, request_id, workload_item_id, tags_json
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        context.request_id,
                        context.workload_item_id,
                        json.dumps(context.tags_dict(), sort_keys=True),
                    )
                    for context in contexts
                ],
            )

    def get_contexts(self, run_id: str) -> list[BenchmarkRequestContext]:
        self.initialize()
        with self._connect() as connection:
            run_exists = connection.execute(
                "SELECT 1 FROM benchmark_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_exists is None:
                raise KeyError(f"Unknown benchmark run_id: {run_id}")
            rows = connection.execute(
                """
                SELECT request_id, workload_item_id, tags_json
                FROM benchmark_request_contexts
                WHERE run_id = ?
                ORDER BY workload_item_id, request_id
                """,
                (run_id,),
            ).fetchall()
        return [_context_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


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


def _context_from_row(row: sqlite3.Row) -> BenchmarkRequestContext:
    try:
        raw_tags = json.loads(str(row["tags_json"]))
    except json.JSONDecodeError as exc:
        raise ValueError("stored benchmark context tags must be valid JSON") from exc
    if not isinstance(raw_tags, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw_tags.items()
    ):
        raise ValueError("stored benchmark context tags must be an object of strings")
    return BenchmarkRequestContext.from_tags(
        request_id=str(row["request_id"]),
        workload_item_id=str(row["workload_item_id"]),
        tags=raw_tags,
    )
