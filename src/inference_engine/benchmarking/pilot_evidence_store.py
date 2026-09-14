from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..domain.models.economics import (
    ExactAttemptCost,
    ExactRateEvidence,
    ExactRequestCostEvidence,
)
from ..domain.models.execution import CostEvidenceKind
from .pilot_contract import (
    PILOT_EVIDENCE_CONTRACT_VERSION,
    AcquisitionSemantics,
    AssignmentSemantics,
    CollectionSemantics,
    MeasurementSemantics,
)

PILOT_EVIDENCE_STORE_VERSION = "1"


class SQLitePilotEvidenceStore:
    """Pilot-only semantic/economic evidence persisted as exact TEXT/JSON surfaces.

    Existing benchmark float columns remain compatibility and statistical surfaces. This store is
    the authoritative persistence layer for the prospective pilot's acquisition semantics and
    exact attempt-chain monetary evidence.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pilot_run_acquisition (
                    run_id TEXT PRIMARY KEY,
                    contract_version TEXT NOT NULL,
                    measurement TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    assignment TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pilot_exact_request_costs (
                    run_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, request_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pilot_evidence_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO pilot_evidence_metadata (key, value)
                VALUES ('store_version', ?)
                """,
                (PILOT_EVIDENCE_STORE_VERSION,),
            )

    def record_run(
        self,
        *,
        run_id: str,
        acquisition: AcquisitionSemantics,
        exact_costs: Sequence[ExactRequestCostEvidence],
        expected_request_ids: Sequence[str],
    ) -> None:
        if not run_id.strip():
            raise ValueError("pilot evidence run_id must be non-empty")
        expected = list(expected_request_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("pilot expected request ids contain duplicates")
        exact_list = list(exact_costs)
        exact_ids = [item.request_id for item in exact_list]
        if len(exact_ids) != len(set(exact_ids)):
            raise ValueError("pilot exact cost evidence contains duplicate request ids")
        if set(exact_ids) != set(expected) or len(exact_ids) != len(expected):
            raise ValueError(
                "pilot exact cost evidence requires exact request coverage; "
                f"missing={sorted(set(expected) - set(exact_ids))}, "
                f"unexpected={sorted(set(exact_ids) - set(expected))}"
            )
        if acquisition.measurement == MeasurementSemantics.ESTIMATED and any(
            item.attempts for item in exact_list
        ):
            raise ValueError(
                "estimated pilot evidence cannot contain observed provider-attempt monetary records"
            )

        self.initialize()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO pilot_run_acquisition (
                    run_id, contract_version, measurement, collection, assignment
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    PILOT_EVIDENCE_CONTRACT_VERSION,
                    acquisition.measurement.value,
                    acquisition.collection.value,
                    acquisition.assignment.value,
                ),
            )
            connection.execute("DELETE FROM pilot_exact_request_costs WHERE run_id = ?", (run_id,))
            connection.executemany(
                """
                INSERT INTO pilot_exact_request_costs (run_id, request_id, evidence_json)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        run_id,
                        item.request_id,
                        json.dumps(asdict(item), sort_keys=True, separators=(",", ":")),
                    )
                    for item in sorted(exact_list, key=lambda value: value.request_id)
                ],
            )

    def get_acquisition(self, run_id: str) -> AcquisitionSemantics:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT contract_version, measurement, collection, assignment
                FROM pilot_run_acquisition
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"No pilot acquisition evidence for run_id: {run_id}")
        if str(row["contract_version"]) != PILOT_EVIDENCE_CONTRACT_VERSION:
            raise ValueError("unsupported stored pilot evidence contract version")
        return AcquisitionSemantics(
            measurement=MeasurementSemantics(str(row["measurement"])),
            collection=CollectionSemantics(str(row["collection"])),
            assignment=AssignmentSemantics(str(row["assignment"])),
        )

    def get_exact_costs(self, run_id: str) -> list[ExactRequestCostEvidence]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT evidence_json
                FROM pilot_exact_request_costs
                WHERE run_id = ?
                ORDER BY request_id
                """,
                (run_id,),
            ).fetchall()
        return [_exact_request_from_dict(json.loads(str(row["evidence_json"]))) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


def _exact_request_from_dict(raw: dict[str, Any]) -> ExactRequestCostEvidence:
    raw_attempts = raw.get("attempts")
    if not isinstance(raw_attempts, list):
        raise ValueError("stored exact request attempts must be an array")
    raw_complete = raw.get("complete")
    if not isinstance(raw_complete, bool):
        raise ValueError("stored exact request complete flag must be boolean")
    attempts = tuple(_exact_attempt_from_dict(item) for item in raw_attempts)
    return ExactRequestCostEvidence(
        request_id=str(raw["request_id"]),
        currency=str(raw["currency"]),
        attempts=attempts,
        complete=raw_complete,
        total_decimal=_optional_text(raw.get("total_decimal")),
    )


def _exact_attempt_from_dict(raw: object) -> ExactAttemptCost:
    if not isinstance(raw, dict):
        raise ValueError("stored exact attempt evidence must be an object")
    raw_rate = raw.get("rate_evidence")
    rate = None
    if raw_rate is not None:
        if not isinstance(raw_rate, dict):
            raise ValueError("stored exact rate evidence must be an object")
        rate = ExactRateEvidence(
            input_per_million=str(raw_rate["input_per_million"]),
            output_per_million=str(raw_rate["output_per_million"]),
            cached_input_per_million=_optional_text(raw_rate.get("cached_input_per_million")),
            pricing_record_id=str(raw_rate["pricing_record_id"]),
            pricing_table_version=str(raw_rate["pricing_table_version"]),
            pricing_observed_at=str(raw_rate["pricing_observed_at"]),
            pricing_source_url=str(raw_rate["pricing_source_url"]),
        )
    return ExactAttemptCost(
        attempt_index=int(raw["attempt_index"]),
        currency=str(raw["currency"]),
        evidence_kind=CostEvidenceKind(str(raw["evidence_kind"])),
        amount_decimal=_optional_text(raw.get("amount_decimal")),
        source=_optional_text(raw.get("source")),
        source_record_id=_optional_text(raw.get("source_record_id")),
        unknown_reason=_optional_text(raw.get("unknown_reason")),
        prompt_tokens=_optional_int(raw.get("prompt_tokens")),
        completion_tokens=_optional_int(raw.get("completion_tokens")),
        cached_tokens=_optional_int(raw.get("cached_tokens")),
        rate_evidence=rate,
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("stored exact token count must be integer or null")
    return value
