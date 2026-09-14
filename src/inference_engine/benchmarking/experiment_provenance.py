"""Git/spec provenance for baseline → candidate experiments.

This is a small fail-closed contract, not a general provenance service. It records whether an
experiment specification was committed and unmodified at execution time, hashes the spec and
workload, and derives execution intervals from immutable source telemetry rather than
hand-written capture timestamps.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Any

_GIT_ISO_FORMAT = "%cI"


class ExperimentProvenanceError(ValueError):
    """Raised when experiment-spec Git provenance cannot be established fail-closed."""


class TimelineStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    UNVERIFIED = "unverified"


class PreRegistrationEvidenceClass(StrEnum):
    GIT_PROVEN = "GIT_PROVEN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class GitSpecSnapshot:
    """Committed, unmodified experiment specification at one Git HEAD."""

    spec_path: str
    spec_relative_path: str
    spec_sha256: str
    spec_commit_sha: str
    spec_commit_committed_at_utc: str
    source_commit_sha: str
    source_commit_committed_at_utc: str
    spec_committed_and_clean: bool


@dataclass(frozen=True)
class ExecutionInterval:
    """Closed interval derived from source telemetry, not a wall-clock placeholder."""

    source: str
    start_field: str
    end_field: str
    earliest_start_utc: str
    latest_end_utc: str
    payload_count: int


@dataclass(frozen=True)
class TimelineAudit:
    status: TimelineStatus
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PreRegistrationRecord:
    declared_status: str | None
    declared_locked_at_utc: str | None
    spec_committed_and_clean: bool
    git_proven_before_execution: bool
    evidence_class: PreRegistrationEvidenceClass
    reason: str
    spec_sha256: str | None
    spec_commit_sha: str | None
    source_commit_sha: str | None
    workload_sha256: str | None


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def require_committed_clean_experiment_spec(spec_path: Path) -> GitSpecSnapshot:
    """Fail closed unless ``spec_path`` is tracked, unmodified, and present in HEAD."""
    resolved = spec_path.resolve()
    if not resolved.is_file():
        raise ExperimentProvenanceError(f"experiment specification does not exist: {resolved}")
    repo_root = _git_toplevel(resolved.parent)
    relative = _relative_posix(repo_root, resolved)
    porcelain = _git_text(repo_root, "status", "--porcelain", "--", relative)
    if porcelain:
        raise ExperimentProvenanceError(
            "experiment specification has uncommitted modifications or is untracked; "
            f"path={relative} status={porcelain!r}"
        )
    tracked = _git_run(repo_root, "ls-files", "--error-unmatch", "--", relative)
    if tracked.returncode != 0:
        raise ExperimentProvenanceError(
            f"experiment specification is not tracked in Git: {relative}"
        )
    spec_digest = file_sha256(resolved)
    head_blob = _git_bytes(repo_root, "show", f"HEAD:{relative}")
    if sha256(head_blob).hexdigest() != spec_digest:
        raise ExperimentProvenanceError(
            "working-tree experiment specification does not match the HEAD blob"
        )
    spec_commit_sha = _git_text(repo_root, "log", "-1", "--format=%H", "--", relative)
    spec_commit_at = _normalize_git_datetime(
        _git_text(repo_root, "log", "-1", f"--format={_GIT_ISO_FORMAT}", "--", relative)
    )
    source_commit_sha = _git_text(repo_root, "rev-parse", "HEAD")
    source_commit_at = _normalize_git_datetime(
        _git_text(repo_root, "log", "-1", f"--format={_GIT_ISO_FORMAT}", "HEAD")
    )
    spec_commit_blob = _git_bytes(repo_root, "show", f"{spec_commit_sha}:{relative}")
    if sha256(spec_commit_blob).hexdigest() != spec_digest:
        raise ExperimentProvenanceError(
            "recorded spec commit does not contain the exact experiment specification bytes"
        )
    return GitSpecSnapshot(
        spec_path=str(resolved),
        spec_relative_path=relative,
        spec_sha256=spec_digest,
        spec_commit_sha=spec_commit_sha,
        spec_commit_committed_at_utc=spec_commit_at,
        source_commit_sha=source_commit_sha,
        source_commit_committed_at_utc=source_commit_at,
        spec_committed_and_clean=True,
    )


def require_clean_git_worktree(start: Path) -> str:
    """Fail closed unless the Git worktree has no uncommitted or untracked paths."""
    resolved = start.resolve()
    probe = resolved if resolved.is_dir() else resolved.parent
    repo_root = _git_toplevel(probe)
    porcelain = _git_text(repo_root, "status", "--porcelain")
    if porcelain:
        raise ExperimentProvenanceError(
            "git worktree is not clean; refusing the execution boundary"
        )
    return _git_text(repo_root, "rev-parse", "HEAD")


def execution_interval_from_standard_logging_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    source: str = "litellm_standard_logging_payload",
    start_field: str = "startTime",
    end_field: str = "endTime",
) -> ExecutionInterval:
    if not payloads:
        raise ExperimentProvenanceError("at least one source payload is required to derive an interval")
    starts = [_unix_seconds(payload, start_field) for payload in payloads]
    ends = [_unix_seconds(payload, end_field) for payload in payloads]
    earliest = min(starts)
    latest = max(ends)
    if latest < earliest:
        raise ExperimentProvenanceError("source telemetry latest endTime precedes earliest startTime")
    return ExecutionInterval(
        source=source,
        start_field=start_field,
        end_field=end_field,
        earliest_start_utc=_unix_to_utc_iso(earliest),
        latest_end_utc=_unix_to_utc_iso(latest),
        payload_count=len(payloads),
    )


def load_standard_logging_payloads(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parsed = _parse_json_object(stripped, path=path, line_number=line_number)
        rows.append(parsed)
    return rows


def audit_evidence_timeline(
    *,
    earliest_start_utc: str | None = None,
    latest_end_utc: str | None = None,
    generated_at_utc: str | None = None,
    captured_at_utc: str | None = None,
    spec_committed_at_utc: str | None = None,
    require_spec_precedes_execution: bool = False,
) -> TimelineAudit:
    reasons: list[str] = []
    start = _parse_optional_datetime(earliest_start_utc)
    end = _parse_optional_datetime(latest_end_utc)
    generated = _parse_optional_datetime(generated_at_utc)
    captured = _parse_optional_datetime(captured_at_utc)

    if start is None or end is None:
        reasons.append("execution interval is missing from source telemetry")
        if require_spec_precedes_execution:
            return TimelineAudit(status=TimelineStatus.INVALID, reasons=tuple(reasons))
        return TimelineAudit(status=TimelineStatus.UNVERIFIED, reasons=tuple(reasons))
    if end < start:
        reasons.append("latest endTime precedes earliest startTime")
        return TimelineAudit(status=TimelineStatus.INVALID, reasons=tuple(reasons))
    if generated is not None and generated < end:
        reasons.append("artifact generated_at_utc precedes source telemetry latest endTime")
        return TimelineAudit(status=TimelineStatus.INVALID, reasons=tuple(reasons))
    if captured is not None:
        if captured < start:
            reasons.append("captured_at_utc precedes source telemetry earliest startTime")
            return TimelineAudit(status=TimelineStatus.INVALID, reasons=tuple(reasons))
        if generated is not None and captured > generated:
            reasons.append(
                "captured_at_utc is later than generated_at_utc and cannot be a capture timestamp "
                "for this decision artifact"
            )
            return TimelineAudit(status=TimelineStatus.INVALID, reasons=tuple(reasons))
    precedes, precede_reason = _spec_precedes_execution(
        spec_committed_at_utc=spec_committed_at_utc,
        earliest_start_utc=earliest_start_utc,
    )
    if spec_committed_at_utc is not None and not precedes:
        reasons.append(precede_reason or "experiment-spec commit does not precede execution")
        if require_spec_precedes_execution:
            return TimelineAudit(status=TimelineStatus.INVALID, reasons=tuple(reasons))
        return TimelineAudit(status=TimelineStatus.UNVERIFIED, reasons=tuple(reasons))
    if require_spec_precedes_execution and spec_committed_at_utc is None:
        reasons.append("experiment-spec commit timestamp is missing")
        return TimelineAudit(status=TimelineStatus.INVALID, reasons=tuple(reasons))
    return TimelineAudit(status=TimelineStatus.VALID, reasons=())


def classify_pre_registration(
    *,
    declared_status: str | None,
    declared_locked_at_utc: str | None,
    snapshot: GitSpecSnapshot | None,
    workload_sha256: str | None,
    earliest_start_utc: str | None,
    require_spec_precedes_execution: bool = False,
) -> PreRegistrationRecord:
    if snapshot is None:
        return PreRegistrationRecord(
            declared_status=declared_status,
            declared_locked_at_utc=declared_locked_at_utc,
            spec_committed_and_clean=False,
            git_proven_before_execution=False,
            evidence_class=PreRegistrationEvidenceClass.INSUFFICIENT_EVIDENCE,
            reason=(
                "no committed Git snapshot of the experiment specification is available; "
                "declared pre-registration is not independently Git-proven"
            ),
            spec_sha256=None,
            spec_commit_sha=None,
            source_commit_sha=None,
            workload_sha256=workload_sha256,
        )
    precedes, precede_reason = _spec_precedes_execution(
        spec_committed_at_utc=snapshot.source_commit_committed_at_utc,
        earliest_start_utc=earliest_start_utc,
    )
    if snapshot.spec_committed_and_clean and precedes:
        return PreRegistrationRecord(
            declared_status=declared_status,
            declared_locked_at_utc=declared_locked_at_utc,
            spec_committed_and_clean=True,
            git_proven_before_execution=True,
            evidence_class=PreRegistrationEvidenceClass.GIT_PROVEN,
            reason=(
                "experiment specification was committed and unmodified in Git, and the source "
                "commit timestamp precedes captured execution telemetry"
            ),
            spec_sha256=snapshot.spec_sha256,
            spec_commit_sha=snapshot.spec_commit_sha,
            source_commit_sha=snapshot.source_commit_sha,
            workload_sha256=workload_sha256,
        )
    reason = (
        "experiment specification was committed and unmodified at inspection time, but Git "
        "cannot prove the commit preceded execution telemetry"
        if snapshot.spec_committed_and_clean
        else "experiment specification is not a committed clean Git snapshot"
    )
    if precede_reason:
        reason = f"{reason}: {precede_reason}"
    if require_spec_precedes_execution:
        raise ExperimentProvenanceError(reason)
    return PreRegistrationRecord(
        declared_status=declared_status,
        declared_locked_at_utc=declared_locked_at_utc,
        spec_committed_and_clean=snapshot.spec_committed_and_clean,
        git_proven_before_execution=False,
        evidence_class=PreRegistrationEvidenceClass.INSUFFICIENT_EVIDENCE,
        reason=reason,
        spec_sha256=snapshot.spec_sha256,
        spec_commit_sha=snapshot.spec_commit_sha,
        source_commit_sha=snapshot.source_commit_sha,
        workload_sha256=workload_sha256,
    )


def merge_execution_intervals(intervals: Sequence[ExecutionInterval]) -> ExecutionInterval:
    if not intervals:
        raise ExperimentProvenanceError("at least one execution interval is required")
    sources = {item.source for item in intervals}
    start_fields = {item.start_field for item in intervals}
    end_fields = {item.end_field for item in intervals}
    if len(sources) != 1 or len(start_fields) != 1 or len(end_fields) != 1:
        raise ExperimentProvenanceError("execution intervals must share one telemetry source contract")
    starts = [_parse_optional_datetime(item.earliest_start_utc) for item in intervals]
    ends = [_parse_optional_datetime(item.latest_end_utc) for item in intervals]
    if any(item is None for item in starts + ends):
        raise ExperimentProvenanceError("execution intervals must include UTC timestamps")
    earliest = min(item for item in starts if item is not None)
    latest = max(item for item in ends if item is not None)
    return ExecutionInterval(
        source=intervals[0].source,
        start_field=intervals[0].start_field,
        end_field=intervals[0].end_field,
        earliest_start_utc=earliest.isoformat(),
        latest_end_utc=latest.isoformat(),
        payload_count=sum(item.payload_count for item in intervals),
    )


def build_execution_provenance(
    *,
    snapshot: GitSpecSnapshot,
    interval: ExecutionInterval,
    workload_sha256: str,
    declared_status: str | None,
    declared_locked_at_utc: str | None,
    generated_at_utc: str | None = None,
    captured_at_utc: str | None = None,
    require_spec_precedes_execution: bool = True,
) -> dict[str, Any]:
    audit = audit_evidence_timeline(
        earliest_start_utc=interval.earliest_start_utc,
        latest_end_utc=interval.latest_end_utc,
        generated_at_utc=generated_at_utc,
        captured_at_utc=captured_at_utc,
        spec_committed_at_utc=snapshot.source_commit_committed_at_utc,
        require_spec_precedes_execution=require_spec_precedes_execution,
    )
    if audit.status == TimelineStatus.INVALID:
        raise ExperimentProvenanceError("; ".join(audit.reasons))
    record = classify_pre_registration(
        declared_status=declared_status,
        declared_locked_at_utc=declared_locked_at_utc,
        snapshot=snapshot,
        workload_sha256=workload_sha256,
        earliest_start_utc=interval.earliest_start_utc,
        require_spec_precedes_execution=require_spec_precedes_execution,
    )
    return {
        "git": snapshot_as_dict(snapshot),
        "workload_sha256": workload_sha256,
        "execution_interval": asdict(interval),
        "timeline": {"status": audit.status.value, "reasons": list(audit.reasons)},
        "pre_registration": pre_registration_as_dict(record),
    }


def withdraw_invalid_captured_at(
    runtime: Mapping[str, Any],
    *,
    earliest_start_utc: str,
    latest_end_utc: str,
    generated_at_utc: str | None,
) -> dict[str, Any]:
    """Remove a misleading captured_at_utc rather than inventing a replacement."""
    corrected = dict(runtime)
    captured = corrected.pop("captured_at_utc", None)
    if captured is None:
        return corrected
    captured_text = str(captured)
    audit = audit_evidence_timeline(
        earliest_start_utc=earliest_start_utc,
        latest_end_utc=latest_end_utc,
        generated_at_utc=generated_at_utc,
        captured_at_utc=captured_text,
    )
    if audit.status != TimelineStatus.INVALID:
        corrected["captured_at_utc"] = captured_text
        return corrected
    withdrawn = dict(corrected.get("withdrawn_runtime_fields") or {})
    if not isinstance(withdrawn, dict):
        withdrawn = {}
    withdrawn["captured_at_utc"] = {
        "value": captured_text,
        "status": TimelineStatus.INVALID.value,
        "reason": "; ".join(audit.reasons),
    }
    corrected["withdrawn_runtime_fields"] = withdrawn
    return corrected


def snapshot_as_dict(snapshot: GitSpecSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def pre_registration_as_dict(record: PreRegistrationRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["evidence_class"] = record.evidence_class.value
    return payload


def _git_toplevel(start: Path) -> Path:
    result = _git_run(start, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        raise ExperimentProvenanceError(
            f"experiment specification is not inside a Git repository: {start}"
        )
    return Path(result.stdout.decode("utf-8").strip())


def _relative_posix(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ExperimentProvenanceError(
            f"experiment specification {path} is not inside repository {repo_root}"
        ) from exc


def _git_text(cwd: Path, *args: str) -> str:
    result = _git_run(cwd, *args)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ExperimentProvenanceError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.decode("utf-8").strip()


def _git_bytes(cwd: Path, *args: str) -> bytes:
    result = _git_run(cwd, *args)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ExperimentProvenanceError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _git_run(cwd: Path, *args: str) -> CompletedProcess[bytes]:
    return run(
        ["git", "-C", str(cwd), *args],
        check=False,
        capture_output=True,
    )


def _unix_seconds(payload: Mapping[str, Any], field: str) -> float:
    if field not in payload:
        raise ExperimentProvenanceError(f"source payload is missing {field}")
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentProvenanceError(f"source payload {field} must be numeric unix seconds")
    resolved = float(value)
    if not isfinite(resolved):
        raise ExperimentProvenanceError(f"source payload {field} must be finite")
    return resolved


def _unix_to_utc_iso(value: float) -> str:
    return datetime.fromtimestamp(value, tz=UTC).isoformat()


def _spec_precedes_execution(
    *,
    spec_committed_at_utc: str | None,
    earliest_start_utc: str | None,
) -> tuple[bool, str | None]:
    spec_committed = _parse_optional_datetime(spec_committed_at_utc)
    start = _parse_optional_datetime(earliest_start_utc)
    if start is None:
        return False, "execution interval is missing from source telemetry"
    if spec_committed is None:
        return False, "experiment-spec commit timestamp is missing"
    if spec_committed >= start:
        return (
            False,
            "experiment-spec commit timestamp does not precede source telemetry earliest startTime",
        )
    return True, None


def _parse_json_object(raw: str, *, path: Path, line_number: int) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise ExperimentProvenanceError(f"{path}:{line_number} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ExperimentProvenanceError(f"{path}:{line_number} must contain a JSON object")
    return parsed


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _normalize_git_datetime(value: str) -> str:
    parsed = _parse_optional_datetime(value)
    if parsed is None:
        raise ExperimentProvenanceError("git commit timestamp is missing")
    return parsed.isoformat()
