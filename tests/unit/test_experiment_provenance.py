from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from inference_engine.benchmarking.experiment_provenance import (
    ExperimentProvenanceError,
    PreRegistrationEvidenceClass,
    TimelineStatus,
    audit_evidence_timeline,
    classify_pre_registration,
    execution_interval_from_standard_logging_payloads,
    file_sha256,
    load_standard_logging_payloads,
    require_clean_git_worktree,
    require_committed_clean_experiment_spec,
    withdraw_invalid_captured_at,
)

MISSION4_EARLIEST_START = "2026-09-11T00:06:44.229721+00:00"
MISSION4_LATEST_END = "2026-09-11T00:07:38.755322+00:00"
MISSION4_GENERATED_AT = "2026-09-11T00:08:08.732495+00:00"
MISSION4_PLACEHOLDER_CAPTURED_AT = "2026-09-11T00:30:00+00:00"
MISSION4_FIRST_GIT_COMMIT = "67825bb5e533fe9e301fc7a29b1ae741affbbc75"
MISSION4_ORIGINAL_SPEC_SHA256 = "feb7f9ab2c6e11feaca2aa65a19eeccef50dc7f5f65318c7ee2666402d8196c0"
REPO_ROOT = Path(__file__).resolve().parents[2]
_GIT_TMP = REPO_ROOT / ".tmp-pytest-git"


def _git_repo_dir(tmp_path: Path) -> Path:
    root = _GIT_TMP / tmp_path.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command_env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        **(env or {}),
    }
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.email=provenance-test@example.com",
            "-c",
            "user.name=Provenance Test",
            *args,
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=command_env,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result


def _init_spec_repo(
    tmp_path: Path,
    *,
    commit_at: str = "2026-09-11T00:00:00+00:00",
    spec: dict[str, object] | None = None,
) -> tuple[Path, Path, str]:
    repo = _git_repo_dir(tmp_path)
    template = repo / ".empty-git-template"
    template.mkdir()
    _git(repo, "init", "--template", str(template))
    spec_path = repo / "experiment.json"
    payload = spec or {
        "experiment_id": "fixture",
        "status": "PRE_REGISTERED_BEFORE_EXECUTION",
        "locked_at_utc": "2026-09-11T00:00:00+00:00",
    }
    spec_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _git(repo, "add", "experiment.json")
    _git(
        repo,
        "commit",
        "-m",
        "lock experiment spec",
        env={
            "GIT_AUTHOR_DATE": commit_at,
            "GIT_COMMITTER_DATE": commit_at,
        },
    )
    sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    return repo, spec_path, sha


def test_dirty_experiment_specification_fails_closed(tmp_path: Path) -> None:
    _repo, spec_path, _sha = _init_spec_repo(tmp_path)
    spec_path.write_text(spec_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ExperimentProvenanceError, match="uncommitted modifications"):
        require_committed_clean_experiment_spec(spec_path)


def test_clean_worktree_returns_head_sha(tmp_path: Path) -> None:
    repo, spec_path, commit_sha = _init_spec_repo(tmp_path)

    assert require_clean_git_worktree(repo) == commit_sha
    assert require_clean_git_worktree(spec_path) == commit_sha


def test_dirty_worktree_fails_closed_at_execution_boundary(tmp_path: Path) -> None:
    repo, spec_path, _sha = _init_spec_repo(tmp_path)
    spec_path.write_text(spec_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ExperimentProvenanceError, match="worktree is not clean"):
        require_clean_git_worktree(repo)


def test_recorded_spec_sha_matches_file_and_commit_blob(tmp_path: Path) -> None:
    repo, spec_path, commit_sha = _init_spec_repo(tmp_path)

    snapshot = require_committed_clean_experiment_spec(spec_path)

    assert snapshot.spec_sha256 == file_sha256(spec_path)
    assert snapshot.spec_commit_sha == commit_sha
    assert snapshot.source_commit_sha == commit_sha
    blob = _git(repo, "show", f"{snapshot.spec_commit_sha}:{snapshot.spec_relative_path}").stdout
    assert blob == spec_path.read_text(encoding="utf-8")


def test_impossible_captured_at_is_invalid() -> None:
    audit = audit_evidence_timeline(
        earliest_start_utc=MISSION4_EARLIEST_START,
        latest_end_utc=MISSION4_LATEST_END,
        generated_at_utc=MISSION4_GENERATED_AT,
        captured_at_utc=MISSION4_PLACEHOLDER_CAPTURED_AT,
    )

    assert audit.status is TimelineStatus.INVALID
    assert any("later than generated_at_utc" in reason for reason in audit.reasons)


def test_spec_commit_after_execution_is_not_git_proven(tmp_path: Path) -> None:
    _repo, spec_path, _sha = _init_spec_repo(tmp_path, commit_at="2026-09-11T00:12:06+00:00")
    snapshot = require_committed_clean_experiment_spec(spec_path)

    record = classify_pre_registration(
        declared_status="PRE_REGISTERED_BEFORE_EXECUTION",
        declared_locked_at_utc="2026-09-11T00:00:00+00:00",
        snapshot=snapshot,
        workload_sha256="workload",
        earliest_start_utc=MISSION4_EARLIEST_START,
        require_spec_precedes_execution=False,
    )

    assert record.git_proven_before_execution is False
    assert record.evidence_class is PreRegistrationEvidenceClass.INSUFFICIENT_EVIDENCE
    assert record.declared_status == "PRE_REGISTERED_BEFORE_EXECUTION"


def test_spec_commit_before_execution_is_git_proven(tmp_path: Path) -> None:
    _repo, spec_path, _sha = _init_spec_repo(tmp_path, commit_at="2026-09-11T00:00:00+00:00")
    snapshot = require_committed_clean_experiment_spec(spec_path)

    record = classify_pre_registration(
        declared_status="PRE_REGISTERED_BEFORE_EXECUTION",
        declared_locked_at_utc="2026-09-11T00:00:00+00:00",
        snapshot=snapshot,
        workload_sha256="workload",
        earliest_start_utc=MISSION4_EARLIEST_START,
        require_spec_precedes_execution=True,
    )

    assert record.git_proven_before_execution is True
    assert record.evidence_class is PreRegistrationEvidenceClass.GIT_PROVEN


def test_spec_commit_after_execution_fails_closed_when_required(tmp_path: Path) -> None:
    _repo, spec_path, _sha = _init_spec_repo(tmp_path, commit_at="2026-09-11T00:12:06+00:00")
    snapshot = require_committed_clean_experiment_spec(spec_path)

    with pytest.raises(ExperimentProvenanceError, match="does not precede"):
        classify_pre_registration(
            declared_status="PRE_REGISTERED_BEFORE_EXECUTION",
            declared_locked_at_utc="2026-09-11T00:00:00+00:00",
            snapshot=snapshot,
            workload_sha256="workload",
            earliest_start_utc=MISSION4_EARLIEST_START,
            require_spec_precedes_execution=True,
        )


def test_withdraw_invalid_captured_at_does_not_invent_replacement() -> None:
    corrected = withdraw_invalid_captured_at(
        {
            "captured_at_utc": MISSION4_PLACEHOLDER_CAPTURED_AT,
            "litellm_version": "1.100.1",
        },
        earliest_start_utc=MISSION4_EARLIEST_START,
        latest_end_utc=MISSION4_LATEST_END,
        generated_at_utc=MISSION4_GENERATED_AT,
    )

    assert "captured_at_utc" not in corrected
    withdrawn = corrected["withdrawn_runtime_fields"]["captured_at_utc"]
    assert withdrawn["value"] == MISSION4_PLACEHOLDER_CAPTURED_AT
    assert withdrawn["status"] == TimelineStatus.INVALID.value


def test_execution_interval_from_mission4_sanitized_payloads() -> None:
    report_dir = REPO_ROOT / "benchmarks" / "reports" / "local-json-mode-20260911"
    payloads = load_standard_logging_payloads(
        report_dir / "local-json-mode-20260911-baseline.sanitized-payloads.jsonl"
    ) + load_standard_logging_payloads(
        report_dir / "local-json-mode-20260911-candidate.sanitized-payloads.jsonl"
    )

    interval = execution_interval_from_standard_logging_payloads(payloads)

    assert interval.payload_count == 64
    assert interval.earliest_start_utc == MISSION4_EARLIEST_START
    assert interval.latest_end_utc == MISSION4_LATEST_END
    assert interval.source == "litellm_standard_logging_payload"


def test_historical_mission4_artifact_remains_readable_without_fabricated_provenance() -> None:
    report_dir = REPO_ROOT / "benchmarks" / "reports" / "local-json-mode-20260911"
    decision = json.loads((report_dir / "decision.json").read_text(encoding="utf-8"))
    runtime = json.loads((report_dir / "runtime.json").read_text(encoding="utf-8"))
    experiment = json.loads(
        (REPO_ROOT / "benchmarks" / "experiments" / "local-json-mode-20260911" / "experiment.json").read_text(
            encoding="utf-8"
        )
    )

    assert decision["final_decision"] == "inconclusive"
    assert decision["generated_at_utc"] == MISSION4_GENERATED_AT
    assert decision["quality"]["candidate_quality_pass_count"] == 8
    assert decision["quality"]["baseline_quality_pass_count"] == 0
    assert "captured_at_utc" not in decision["runtime"]
    assert "captured_at_utc" not in runtime
    withdrawn = runtime["withdrawn_runtime_fields"]["captured_at_utc"]
    assert withdrawn["value"] == MISSION4_PLACEHOLDER_CAPTURED_AT
    assert withdrawn["status"] == "invalid"
    interval = runtime["execution_interval"]
    assert interval["earliest_start_utc"] == MISSION4_EARLIEST_START
    assert interval["latest_end_utc"] == MISSION4_LATEST_END
    assert interval["payload_count"] == 64
    pre_registration = decision["pre_registration"]
    assert pre_registration["declared_status"] == "PRE_REGISTERED_BEFORE_EXECUTION"
    assert pre_registration["git_proven_before_execution"] is False
    assert pre_registration["evidence_class"] == "INSUFFICIENT_EVIDENCE"
    assert pre_registration["spec_commit_sha"] is None
    assert experiment["status"] == "PRE_REGISTERED_BEFORE_EXECUTION"
    assert experiment["pre_registration_git_proof"]["evidence_class"] == "INSUFFICIENT_EVIDENCE"
    assert experiment["pre_registration_git_proof"]["first_git_commit_sha"] == MISSION4_FIRST_GIT_COMMIT
    assert experiment["pre_registration_git_proof"]["original_spec_sha256"] == MISSION4_ORIGINAL_SPEC_SHA256
    assert datetime.fromisoformat(interval["earliest_start_utc"]).astimezone(UTC)
