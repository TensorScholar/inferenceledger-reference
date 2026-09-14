from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import scripts.decide_change as decide_cli
from inference_engine.benchmarking.context_store import SQLiteBenchmarkContextStore
from inference_engine.benchmarking.harness import BenchmarkReport
from inference_engine.benchmarking.segmentation import BenchmarkRequestContext
from inference_engine.benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from inference_engine.infrastructure.telemetry.request_log import RequestTrace


def _report(*, request_count: int, provider: str, workload_sha256: str) -> BenchmarkReport:
    return BenchmarkReport(
        workload_path="benchmarks/workloads/json-contract-synthetic-v1.jsonl",
        workload_sha256=workload_sha256,
        strategy="litellm_baseline",
        provider=provider,
        model="gemma3:270m",
        request_count=request_count,
        success_count=request_count,
        failure_count=0,
        error_rate=0.0,
        latency_p50_ms=100,
        latency_p95_ms=120,
        prompt_tokens=50,
        completion_tokens=25,
        total_tokens=75,
        estimated_cost_usd=0.0,
        cost_evidence_complete=True,
        provider_attempt_count=request_count,
        provider_retry_count=0,
        route_count=0,
        budget_violation_count=0,
        model_distribution={"gemma3:270m": request_count},
        route_reason_distribution={},
        observed_latency_ms_by_model={"gemma3:270m": {"count": request_count, "p50": 100, "p95": 120}},
        quality_count=request_count,
        quality_pass_count=request_count,
        quality_pass_rate=1.0,
        quality_score_avg=1.0,
        ledger_path="ledger.jsonl",
        limitations=[],
    )


def _trace(*, request_id: str, latency_ms: int) -> RequestTrace:
    return RequestTrace(
        request_id=request_id,
        provider="ollama",
        model="gemma3:270m",
        latency_ms=latency_ms,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.0,
        pricing_table_version="external_reported:litellm_standard_logging_payload",
        cache_hit=None,
        error_type=None,
        error_message=None,
        timestamp="2026-09-11T00:00:00+00:00",
        quality_passed=True,
        quality_score=1.0,
        quality_reason="all expected JSON fields matched",
        eval_type="json_field_equals",
        provider_attempt_count=1,
        provider_retry_count=0,
        cost_evidence_complete=True,
    )


_GIT_TMP = Path(__file__).resolve().parents[2] / ".tmp-pytest-git"


def _git_repo_dir(tmp_path: Path) -> Path:
    root = _GIT_TMP / tmp_path.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _write_experiment(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment_id": "fixture-decision",
                "status": "PRE_REGISTERED_BEFORE_EXECUTION",
                "locked_at_utc": "2026-09-11T00:00:00+00:00",
                "classification": "SYNTHETIC",
                "workload": {
                    "path": "benchmarks/workloads/json-contract-synthetic-v1.jsonl",
                    "sha256": "fixture-sha",
                    "item_count": 2,
                },
                "baseline": {"run_id": "baseline", "definition": "fixture", "rationale": "test"},
                "candidate": {"run_id": "candidate", "definition": "fixture", "rationale": "test"},
                "statistics": {
                    "confidence_level": 0.95,
                    "bootstrap_iterations": 1000,
                    "minimum_samples": 2,
                    "seed": 7,
                },
                "change_gate_policy": {
                    "max_mean_cost_delta_usd": 0.0,
                    "max_failure_harm_rate": 0.05,
                    "max_mean_successful_latency_delta_ms": 750.0,
                    "max_accepted_outcome_harm_rate": 0.05,
                    "minimum_bootstrap_iterations": 1000,
                    "require_tail_latency_inference": False,
                },
                "unsupported_claims_before_results": ["production proven"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _commit_experiment(path: Path) -> None:
    repo = path.parent
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    (repo / ".empty-git-template").mkdir(exist_ok=True)
    completed = subprocess.run(
        ["git", "init", "--template", str(repo / ".empty-git-template")],
        cwd=repo,
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(f"git init failed: {completed.stderr.strip()}")
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=provenance-test@example.com",
            "-c",
            "user.name=Provenance Test",
            "add",
            path.name,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=provenance-test@example.com",
            "-c",
            "user.name=Provenance Test",
            "commit",
            "-m",
            "lock experiment spec",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        env=env,
    )


def test_decide_change_writes_reconstructable_bundle(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(ledger_path)
    context_store = SQLiteBenchmarkContextStore(ledger_path)
    baseline_contexts = [
        BenchmarkRequestContext.from_tags(
            request_id="b-1",
            workload_item_id="item-1",
            tags={"output_contract": "json"},
        ),
        BenchmarkRequestContext.from_tags(
            request_id="b-2",
            workload_item_id="item-2",
            tags={"output_contract": "json"},
        ),
    ]
    candidate_contexts = [
        BenchmarkRequestContext.from_tags(
            request_id="c-1",
            workload_item_id="item-1",
            tags={"output_contract": "json"},
        ),
        BenchmarkRequestContext.from_tags(
            request_id="c-2",
            workload_item_id="item-2",
            tags={"output_contract": "json"},
        ),
    ]
    ledger.record_run(
        run_id="baseline",
        report=_report(request_count=2, provider="ollama", workload_sha256="fixture-sha"),
        traces=[_trace(request_id="b-1", latency_ms=100), _trace(request_id="b-2", latency_ms=110)],
    )
    context_store.record_contexts(run_id="baseline", contexts=baseline_contexts)
    ledger.record_run(
        run_id="candidate",
        report=_report(request_count=2, provider="ollama", workload_sha256="fixture-sha"),
        traces=[_trace(request_id="c-1", latency_ms=105), _trace(request_id="c-2", latency_ms=108)],
    )
    context_store.record_contexts(run_id="candidate", contexts=candidate_contexts)

    experiment_path = _git_repo_dir(tmp_path) / "experiment.json"
    _write_experiment(experiment_path)
    _commit_experiment(experiment_path)
    output_dir = tmp_path / "decision"
    exit_code = decide_cli._run(
        argparse.Namespace(
            experiment_path=str(experiment_path),
            sqlite_ledger_path=str(ledger_path),
            output_dir=str(output_dir),
            runtime_json=None,
        )
    )

    assert exit_code == 0
    decision = json.loads((output_dir / "decision.json").read_text(encoding="utf-8"))
    pairing = json.loads((output_dir / "pairing-audit.json").read_text(encoding="utf-8"))
    gate = json.loads((output_dir / "change-gate.json").read_text(encoding="utf-8"))
    assert pairing["matched_pair_count"] == 2
    assert pairing["ambiguous"] is False
    assert decision["final_decision"] == gate["decision"]
    floor_check = next(
        check
        for check in gate["checks"]
        if check["check_id"] == "overall:candidate_accepted_outcome_floor"
    )
    assert floor_check["status"] == "inconclusive"
    assert floor_check["threshold"] is None
    assert gate["decision"] == "inconclusive"
    assert decision["pre_registration"]["spec_committed_and_clean"] is True
    assert decision["pre_registration"]["git_proven_before_execution"] is False
    assert decision["pre_registration"]["evidence_class"] == "INSUFFICIENT_EVIDENCE"
    assert decision["experiment_provenance"]["spec_sha256"]
    assert (output_dir / "decision.md").exists()
    assert (output_dir / "paired-evidence.json").exists()


def test_decide_change_fail_closes_on_unmatched_ids(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(ledger_path)
    context_store = SQLiteBenchmarkContextStore(ledger_path)
    baseline_contexts = [
        BenchmarkRequestContext.from_tags(
            request_id="b-1",
            workload_item_id="item-1",
            tags={"output_contract": "json"},
        )
    ]
    candidate_contexts = [
        BenchmarkRequestContext.from_tags(
            request_id="c-1",
            workload_item_id="item-2",
            tags={"output_contract": "json"},
        )
    ]
    ledger.record_run(
        run_id="baseline",
        report=_report(request_count=1, provider="ollama", workload_sha256="fixture-sha"),
        traces=[_trace(request_id="b-1", latency_ms=100)],
    )
    context_store.record_contexts(run_id="baseline", contexts=baseline_contexts)
    ledger.record_run(
        run_id="candidate",
        report=_report(request_count=1, provider="ollama", workload_sha256="fixture-sha"),
        traces=[_trace(request_id="c-1", latency_ms=100)],
    )
    context_store.record_contexts(run_id="candidate", contexts=candidate_contexts)
    experiment_path = _git_repo_dir(tmp_path) / "experiment.json"
    _write_experiment(experiment_path)
    _commit_experiment(experiment_path)
    output_dir = tmp_path / "decision"

    exit_code = decide_cli._run(
        argparse.Namespace(
            experiment_path=str(experiment_path),
            sqlite_ledger_path=str(ledger_path),
            output_dir=str(output_dir),
            runtime_json=None,
        )
    )

    assert exit_code == 2
    pairing = json.loads((output_dir / "pairing-audit.json").read_text(encoding="utf-8"))
    assert pairing["ambiguous"] is True
    assert not (output_dir / "decision.json").exists()


def test_decide_change_fails_closed_on_dirty_experiment_spec(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = SQLiteBenchmarkLedger(ledger_path)
    context_store = SQLiteBenchmarkContextStore(ledger_path)
    ledger.record_run(
        run_id="baseline",
        report=_report(request_count=1, provider="ollama", workload_sha256="fixture-sha"),
        traces=[_trace(request_id="b-1", latency_ms=100)],
    )
    context_store.record_contexts(
        run_id="baseline",
        contexts=[
            BenchmarkRequestContext.from_tags(
                request_id="b-1",
                workload_item_id="item-1",
                tags={"output_contract": "json"},
            )
        ],
    )
    ledger.record_run(
        run_id="candidate",
        report=_report(request_count=1, provider="ollama", workload_sha256="fixture-sha"),
        traces=[_trace(request_id="c-1", latency_ms=100)],
    )
    context_store.record_contexts(
        run_id="candidate",
        contexts=[
            BenchmarkRequestContext.from_tags(
                request_id="c-1",
                workload_item_id="item-1",
                tags={"output_contract": "json"},
            )
        ],
    )
    experiment_path = _git_repo_dir(tmp_path) / "experiment.json"
    _write_experiment(experiment_path)
    _commit_experiment(experiment_path)
    experiment_path.write_text(experiment_path.read_text(encoding="utf-8").replace("fixture", "mutated"), encoding="utf-8")

    exit_code = decide_cli._run(
        argparse.Namespace(
            experiment_path=str(experiment_path),
            sqlite_ledger_path=str(ledger_path),
            output_dir=str(tmp_path / "decision"),
            runtime_json=None,
        )
    )

    assert exit_code == 2
    assert not (tmp_path / "decision" / "decision.json").exists()
