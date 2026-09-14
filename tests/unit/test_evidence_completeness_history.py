from __future__ import annotations

import json
from pathlib import Path

from inference_engine.benchmarking.evidence_completeness import (
    assess_decision_evidence_completeness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_decision(report_dir: str) -> dict[str, object]:
    path = REPO_ROOT / "benchmarks" / "reports" / report_dir / "decision.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_git_proven_ifstruct_history_satisfies_completeness_without_relabeling() -> None:
    payload = _load_decision("ifstruct-json-mode-20260911")

    result = assess_decision_evidence_completeness(payload)

    assert payload["final_decision"] == "inconclusive"
    assert result.contract_satisfied is True
    assert result.missing_count == 0
    assert "pre_registration" not in result.explicitly_unavailable_checks
    assert "experiment_provenance" not in result.explicitly_unavailable_checks


def test_pre_git_proven_synthetic_history_fails_closed_without_relabeling() -> None:
    payload = _load_decision("local-json-mode-20260911")

    result = assess_decision_evidence_completeness(payload)

    assert payload["final_decision"] == "inconclusive"
    assert result.contract_satisfied is False
    assert "experiment_provenance" in result.missing_checks
    assert "pre_registration" in result.explicitly_unavailable_checks
