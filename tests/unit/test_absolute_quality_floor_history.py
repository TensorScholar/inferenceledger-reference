from __future__ import annotations

import json
from pathlib import Path

from inference_engine.benchmarking.change_gate import (
    change_gate_policy_from_mapping,
    load_change_gate_policy,
)

_REPO = Path(__file__).resolve().parents[2]
_MISSION4_EXPERIMENT = _REPO / "benchmarks/experiments/local-json-mode-20260911/experiment.json"
_MISSION4_DECISION = _REPO / "benchmarks/reports/local-json-mode-20260911/decision.json"
_MISSION5_EXPERIMENT = _REPO / "benchmarks/experiments/ifstruct-json-mode-20260911/experiment.json"
_MISSION5_DECISION = _REPO / "benchmarks/reports/ifstruct-json-mode-20260911/decision.json"
_MISSION5_GATE = _REPO / "benchmarks/reports/ifstruct-json-mode-20260911/change-gate.json"


def test_historical_mission4_and_mission5_policies_remain_readable_without_a_floor() -> None:
    mission4_experiment = json.loads(_MISSION4_EXPERIMENT.read_text(encoding="utf-8"))
    mission5_experiment = json.loads(_MISSION5_EXPERIMENT.read_text(encoding="utf-8"))
    mission4_policy = load_change_gate_policy(_MISSION4_EXPERIMENT)
    mission5_policy = change_gate_policy_from_mapping(mission5_experiment)

    assert mission4_policy.minimum_candidate_accepted_outcome_rate is None
    assert mission5_policy.minimum_candidate_accepted_outcome_rate is None
    assert "minimum_candidate_accepted_outcome_rate" not in mission4_experiment["change_gate_policy"]
    assert "minimum_candidate_accepted_outcome_rate" not in mission5_experiment["change_gate_policy"]


def test_historical_mission5_stored_decision_remains_inconclusive() -> None:
    decision = json.loads(_MISSION5_DECISION.read_text(encoding="utf-8"))
    gate = json.loads(_MISSION5_GATE.read_text(encoding="utf-8"))
    gate_text = _MISSION5_GATE.read_text(encoding="utf-8")
    decision_text = _MISSION5_DECISION.read_text(encoding="utf-8")

    assert decision["final_decision"] == "inconclusive"
    assert gate["decision"] == "inconclusive"
    assert decision["quality"]["candidate_quality_pass_count"] == 0
    assert decision["quality"]["candidate_quality_count"] == 100
    assert decision["quality"]["baseline_quality_pass_count"] == 0
    assert "minimum_candidate_accepted_outcome_rate" not in decision["change_gate_policy"]
    assert "candidate_accepted_outcome_floor" not in gate_text
    assert "candidate_accepted_outcome_floor" not in decision_text


def test_historical_mission4_stored_decision_remains_inconclusive() -> None:
    decision = json.loads(_MISSION4_DECISION.read_text(encoding="utf-8"))
    experiment = json.loads(_MISSION4_EXPERIMENT.read_text(encoding="utf-8"))

    assert decision["final_decision"] == "inconclusive"
    assert "minimum_candidate_accepted_outcome_rate" not in experiment["change_gate_policy"]
    assert "minimum_candidate_accepted_outcome_rate" not in decision["change_gate_policy"]
