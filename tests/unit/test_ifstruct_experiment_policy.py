from __future__ import annotations

import json
from pathlib import Path

from inference_engine.benchmarking.change_gate import change_gate_policy_from_mapping

_REPO = Path(__file__).resolve().parents[2]


def test_pre_registered_ifstruct_policy_omits_operational_deltas() -> None:
    experiment = json.loads(
        (
            _REPO / "benchmarks/experiments/ifstruct-json-mode-20260911/experiment.json"
        ).read_text(encoding="utf-8")
    )
    policy = change_gate_policy_from_mapping(experiment)

    assert experiment["classification"] == "PUBLISHED_REFERENCE"
    assert experiment["workload"]["sha256"] == (
        "35648eb88ea2b07008113e913b406186f5c9399cff3cf88bf853600fed2f7192"
    )
    assert experiment["generation"]["max_tokens"] == 1024
    assert experiment["candidate"]["response_format"] == {"type": "json_object"}
    assert policy.max_mean_successful_latency_delta_ms == 15000.0
    assert policy.max_failure_harm_rate == 0.05
    assert policy.max_accepted_outcome_harm_rate == 0.05
    assert policy.max_mean_provider_attempt_delta is None
    assert policy.max_mean_provider_retry_delta is None
    assert policy.minimum_candidate_accepted_outcome_rate is None
    assert "minimum_candidate_accepted_outcome_rate" not in experiment["change_gate_policy"]
    assert policy.critical_segments[0].tag_key == "output_contract"
