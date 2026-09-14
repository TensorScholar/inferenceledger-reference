from __future__ import annotations

from pathlib import Path

import pytest

from inference_engine.benchmarking.change_gate import load_change_gate_policy
from inference_engine.benchmarking.paired_comparison import audit_workload_pairing
from inference_engine.benchmarking.segmentation import BenchmarkRequestContext


def _context(prefix: str, item_id: str, request_id: str | None = None) -> BenchmarkRequestContext:
    return BenchmarkRequestContext.from_tags(
        request_id=request_id or f"{prefix}-{item_id}",
        workload_item_id=item_id,
        tags={"task": "qa"},
    )


def test_pairing_audit_reports_full_coverage() -> None:
    baseline = [_context("b", "item-1"), _context("b", "item-2")]
    candidate = [_context("c", "item-1"), _context("c", "item-2")]

    audit = audit_workload_pairing(baseline_contexts=baseline, candidate_contexts=candidate)

    assert audit.baseline_count == 2
    assert audit.candidate_count == 2
    assert audit.matched_pair_count == 2
    assert audit.coverage == 1.0
    assert audit.ambiguous is False
    assert audit.fail_closed_reason is None


def test_pairing_audit_reports_unmatched_ids_and_fail_closes() -> None:
    baseline = [_context("b", "item-1"), _context("b", "item-2")]
    candidate = [_context("c", "item-1"), _context("c", "item-3")]

    audit = audit_workload_pairing(baseline_contexts=baseline, candidate_contexts=candidate)

    assert audit.matched_pair_count == 1
    assert audit.unmatched_baseline_item_ids == ("item-2",)
    assert audit.unmatched_candidate_item_ids == ("item-3",)
    assert audit.ambiguous is True
    assert audit.fail_closed_reason is not None


def test_pairing_audit_reports_duplicate_ids_and_fail_closes() -> None:
    baseline = [
        _context("b", "item-1", request_id="b-1"),
        _context("b", "item-1", request_id="b-2"),
    ]
    candidate = [_context("c", "item-1"), _context("c", "item-2")]

    audit = audit_workload_pairing(baseline_contexts=baseline, candidate_contexts=candidate)

    assert audit.duplicate_baseline_item_ids == ("item-1",)
    assert audit.ambiguous is True


def test_experiment_policy_json_loads_without_unknown_fields() -> None:
    path = Path("benchmarks/experiments/local-json-mode-20260911/experiment.json")
    policy = load_change_gate_policy(path)

    assert policy.max_mean_cost_delta_usd == 0.0
    assert policy.max_failure_harm_rate == 0.05
    assert policy.max_mean_successful_latency_delta_ms == 750.0
    assert policy.require_tail_latency_inference is False
    assert policy.critical_segments[0].tag_key == "output_contract"
    assert policy.critical_segments[0].tag_value == "json"
    assert policy.minimum_candidate_accepted_outcome_rate is None


def test_unknown_policy_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        '{"max_mean_cost_delta_usd": 0, "max_failure_harm_rate": 0.05, '
        '"max_mean_successful_latency_delta_ms": 1, "max_accepted_outcome_harm_rate": 0.05, '
        '"bonus_metric": 1}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown ChangeGatePolicy fields"):
        load_change_gate_policy(path)
