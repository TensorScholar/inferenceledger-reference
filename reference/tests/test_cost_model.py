from src.cost_model import Attempt, aggregate_attempts, non_compensatory_gate
from src.scenarios import (
    gate_boundary_matrix_example,
    latency_cost_tradeoff_within_margin_example,
    latency_tradeoff_example,
    multi_provider_example,
    reliability_gating_example,
    retry_economics_example,
    uncertainty_not_zero_example,
)


def test_unknown_not_zeroed():
    assert uncertainty_not_zero_example()["pass"] is True


def test_gate_no_go_on_reliability():
    decision = non_compensatory_gate(
        baseline_success_rate=1.0,
        candidate_success_rate=0.8,
        min_success_rate=0.99,
        baseline_known_cost=1.0,
        candidate_known_cost=0.1,
        candidate_unknown_attempts=0,
    )
    assert decision == "NO-GO"


def test_scenarios():
    assert multi_provider_example()["decision"] == "SHIP"
    assert retry_economics_example()["decision"] == "NO-GO"
    assert latency_tradeoff_example()["decision"] == "NO-GO"


def test_uncertainty_propagates_to_review():
    ex = reliability_gating_example()
    assert ex["candidate_unknown"] == 1
    assert ex["decision"] == "REVIEW"


def test_bounded_tradeoff_ships():
    assert latency_cost_tradeoff_within_margin_example()["decision"] == "SHIP"


def test_gate_boundary_matrix():
    ex = gate_boundary_matrix_example()
    assert ex["pass"] is True
    assert ex["cases"]["unknown_review"] == "REVIEW"
    assert ex["cases"]["clean_win_ship"] == "SHIP"
