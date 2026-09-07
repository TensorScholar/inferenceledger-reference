"""Worked examples: multi-provider, retries, uncertainty, latency tradeoff."""

from __future__ import annotations

from .cost_model import Attempt, aggregate_attempts, non_compensatory_gate, success_rate


def multi_provider_example() -> dict:
    """Baseline: provider A. Candidate: cheaper provider B, same successes."""
    baseline = [
        Attempt("b1", "A", "a-large", True, 0.04, 120),
        Attempt("b2", "A", "a-large", True, 0.04, 130),
        Attempt("b3", "A", "a-large", True, 0.04, 110),
    ]
    candidate = [
        Attempt("c1", "B", "b-small", True, 0.012, 140),
        Attempt("c2", "B", "b-small", True, 0.012, 150),
        Attempt("c3", "B", "b-small", True, 0.012, 135),
    ]
    ba, ca = aggregate_attempts(baseline), aggregate_attempts(candidate)
    decision = non_compensatory_gate(
        baseline_success_rate=success_rate(ba),
        candidate_success_rate=success_rate(ca),
        min_success_rate=0.99,
        baseline_known_cost=ba.known_cost_usd,
        candidate_known_cost=ca.known_cost_usd,
        candidate_unknown_attempts=ca.unknown_attempts,
        baseline_p95_latency_ms=130,
        candidate_p95_latency_ms=150,
    )
    return {
        "name": "multi_provider_cheaper_ok",
        "baseline_cost": ba.known_cost_usd,
        "candidate_cost": ca.known_cost_usd,
        "decision": decision,
        "expect": "SHIP",
    }


def retry_economics_example() -> dict:
    """Candidate looks cheaper per success but burns retries (and one unknown)."""
    baseline = [
        Attempt("b1", "A", "a-large", True, 0.03, 100),
        Attempt("b2", "A", "a-large", True, 0.03, 100),
    ]
    candidate = [
        Attempt("c1", "B", "b-flaky", False, 0.005, 400, is_retry=False, error="timeout"),
        Attempt("c1r", "B", "b-flaky", True, 0.005, 420, is_retry=True),
        Attempt("c2", "B", "b-flaky", False, None, 500, is_retry=False, error="unknown_billing"),
        Attempt("c2r", "B", "b-flaky", True, 0.005, 200, is_retry=True),
    ]
    ba, ca = aggregate_attempts(baseline), aggregate_attempts(candidate)
    decision = non_compensatory_gate(
        baseline_success_rate=1.0,
        candidate_success_rate=0.5,  # 2 success / 4 attempts
        min_success_rate=0.99,
        baseline_known_cost=ba.known_cost_usd,
        candidate_known_cost=ca.known_cost_usd,
        candidate_unknown_attempts=ca.unknown_attempts,
    )
    return {
        "name": "retry_economics_unknown",
        "candidate_retries": ca.retry_count,
        "candidate_unknown": ca.unknown_attempts,
        "known_cost_not_zeroed": ca.known_cost_usd == 0.015,
        "decision": decision,
        "expect": "NO-GO",
    }


def latency_tradeoff_example() -> dict:
    """Cheaper but p95 latency regresses beyond the allowed margin."""
    decision = non_compensatory_gate(
        baseline_success_rate=1.0,
        candidate_success_rate=1.0,
        min_success_rate=0.99,
        baseline_known_cost=1.0,
        candidate_known_cost=0.2,
        candidate_unknown_attempts=0,
        baseline_p95_latency_ms=200,
        candidate_p95_latency_ms=900,
        max_latency_regression_ms=200,
    )
    return {
        "name": "latency_regression_blocks_cost_win",
        "decision": decision,
        "expect": "NO-GO",
    }


def uncertainty_not_zero_example() -> dict:
    attempts = [
        Attempt("1", "A", "m", True, 0.01, 80),
        Attempt("2", "A", "m", False, None, 90, error="provider_partial"),
    ]
    agg = aggregate_attempts(attempts)
    return {
        "name": "unknown_cost_preserved",
        "known_cost_usd": agg.known_cost_usd,
        "unknown_attempts": agg.unknown_attempts,
        "pass": agg.known_cost_usd == 0.01 and agg.unknown_attempts == 1,
        "expect_pass": True,
    }


def reliability_gating_example() -> dict:
    """Candidate meets the reliability bar but one attempt has unknown billing.

    Uncertainty propagation: unknown cost cannot SHIP even when reliability
    and latency pass. The gate must return REVIEW so a human prices the
    unknown before migration.
    """
    baseline = [
        Attempt("b1", "A", "a-large", True, 0.03, 100),
        Attempt("b2", "A", "a-large", True, 0.03, 100),
    ]
    candidate = [
        Attempt("c1", "B", "b-new", True, 0.01, 105),
        Attempt("c2", "B", "b-new", True, None, 110, error="usage_missing"),
    ]
    ba, ca = aggregate_attempts(baseline), aggregate_attempts(candidate)
    decision = non_compensatory_gate(
        baseline_success_rate=success_rate(ba),
        candidate_success_rate=success_rate(ca),
        min_success_rate=0.99,
        baseline_known_cost=ba.known_cost_usd,
        candidate_known_cost=ca.known_cost_usd,
        candidate_unknown_attempts=ca.unknown_attempts,
        baseline_p95_latency_ms=100,
        candidate_p95_latency_ms=110,
    )
    return {
        "name": "reliability_gating_unknown_forces_review",
        "baseline_success": success_rate(ba),
        "candidate_success": success_rate(ca),
        "candidate_unknown": ca.unknown_attempts,
        "decision": decision,
        "expect": "REVIEW",
    }


def latency_cost_tradeoff_within_margin_example() -> dict:
    """Cheaper candidate, slightly slower but inside the latency margin: SHIP.

    Pairs with latency_tradeoff_example (regression beyond margin -> NO-GO).
    Together they pin the tradeoff boundary: cost may buy latency only
    within the pre-declared regression budget.
    """
    decision = non_compensatory_gate(
        baseline_success_rate=1.0,
        candidate_success_rate=1.0,
        min_success_rate=0.99,
        baseline_known_cost=1.0,
        candidate_known_cost=0.4,
        candidate_unknown_attempts=0,
        baseline_p95_latency_ms=200,
        candidate_p95_latency_ms=280,
        max_latency_regression_ms=200,
    )
    return {
        "name": "latency_cost_tradeoff_within_margin",
        "decision": decision,
        "expect": "SHIP",
    }


def gate_boundary_matrix_example() -> dict:
    """Sweep the gate across its decision boundaries (deterministic).

    Pins: reliability miss -> NO-GO despite 10x cost win; latency breach ->
    NO-GO despite cost win; unknown -> REVIEW despite full reliability;
    clean win -> SHIP. Returns per-case verdicts for the runner to check.
    """
    cases = {
        "reliability_miss_no_go": non_compensatory_gate(
            baseline_success_rate=1.0, candidate_success_rate=0.8,
            min_success_rate=0.99, baseline_known_cost=1.0,
            candidate_known_cost=0.1, candidate_unknown_attempts=0,
        ),
        "latency_breach_no_go": non_compensatory_gate(
            baseline_success_rate=1.0, candidate_success_rate=1.0,
            min_success_rate=0.99, baseline_known_cost=1.0,
            candidate_known_cost=0.1, candidate_unknown_attempts=0,
            baseline_p95_latency_ms=200, candidate_p95_latency_ms=900,
            max_latency_regression_ms=200,
        ),
        "unknown_review": non_compensatory_gate(
            baseline_success_rate=1.0, candidate_success_rate=1.0,
            min_success_rate=0.99, baseline_known_cost=1.0,
            candidate_known_cost=0.5, candidate_unknown_attempts=2,
        ),
        "clean_win_ship": non_compensatory_gate(
            baseline_success_rate=1.0, candidate_success_rate=1.0,
            min_success_rate=0.99, baseline_known_cost=1.0,
            candidate_known_cost=0.5, candidate_unknown_attempts=0,
            baseline_p95_latency_ms=200, candidate_p95_latency_ms=210,
            max_latency_regression_ms=200,
        ),
    }
    expected = {
        "reliability_miss_no_go": "NO-GO",
        "latency_breach_no_go": "NO-GO",
        "unknown_review": "REVIEW",
        "clean_win_ship": "SHIP",
    }
    return {
        "name": "gate_boundary_matrix",
        "cases": cases,
        "expect_cases": expected,
        "pass": cases == expected,
        "expect_pass": True,
    }
