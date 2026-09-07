#!/usr/bin/env python3
"""InferenceLedger quickstart: compare two providers through the gate.

Run from the repository root:  python3 examples/quickstart.py
Stdlib only. Synthetic attempts, not production traffic or billing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))

from src.cost_model import Attempt, aggregate_attempts, non_compensatory_gate, success_rate


def main() -> None:
    baseline = [Attempt("b1", "A", "a-large", True, 0.04, 120),
                Attempt("b2", "A", "a-large", True, 0.04, 130)]
    candidate = [Attempt("c1", "B", "b-small", True, 0.012, 140),
                 Attempt("c2", "B", "b-small", True, 0.012, 150)]
    ba, ca = aggregate_attempts(baseline), aggregate_attempts(candidate)
    decision = non_compensatory_gate(
        baseline_success_rate=success_rate(ba), candidate_success_rate=success_rate(ca),
        min_success_rate=0.99, baseline_known_cost=ba.known_cost_usd,
        candidate_known_cost=ca.known_cost_usd,
        candidate_unknown_attempts=ca.unknown_attempts)
    print(f"baseline cost=${ba.known_cost_usd}  candidate cost=${ca.known_cost_usd}")
    print(f"unknown attempts: {ca.unknown_attempts}  decision: {decision}")
    print("Unknown billing is preserved, never zeroed. See experiments/ for retry/latency/gating cases.")


if __name__ == "__main__":
    main()
