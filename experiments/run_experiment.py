#!/usr/bin/env python3
"""Reproducible InferenceLedger reference experiments (v4).

Portfolio-wide result schema per entry:
    scenario, configuration, expected_behavior, observed_behavior,
    evidence_level, limitations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "reference"
sys.path.insert(0, str(ROOT))

from src.scenarios import (  # noqa: E402
    gate_boundary_matrix_example,
    latency_cost_tradeoff_within_margin_example,
    latency_tradeoff_example,
    multi_provider_example,
    reliability_gating_example,
    retry_economics_example,
    uncertainty_not_zero_example,
)

EVIDENCE_LEVEL = "L2"
LIMITATIONS = [
    "Educational model with synthetic attempt streams, not production traffic.",
    "Historical 5-request smoke remains a separate L2 frozen artifact and is not re-interpreted here.",
    "Latency figures are example p95 inputs, not measured production percentiles.",
    "No billing-grade cost claim; unknown billing is preserved, never priced.",
]
CONFIGURATION = "attempt-aware cost model, non-compensatory gate, deterministic synthetic attempts"


def to_record(ex: dict) -> dict:
    name = ex["name"]
    if "decision" in ex:
        passed = ex["decision"] == ex["expect"]
        expected = f"gate verdict = {ex['expect']}"
        observed = {k: v for k, v in ex.items() if k not in {"name", "expect"}}
    else:
        passed = bool(ex.get("pass"))
        expected = f"pass = {ex.get('expect_pass')}"
        observed = {k: v for k, v in ex.items() if k not in {"name", "expect_pass"}}
    return {
        "scenario": name,
        "configuration": CONFIGURATION,
        "expected_behavior": expected,
        "observed_behavior": observed,
        "pass": passed,
        "evidence_level": EVIDENCE_LEVEL,
        "limitations": list(LIMITATIONS),
    }


def main() -> int:
    examples = [
        multi_provider_example(),
        retry_economics_example(),
        latency_tradeoff_example(),
        latency_cost_tradeoff_within_margin_example(),
        reliability_gating_example(),
        uncertainty_not_zero_example(),
        gate_boundary_matrix_example(),
    ]
    results = [to_record(ex) for ex in examples]
    payload = {
        "project": "InferenceLedger",
        "artifact": "reference-cost-model",
        "evidence_maturity": EVIDENCE_LEVEL,
        "note": "Educational model. Historical 5-request smoke remains a separate L2 frozen artifact "
        "and is not re-interpreted here.",
        "results": results,
        "all_pass": all(r["pass"] for r in results),
    }
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
