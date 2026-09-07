"""Repo-level evidence-schema tests (run from the repository root).

Guards the public contract: claims ledger shape, maturity vocabulary,
and experiment results schema. These run in CI via `make verify`.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MATURITY = {"L0", "L1", "L2", "L3", "Conceptual"}

# results.json locations relative to repo root: (path, allow_L3)
RESULT_FILES = {
    "agentguard-showcase": [("reference-kernel/experiments/results.json", False)],
    "proofdiff-showcase": [("experiments/results.json", True)],
    "permitdiff-showcase": [("experiments/results.json", False)],
    "inferenceledger-reference": [("experiments/results.json", False)],
}

REQUIRED_CLAIM_FIELDS = {"id", "statement", "evidence_maturity", "scope", "falsification_condition"}
REQUIRED_RESULT_FIELDS = {"scenario", "configuration", "expected_behavior",
                          "observed_behavior", "evidence_level", "limitations", "pass"}


def repo_name() -> str:
    return ROOT.name


def test_claims_ledger_schema():
    claims = json.loads((ROOT / "evidence" / "claims.json").read_text(encoding="utf-8"))
    assert claims["claims"], "claims ledger must not be empty"
    ids = set()
    for c in claims["claims"]:
        missing = REQUIRED_CLAIM_FIELDS - set(c.keys())
        assert not missing, f"{c.get('id')}: missing {missing}"
        assert c["id"] not in ids, f"duplicate claim id {c['id']}"
        ids.add(c["id"])
        assert c["evidence_maturity"] in MATURITY, f"{c['id']}: bad maturity"


def test_no_unexpected_L3():
    name = repo_name()
    if name == "proofdiff-showcase":
        return  # the single preregistered N=1 pilot is the documented exception
    claims = json.loads((ROOT / "evidence" / "claims.json").read_text(encoding="utf-8"))
    l3 = [c["id"] for c in claims["claims"] if c.get("evidence_maturity") == "L3"]
    assert not l3, f"unexpected L3 claims (no external validation in this repo): {l3}"


def test_results_schema():
    name = repo_name()
    for rel, _ in RESULT_FILES.get(name, []):
        payload = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        assert payload.get("results"), f"{rel}: empty results"
        for r in payload["results"]:
            missing = REQUIRED_RESULT_FIELDS - set(r.keys())
            assert not missing, f"{r.get('scenario')}: missing {missing}"
        assert payload.get("all_pass") is True, f"{rel}: all_pass is not true"
