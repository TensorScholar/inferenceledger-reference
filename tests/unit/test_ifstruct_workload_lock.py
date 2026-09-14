from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from inference_engine.benchmarking.eval import EvalType
from inference_engine.benchmarking.harness import load_workload

_REPO = Path(__file__).resolve().parents[2]


def test_committed_ifstruct_subset_matches_source_manifest() -> None:
    source = json.loads(
        (_REPO / "benchmarks/references/ifstruct-v1.0/SOURCE.json").read_text(encoding="utf-8")
    )
    workload_path = _REPO / str(source["subset"]["workload_path"])
    actual_sha = sha256(workload_path.read_bytes()).hexdigest()

    assert source["source_sha256"] == (
        "2a62e39293040aed7db6690bfa31c764cb761f71daf0ae0bd4abc4249f5f574e"
    )
    assert source["source_commit"] == "1948dda22fb08bb1fe15eeb5332119d6087889ac"
    assert actual_sha == source["subset"]["sha256"]
    assert source["subset"]["item_count"] == 100

    items = load_workload(workload_path)
    assert len(items) == 100
    assert items[0].id == "ifstruct-1"
    assert all(item.eval_spec is not None for item in items)
    assert all(item.eval_spec.eval_type == EvalType.IFSTRUCT_STRUCTURE for item in items if item.eval_spec)
    assert all(item.tags["output_contract"] == "json" for item in items)
    assert all(item.tags["require_wrapper_key"] == "true" for item in items)
