#!/usr/bin/env python3
"""Freeze a deterministic IFStruct v1.0 JSON-wrapper subset.

This script does not call models. It records provenance and writes the workload
that later executions must hash-match.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from inference_engine.benchmarking.ifstruct_subset import (
    JSON_WRAPPER_SUBSET_ID,
    JSON_WRAPPER_SUBSET_LIMIT,
    select_json_wrapper_prefix,
    workload_item_id,
)

PINNED_SOURCE_COMMIT = "1948dda22fb08bb1fe15eeb5332119d6087889ac"
PINNED_SOURCE_URL = (
    "https://raw.githubusercontent.com/Liquid4All/ifstruct/"
    f"{PINNED_SOURCE_COMMIT}/data/test.jsonl"
)
PINNED_SOURCE_SHA256 = "2a62e39293040aed7db6690bfa31c764cb761f71daf0ae0bd4abc4249f5f574e"
UPSTREAM_REPO = "https://github.com/Liquid4All/ifstruct"
UPSTREAM_DATASET = "https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0"
LICENSE_NAME = "Apache-2.0"


def main() -> int:
    parser = argparse.ArgumentParser(prog="freeze_ifstruct_subset")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument(
        "--output-dir",
        default="benchmarks/references/ifstruct-v1.0",
    )
    parser.add_argument(
        "--workload-path",
        default="benchmarks/workloads/ifstruct-v1.0-json-wrapper-n100.jsonl",
    )
    parser.add_argument("--limit", type=int, default=JSON_WRAPPER_SUBSET_LIMIT)
    parser.add_argument("--acquisition-date", default=datetime.now(tz=UTC).date().isoformat())
    args = parser.parse_args()
    return _freeze(args)


def _freeze(args: argparse.Namespace) -> int:
    source_path = Path(args.source_jsonl)
    source_bytes = source_path.read_bytes()
    source_sha = sha256(source_bytes).hexdigest()
    if source_sha != PINNED_SOURCE_SHA256:
        raise SystemExit(
            "source test.jsonl SHA-256 does not match the pinned IFStruct revision; "
            f"expected={PINNED_SOURCE_SHA256}, actual={source_sha}"
        )
    rows = [
        _parse_row(line, index)
        for index, line in enumerate(source_path.read_text().splitlines(), 1)
        if line.strip()
    ]
    eligible_count = sum(
        1
        for row in rows
        if row.get("output_format") == "json" and row.get("require_wrapper_key") is True
    )
    selected = select_json_wrapper_prefix(rows, limit=args.limit)
    workload_rows = [_to_workload_item(row) for row in selected]
    workload_path = Path(args.workload_path)
    workload_path.parent.mkdir(parents=True, exist_ok=True)
    workload_text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in workload_rows)
    workload_path.write_text(workload_text, encoding="utf-8")
    subset_sha = sha256(workload_path.read_bytes()).hexdigest()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "benchmark_name": "IFStruct",
        "benchmark_version": "v1.0",
        "classification": "PUBLISHED_REFERENCE",
        "upstream_repository": UPSTREAM_REPO,
        "upstream_dataset": UPSTREAM_DATASET,
        "source_commit": PINNED_SOURCE_COMMIT,
        "source_url": PINNED_SOURCE_URL,
        "source_artifact": "data/test.jsonl",
        "source_item_count": len(rows),
        "source_sha256": source_sha,
        "license": LICENSE_NAME,
        "acquisition_date": args.acquisition_date,
        "redistribution": (
            "Apache-2.0 permits reproducing this subset with attribution. Item prompts "
            "and schemas are selected, not rewritten."
        ),
        "subset": {
            "identity": JSON_WRAPPER_SUBSET_ID,
            "item_count": len(selected),
            "sha256": subset_sha,
            "workload_path": workload_path.as_posix(),
            "selection_rule": {
                "source": "pinned data/test.jsonl",
                "predicate": [
                    "output_format == json",
                    "require_wrapper_key is True",
                ],
                "order": "ascending (seed, entity_type)",
                "take": args.limit,
                "cherry_pick": False,
                "model_outputs_not_observed": True,
            },
            "eligible_json_wrapper_count": eligible_count,
            "seeds": [row["seed"] for row in selected],
        },
        "scoring_semantics": {
            "measures": [
                "JSON parse",
                "optional fenced code block",
                "optional no-commentary",
                "top-level wrapper object and key",
                "item count or range",
                "JSON Schema types, required fields, enums, numeric bounds, extraneous fields",
            ],
            "does_not_measure": [
                "semantic quality",
                "factual correctness of generated values beyond schema/enum/range constraints",
                "helpfulness",
                "official IFStruct leaderboard protocol (YAML + unconstrained decoding)",
            ],
            "pass_rule": "binary; all six check stages must produce zero errors",
        },
        "technical_applicability_filter": {
            "reason": (
                "Candidate response_format=json_object cannot express YAML or a bare JSON array. "
                "The subset is therefore JSON + wrapper-object only."
            ),
            "json_object_is_not_json_schema_enforcement": True,
        },
    }
    (output_dir / "SOURCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        " ".join(
            [
                f"source_sha256={source_sha}",
                f"subset_items={len(selected)}",
                f"subset_sha256={subset_sha}",
                f"workload_path={workload_path}",
            ]
        )
    )
    return 0


def _parse_row(line: str, index: int) -> dict[str, Any]:
    parsed = json.loads(line)
    if not isinstance(parsed, dict):
        raise ValueError(f"source line {index} must be a JSON object")
    return parsed


def _to_workload_item(row: dict[str, Any]) -> dict[str, Any]:
    seed = row["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"invalid IFStruct seed: {seed!r}")
    top_level_key = row["top_level_key"]
    if not isinstance(top_level_key, str) or not top_level_key.strip():
        raise ValueError(f"wrapper subset row seed={seed} missing top_level_key")
    return {
        "id": workload_item_id(seed),
        "prompt": row["prompt"],
        "tags": {
            "suite": JSON_WRAPPER_SUBSET_ID,
            "output_format": "json",
            "output_contract": "json",
            "entity_type": str(row["entity_type"]),
            "require_code_block": _bool_tag(row["require_code_block"]),
            "require_no_commentary": _bool_tag(row["require_no_commentary"]),
            "require_wrapper_key": "true",
            "top_level_key": top_level_key,
        },
        "eval": {
            "type": "ifstruct_structure",
            "ifstruct": {
                "json_schema": row["json_schema"],
                "output_format": "json",
                "require_code_block": row["require_code_block"],
                "require_no_commentary": row["require_no_commentary"],
                "require_wrapper_key": True,
                "top_level_count": row["top_level_count"],
                "top_level_key": top_level_key,
            },
        },
    }


def _bool_tag(value: object) -> str:
    if not isinstance(value, bool):
        raise ValueError(f"expected boolean tag source, got {value!r}")
    return "true" if value else "false"


if __name__ == "__main__":
    raise SystemExit(main())
