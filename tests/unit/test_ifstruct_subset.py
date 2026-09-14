from __future__ import annotations

import pytest

from inference_engine.benchmarking.ifstruct_subset import (
    select_json_wrapper_prefix,
    workload_item_id,
)


def test_select_json_wrapper_prefix_is_deterministic_and_ignores_yaml_and_bare_lists() -> None:
    rows = [
        {
            "seed": 20,
            "entity_type": "b",
            "output_format": "json",
            "require_wrapper_key": True,
        },
        {
            "seed": 10,
            "entity_type": "a",
            "output_format": "yaml",
            "require_wrapper_key": True,
        },
        {
            "seed": 10,
            "entity_type": "a",
            "output_format": "json",
            "require_wrapper_key": False,
        },
        {
            "seed": 5,
            "entity_type": "z",
            "output_format": "json",
            "require_wrapper_key": True,
        },
        {
            "seed": 20,
            "entity_type": "a",
            "output_format": "json",
            "require_wrapper_key": True,
        },
    ]

    selected = select_json_wrapper_prefix(rows, limit=2)

    assert [item["seed"] for item in selected] == [5, 20]
    assert [item["entity_type"] for item in selected] == ["z", "a"]
    assert select_json_wrapper_prefix(rows, limit=2) == selected


def test_select_json_wrapper_prefix_fails_closed_when_too_few_rows() -> None:
    rows = [
        {
            "seed": 1,
            "entity_type": "a",
            "output_format": "json",
            "require_wrapper_key": True,
        }
    ]

    with pytest.raises(ValueError, match="below requested limit"):
        select_json_wrapper_prefix(rows, limit=2)


def test_workload_item_id_uses_seed() -> None:
    assert workload_item_id(12) == "ifstruct-12"
