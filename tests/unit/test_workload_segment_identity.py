from __future__ import annotations

import pytest

from inference_engine.benchmarking.harness import load_workload


def test_load_workload_rejects_duplicate_item_ids_before_execution(tmp_path) -> None:
    path = tmp_path / "workload.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id":"same","prompt":"one","tags":{"task":"qa"}}',
                '{"id":"same","prompt":"two","tags":{"task":"code"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate workload item id"):
        load_workload(path)


def test_load_workload_rejects_empty_segment_tag_values(tmp_path) -> None:
    path = tmp_path / "workload.jsonl"
    path.write_text(
        '{"id":"one","prompt":"hello","tags":{"risk":""}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty strings"):
        load_workload(path)
