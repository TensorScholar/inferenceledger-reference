"""Deterministic IFStruct subset selection.

Selection is a filter-and-prefix of the published test set. It does not cherry-pick
items by model output or by expected difficulty after seeing results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

JSON_WRAPPER_SUBSET_ID = "ifstruct-v1.0-json-wrapper-n100"
JSON_WRAPPER_SUBSET_LIMIT = 100
SOURCE_OUTPUT_FORMAT = "json"


def select_json_wrapper_prefix(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = JSON_WRAPPER_SUBSET_LIMIT,
) -> list[dict[str, Any]]:
    """Return the first ``limit`` JSON + wrapper-object IFStruct rows, ordered by seed.

    Predicate, specified before execution:

    1. ``output_format == "json"``
    2. ``require_wrapper_key is True``
    3. stable sort by ``(seed, entity_type)``
    4. take the first ``limit`` rows

    YAML and bare-list JSON items are excluded because LiteLLM
    ``response_format={"type":"json_object"}`` cannot express YAML or a bare JSON
    array. This is a technical-applicability filter, not a quality filter.
    """
    if limit < 1:
        raise ValueError("subset limit must be positive")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("output_format") != SOURCE_OUTPUT_FORMAT:
            continue
        if row.get("require_wrapper_key") is not True:
            continue
        selected.append(dict(row))
    selected.sort(key=lambda item: (_require_int_seed(item), str(item.get("entity_type") or "")))
    if len(selected) < limit:
        raise ValueError(
            f"eligible JSON-wrapper IFStruct rows={len(selected)} is below requested limit={limit}"
        )
    return selected[:limit]


def workload_item_id(seed: int) -> str:
    return f"ifstruct-{seed}"


def _require_int_seed(row: Mapping[str, Any]) -> int:
    seed = row.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"IFStruct row seed must be an int, got {seed!r}")
    return seed
