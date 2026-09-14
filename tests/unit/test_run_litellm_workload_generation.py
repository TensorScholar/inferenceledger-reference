from __future__ import annotations

import pytest

from scripts.run_litellm_workload import (
    _completion_kwargs,
    _generation_settings,
    _quality_limitation,
)


def test_generation_settings_require_positive_max_tokens() -> None:
    settings = _generation_settings({"generation": {"temperature": 0, "max_tokens": 1024, "stream": False}})

    assert settings == {"temperature": 0.0, "max_tokens": 1024, "stream": False}


def test_generation_settings_reject_missing_block() -> None:
    with pytest.raises(ValueError, match="experiment.generation is required"):
        _generation_settings({})


def test_candidate_kwargs_require_declared_response_format() -> None:
    kwargs = _completion_kwargs(
        model="ollama/gemma3:270m",
        api_base="http://127.0.0.1:11434",
        prompt="hello",
        generation={"temperature": 0.0, "max_tokens": 1024, "stream": False},
        arm="candidate",
        arm_spec={"response_format": {"type": "json_object"}},
    )

    assert kwargs["max_tokens"] == 1024
    assert kwargs["response_format"] == {"type": "json_object"}


def test_baseline_kwargs_omit_response_format() -> None:
    kwargs = _completion_kwargs(
        model="ollama/gemma3:270m",
        api_base="http://127.0.0.1:11434",
        prompt="hello",
        generation={"temperature": 0.0, "max_tokens": 1024, "stream": False},
        arm="baseline",
        arm_spec={},
    )

    assert "response_format" not in kwargs


def test_quality_limitation_uses_evaluator_measures() -> None:
    text = _quality_limitation(
        {"quality_evaluator": {"measures": "IFStruct JSON structure checks, not semantic quality"}}
    )

    assert "IFStruct JSON structure checks" in text
