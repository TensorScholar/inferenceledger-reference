from __future__ import annotations

from typing import Any

import pytest

from inference_engine.infrastructure.models import openai_backend
from inference_engine.infrastructure.models.openai_backend import OpenAIBackend


class FakeClient:
    def __init__(self, **_kwargs: Any) -> None:
        pass


def test_pricing_provider_cannot_differ_from_execution_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_backend, "AsyncOpenAI", FakeClient)

    with pytest.raises(ValueError, match="must match provider_name"):
        OpenAIBackend(
            api_key="test-key",
            model_name="test-model",
            base_url="https://provider.example/v1",
            provider_name="azure",
            pricing_provider="openai",
        )


def test_explicit_non_openai_identity_does_not_inherit_openai_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_backend, "AsyncOpenAI", FakeClient)

    backend = OpenAIBackend(
        api_key="test-key",
        model_name="test-model",
        provider_name="internal-proxy",
    )

    assert backend.provider_name == "internal-proxy"
    assert backend.pricing_provider is None
