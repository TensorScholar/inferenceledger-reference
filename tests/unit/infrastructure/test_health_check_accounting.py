from __future__ import annotations

from typing import Any

import pytest

from inference_engine.infrastructure.models import openai_backend
from inference_engine.infrastructure.models.openai_backend import OpenAIBackend


class _NoInferenceCompletions:
    async def create(self, **_kwargs: Any) -> Any:
        raise AssertionError("health_check must not issue a chat completion")


class _NoInferenceChat:
    def __init__(self) -> None:
        self.completions = _NoInferenceCompletions()


class _FakeModels:
    def __init__(self, *, fail: bool) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def retrieve(self, model: str) -> object:
        self.calls.append(model)
        if self.fail:
            raise RuntimeError("model metadata unavailable")
        return {"id": model}


class _HealthClient:
    instances: list[_HealthClient] = []
    fail = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        timeout: float,
        max_retries: int,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.chat = _NoInferenceChat()
        self.models = _FakeModels(fail=self.fail)
        self.instances.append(self)


@pytest.fixture(autouse=True)
def fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _HealthClient.instances = []
    _HealthClient.fail = False
    monkeypatch.setattr(openai_backend, "AsyncOpenAI", _HealthClient)


@pytest.mark.asyncio
async def test_health_check_uses_non_inference_metadata_with_sdk_retries_disabled() -> None:
    backend = OpenAIBackend(api_key="test", model_name="test-model")
    assert await backend.health_check() is True
    client = _HealthClient.instances[0]
    assert client.max_retries == 0
    assert client.models.calls == ["test-model"]


@pytest.mark.asyncio
async def test_health_check_fails_closed_instead_of_falling_back_to_completion() -> None:
    _HealthClient.fail = True
    backend = OpenAIBackend(api_key="test", model_name="test-model")
    assert await backend.health_check() is False
    assert _HealthClient.instances[0].models.calls == ["test-model"]
