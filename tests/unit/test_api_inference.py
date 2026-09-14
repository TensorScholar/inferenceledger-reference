from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException, status

from inference_engine.adapters.api.app import app
from inference_engine.adapters.api.routes import inference as inference_route
from inference_engine.adapters.api.routes.inference import InferenceRequestBody, create_inference
from inference_engine.domain.models.request import InferenceRequest
from inference_engine.domain.models.response import CacheInfo, InferenceResponse, UsageMetrics
from inference_engine.infrastructure.models.errors import ProviderError, ProviderErrorType


@pytest.mark.asyncio
async def test_api_inference_requires_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        await create_inference(InferenceRequestBody(prompt="hello"))

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "OPENAI_API_KEY" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_api_inference_uses_provider_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeBackend:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def infer(self, request: InferenceRequest) -> InferenceResponse:
            assert request.prompt == "hello"
            assert request.parameters.max_tokens == 16
            return InferenceResponse(
                request_id=request.id,
                text="ok",
                model_used=self.kwargs["model_name"],
                finish_reason="stop",
                usage=UsageMetrics(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cached_tokens=2,
                    cost_usd=0.001,
                ),
                cache_info=CacheInfo(hit=False),
                latency_ms=123,
                provider_attempt_count=2,
                provider_retry_count=1,
            )

    monkeypatch.setattr(inference_route, "OpenAIBackend", FakeBackend)

    response = await create_inference(
        InferenceRequestBody(prompt="hello", model="test-model", max_tokens=16)
    )

    assert response.text == "ok"
    assert response.model_used == "test-model"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert response.cached_tokens == 2
    assert response.final_attempt_cost_usd == 0.001
    assert response.provider_attempt_count == 2
    assert response.provider_retry_count == 1


@pytest.mark.asyncio
async def test_api_inference_preserves_unknown_final_attempt_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class UnpricedBackend:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def infer(self, request: InferenceRequest) -> InferenceResponse:
            return InferenceResponse(
                request_id=request.id,
                text="ok",
                model_used="test-model",
                usage=UsageMetrics(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cost_usd=None,
                ),
                cache_info=CacheInfo(hit=False),
                latency_ms=123,
            )

    monkeypatch.setattr(inference_route, "OpenAIBackend", UnpricedBackend)

    response = await create_inference(InferenceRequestBody(prompt="hello", model="test-model"))

    assert response.final_attempt_cost_usd is None
    assert response.total_tokens == 15


@pytest.mark.asyncio
async def test_api_inference_maps_provider_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FailingBackend:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def infer(self, _request: InferenceRequest) -> InferenceResponse:
            raise ProviderError(
                ProviderErrorType.RATE_LIMIT,
                "rate limited",
                provider="test-provider",
                retryable=True,
                status_code=429,
                provider_attempt_count=2,
                provider_retry_count=1,
            )

    monkeypatch.setattr(inference_route, "OpenAIBackend", FailingBackend)

    with pytest.raises(HTTPException) as exc_info:
        await create_inference(InferenceRequestBody(prompt="hello"))

    assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert exc_info.value.detail["type"] == "rate_limit"
    assert exc_info.value.detail["provider_attempt_count"] == 2
    assert exc_info.value.detail["provider_retry_count"] == 1


def test_api_app_is_narrow_reference_executor() -> None:
    schema = app.openapi()
    paths = set(schema["paths"])

    assert schema["info"]["title"] == "InferenceLedger Reference Executor"
    assert "/v1/inference" in paths
    assert "/v1/models" not in paths
    assert "/v1/metrics/summary" not in paths
    assert "/v1/metrics/cache" not in paths
    assert "/v1/metrics/cost" not in paths
    assert "/v1/cache/stats" not in paths
    assert "/v1/cache" not in paths
