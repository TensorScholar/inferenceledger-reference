from __future__ import annotations

import importlib.util
import os

import pytest

from inference_engine.domain.models.request import InferenceRequest, ModelParameters
from inference_engine.infrastructure.models.openai_backend import OpenAIBackend, RetryPolicy

_HAS_OPENAI = importlib.util.find_spec("openai") is not None


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not _HAS_OPENAI,
    reason="requires OPENAI_API_KEY and the openai provider dependency",
)
@pytest.mark.asyncio
async def test_real_openai_compatible_provider_returns_usage_metadata() -> None:
    model = os.getenv("OPENAI_TEST_MODEL", "gpt-5.4-mini")
    backend = OpenAIBackend(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=model,
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout_seconds=30.0,
        retry_policy=RetryPolicy(max_attempts=1),
    )

    response = await backend.infer(
        InferenceRequest(
            prompt="Reply with exactly: ok",
            parameters=ModelParameters(max_tokens=8, temperature=0.0),
        )
    )

    assert response.model_used == model
    assert response.text.strip()
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens >= (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )
    assert response.usage.cost_usd > 0
    assert response.latency_ms > 0
    assert response.provider_attempt_count == 1
    assert response.provider_retry_count == 0
