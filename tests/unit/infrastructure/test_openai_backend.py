from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from inference_engine.domain.cost.calculator import CostCalculator
from inference_engine.domain.cost.pricing import ModelPricing, PricingTable
from inference_engine.domain.models.execution import AttemptOutcome, CostEvidenceKind
from inference_engine.domain.models.request import InferenceRequest, ModelParameters
from inference_engine.infrastructure.models import openai_backend
from inference_engine.infrastructure.models.errors import ProviderError, ProviderErrorType
from inference_engine.infrastructure.models.openai_backend import OpenAIBackend, RetryPolicy

_DEFAULT_USAGE = object()


@dataclass
class FakePromptTokensDetails:
    cached_tokens: int = 0


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: FakePromptTokensDetails | None = None


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage | None


class FakeOpenAIClient:
    instances: list[FakeOpenAIClient] = []
    queued_results: list[Any] = []

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
        self.calls: list[dict[str, Any]] = []
        self.chat = _FakeChat(self)
        self.instances.append(self)


class _FakeChat:
    def __init__(self, client: FakeOpenAIClient) -> None:
        self.completions = _FakeCompletions(client)


class _FakeCompletions:
    def __init__(self, client: FakeOpenAIClient) -> None:
        self.client = client

    async def create(self, **kwargs: Any) -> Any:
        self.client.calls.append(kwargs)
        result = FakeOpenAIClient.queued_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


@pytest.fixture(autouse=True)
def fake_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOpenAIClient.instances = []
    FakeOpenAIClient.queued_results = []
    monkeypatch.setattr(openai_backend, "AsyncOpenAI", FakeOpenAIClient)


def _calculator() -> CostCalculator:
    pricing = ModelPricing(
        provider="test-provider",
        model="test-model",
        input_per_million=1.00,
        output_per_million=2.00,
        cached_input_per_million=0.25,
        source_url="https://pricing.example/test-model",
    )
    return CostCalculator(
        PricingTable(
            prices={(pricing.provider, pricing.model): pricing},
            version="test",
        )
    )


def _priced_backend(*, max_attempts: int = 1, backoff_seconds: float = 0) -> OpenAIBackend:
    return OpenAIBackend(
        api_key="test-key",
        model_name="test-model",
        base_url="https://provider.example/v1",
        provider_name="test-provider",
        pricing_provider="test-provider",
        retry_policy=RetryPolicy(max_attempts=max_attempts, backoff_seconds=backoff_seconds),
        cost_calculator=_calculator(),
    )


def _response(usage: FakeUsage | None | object = _DEFAULT_USAGE) -> FakeResponse:
    response_usage = (
        FakeUsage(
            prompt_tokens=1_000,
            completion_tokens=500,
            total_tokens=1_500,
        )
        if usage is _DEFAULT_USAGE
        else usage
    )
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(content="hello"))],
        usage=response_usage if isinstance(response_usage, FakeUsage) else None,
    )


@pytest.mark.asyncio
async def test_openai_backend_sends_real_chat_completion_shape_with_pricing_provenance() -> None:
    FakeOpenAIClient.queued_results = [_response()]
    backend = OpenAIBackend(
        api_key="test-key",
        model_name="test-model",
        base_url="https://provider.example/v1",
        provider_name="test-provider",
        pricing_provider="test-provider",
        timeout_seconds=7.5,
        retry_policy=RetryPolicy(max_attempts=1),
        cost_calculator=_calculator(),
    )

    response = await backend.infer(
        InferenceRequest(
            prompt="Hello",
            parameters=ModelParameters(max_tokens=32, temperature=0.2, top_p=0.8),
        )
    )

    client = FakeOpenAIClient.instances[0]
    assert client.api_key == "test-key"
    assert client.base_url == "https://provider.example/v1"
    assert client.timeout == 7.5
    assert client.max_retries == 0
    assert client.calls[0]["model"] == "test-model"
    assert client.calls[0]["messages"] == [{"role": "user", "content": "Hello"}]
    assert client.calls[0]["max_tokens"] == 32
    assert response.text == "hello"
    assert response.usage.cost_usd == pytest.approx(0.002)
    assert response.provider_attempt_count == 1
    assert response.provider_retry_count == 0
    attempt = response.provider_attempts[0]
    assert attempt.provider == "test-provider"
    assert attempt.attempt_index == 1
    assert attempt.outcome == AttemptOutcome.SUCCEEDED
    assert attempt.total_tokens == 1_500
    assert attempt.calculated_cost_usd == pytest.approx(0.002)
    assert attempt.cost_evidence == CostEvidenceKind.CALCULATED_FROM_USAGE
    assert attempt.pricing_table_version == "test"
    assert attempt.pricing_record_id is not None
    assert attempt.pricing_observed_at is not None
    assert attempt.pricing_source_url == "https://pricing.example/test-model"


@pytest.mark.asyncio
async def test_custom_compatible_endpoint_is_unpriced_without_explicit_billing_identity() -> None:
    FakeOpenAIClient.queued_results = [_response()]
    backend = OpenAIBackend(
        api_key="test-key",
        model_name="test-model",
        base_url="https://provider.example/v1",
        retry_policy=RetryPolicy(max_attempts=1),
        cost_calculator=_calculator(),
    )

    response = await backend.infer(InferenceRequest(prompt="Hello"))

    assert response.text == "hello"
    assert response.usage.total_tokens == 1_500
    assert response.usage.cost_usd is None
    attempt = response.provider_attempts[0]
    assert attempt.provider == "openai-compatible-unknown"
    assert attempt.cost_evidence == CostEvidenceKind.UNKNOWN
    assert attempt.calculated_cost_usd is None
    assert attempt.pricing_table_version is None


def test_custom_pricing_identity_requires_explicit_execution_provider() -> None:
    with pytest.raises(ValueError, match="explicit provider_name"):
        OpenAIBackend(
            api_key="test-key",
            model_name="test-model",
            base_url="https://provider.example/v1",
            pricing_provider="test-provider",
            cost_calculator=_calculator(),
        )


@pytest.mark.asyncio
async def test_direct_openai_defaults_execution_provider_identity_to_openai() -> None:
    FakeOpenAIClient.queued_results = [_response()]
    backend = OpenAIBackend(
        api_key="test-key",
        model_name="test-model",
        retry_policy=RetryPolicy(max_attempts=1),
        cost_calculator=_calculator(),
    )

    response = await backend.infer(InferenceRequest(prompt="Hello"))

    assert response.provider_attempts[0].provider == "openai"
    # The test table contains no OpenAI/test-model record, so execution remains observed but unpriced.
    assert response.usage.cost_usd is None
    assert response.provider_attempts[0].cost_evidence == CostEvidenceKind.UNKNOWN


@pytest.mark.asyncio
async def test_openai_backend_raises_when_usage_missing() -> None:
    FakeOpenAIClient.queued_results = [_response(usage=None)]
    backend = _priced_backend()

    with pytest.raises(ProviderError) as exc_info:
        await backend.infer(InferenceRequest(prompt="Hello"))

    error = exc_info.value
    assert error.error_type == ProviderErrorType.MISSING_USAGE
    assert error.provider == "test-provider"
    assert error.retryable is False
    assert error.provider_attempt_count == 1
    assert error.provider_retry_count == 0
    assert error.provider_attempts[0].provider == "test-provider"
    assert error.provider_attempts[0].outcome == AttemptOutcome.SUCCEEDED
    assert error.provider_attempts[0].error_type == "missing_usage"
    assert error.provider_attempts[0].calculated_cost_usd is None
    assert error.provider_attempts[0].cost_evidence == CostEvidenceKind.UNKNOWN


@pytest.mark.asyncio
async def test_openai_backend_retries_retryable_errors() -> None:
    FakeOpenAIClient.queued_results = [FakeStatusError(429), _response()]
    backend = _priced_backend(max_attempts=2)

    response = await backend.infer(InferenceRequest(prompt="Hello"))

    assert response.text == "hello"
    assert len(FakeOpenAIClient.instances[0].calls) == 2
    assert response.provider_attempt_count == 2
    assert response.provider_retry_count == 1
    first, second = response.provider_attempts
    assert first.provider == "test-provider"
    assert first.outcome == AttemptOutcome.FAILED
    assert first.error_type == ProviderErrorType.RATE_LIMIT.value
    assert first.status_code == 429
    assert first.calculated_cost_usd is None
    assert first.cost_evidence == CostEvidenceKind.UNKNOWN
    assert second.provider == "test-provider"
    assert second.outcome == AttemptOutcome.SUCCEEDED
    assert second.calculated_cost_usd == pytest.approx(0.002)
    assert second.cost_evidence == CostEvidenceKind.CALCULATED_FROM_USAGE


@pytest.mark.asyncio
async def test_openai_backend_does_not_retry_invalid_request() -> None:
    FakeOpenAIClient.queued_results = [FakeStatusError(400)]
    backend = _priced_backend(max_attempts=3)

    with pytest.raises(ProviderError) as exc_info:
        await backend.infer(InferenceRequest(prompt="Hello"))

    error = exc_info.value
    assert error.error_type == ProviderErrorType.INVALID_REQUEST
    assert error.provider == "test-provider"
    assert len(FakeOpenAIClient.instances[0].calls) == 1
    assert error.provider_attempt_count == 1
    assert error.provider_retry_count == 0
    assert error.provider_attempts[0].provider == "test-provider"
    assert error.provider_attempts[0].outcome == AttemptOutcome.FAILED
    assert error.provider_attempts[0].calculated_cost_usd is None


@pytest.mark.asyncio
async def test_openai_backend_records_exhausted_retry_attempts() -> None:
    FakeOpenAIClient.queued_results = [FakeStatusError(429), FakeStatusError(429)]
    backend = _priced_backend(max_attempts=2)

    with pytest.raises(ProviderError) as exc_info:
        await backend.infer(InferenceRequest(prompt="Hello"))

    error = exc_info.value
    assert error.error_type == ProviderErrorType.RATE_LIMIT
    assert error.provider == "test-provider"
    assert len(FakeOpenAIClient.instances[0].calls) == 2
    assert error.provider_attempt_count == 2
    assert error.provider_retry_count == 1
    assert all(attempt.provider == "test-provider" for attempt in error.provider_attempts)
    assert all(attempt.outcome == AttemptOutcome.FAILED for attempt in error.provider_attempts)
    assert all(attempt.calculated_cost_usd is None for attempt in error.provider_attempts)
