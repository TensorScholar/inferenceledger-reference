from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ....domain.models.request import InferenceRequest, ModelParameters
from ....infrastructure.models.errors import ProviderError, ProviderErrorType
from ....infrastructure.models.openai_backend import OpenAIBackend, RetryPolicy

router = APIRouter()


class InferenceRequestBody(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = "gpt-4o-mini"
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class InferenceResponseBody(BaseModel):
    text: str
    model_used: str
    finish_reason: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    final_attempt_cost_usd: float | None
    provider_attempt_count: int
    provider_retry_count: int


@router.post("/inference", response_model=InferenceResponseBody)
async def create_inference(payload: InferenceRequestBody) -> InferenceResponseBody:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is required for real provider inference.",
        )

    try:
        backend = OpenAIBackend(
            api_key=api_key,
            model_name=payload.model,
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout_seconds=payload.timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=2),
        )
        response = await backend.infer(
            InferenceRequest(
                prompt=payload.prompt,
                parameters=ModelParameters(
                    max_tokens=payload.max_tokens,
                    temperature=payload.temperature,
                    top_p=payload.top_p,
                ),
            )
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI provider dependency is not installed.",
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=_provider_error_status(exc),
            detail={
                "type": exc.error_type.value,
                "message": exc.message,
                "retryable": exc.retryable,
                "provider_attempt_count": exc.provider_attempt_count,
                "provider_retry_count": exc.provider_retry_count,
            },
        ) from exc

    return InferenceResponseBody(
        text=response.text,
        model_used=response.model_used,
        finish_reason=response.finish_reason,
        latency_ms=response.latency_ms,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
        cached_tokens=response.usage.cached_tokens,
        final_attempt_cost_usd=response.usage.cost_usd,
        provider_attempt_count=response.provider_attempt_count,
        provider_retry_count=response.provider_retry_count,
    )


def _provider_error_status(error: ProviderError) -> int:
    return {
        ProviderErrorType.AUTHENTICATION: status.HTTP_502_BAD_GATEWAY,
        ProviderErrorType.INVALID_REQUEST: status.HTTP_400_BAD_REQUEST,
        ProviderErrorType.MISSING_USAGE: status.HTTP_502_BAD_GATEWAY,
        ProviderErrorType.RATE_LIMIT: status.HTTP_429_TOO_MANY_REQUESTS,
        ProviderErrorType.SERVER_ERROR: status.HTTP_502_BAD_GATEWAY,
        ProviderErrorType.TIMEOUT: status.HTTP_504_GATEWAY_TIMEOUT,
        ProviderErrorType.UNKNOWN: status.HTTP_502_BAD_GATEWAY,
    }[error.error_type]
