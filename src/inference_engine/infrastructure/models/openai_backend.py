import importlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, cast

import structlog

try:
    _openai_module = importlib.import_module("openai")
    AsyncOpenAI: Any = _openai_module.AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from ...domain.cost.calculator import CostCalculator
from ...domain.cost.pricing import PricingQuote, UnknownModelPricingError
from ...domain.models.execution import AttemptOutcome, CostEvidenceKind, ProviderAttempt
from ...domain.models.request import InferenceRequest
from ...domain.models.response import CacheInfo, InferenceResponse, UsageMetrics
from .base import AbstractModelBackend
from .errors import ProviderError, classify_openai_error, missing_usage_error

logger = structlog.get_logger()

_PROVIDER_PROTOCOL = "openai-compatible"
_UNKNOWN_COMPATIBLE_PROVIDER = "openai-compatible-unknown"


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry policy for provider calls."""

    max_attempts: int = 2
    backoff_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")


@dataclass(frozen=True)
class ProviderCallResult:
    """Provider response plus every local provider invocation used to obtain it."""

    response: Any
    attempts: tuple[ProviderAttempt, ...]

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


class OpenAIBackend(AbstractModelBackend):
    """OpenAI-compatible chat completions backend.

    Transport protocol, execution-provider identity, and billing identity are deliberately separate.
    Direct OpenAI calls safely default both identities to ``openai``. A custom ``base_url`` is
    unpriced by default and records an unknown compatible provider unless the caller supplies an
    explicit provider identity. Until a dedicated billing-mapping abstraction exists, a calculated
    price is allowed only when execution-provider and pricing-provider identities are identical.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        *,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        cost_calculator: CostCalculator | None = None,
        provider_name: str | None = None,
        pricing_provider: str | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise ImportError("openai package not installed")
        if provider_name is not None and not provider_name.strip():
            raise ValueError("provider_name must be non-empty when supplied")
        if pricing_provider is not None and not pricing_provider.strip():
            raise ValueError("pricing_provider must be non-empty when supplied")
        if base_url is not None and pricing_provider is not None and provider_name is None:
            raise ValueError(
                "custom OpenAI-compatible pricing requires an explicit provider_name as well as pricing_provider"
            )

        resolved_provider = provider_name or (
            "openai" if base_url is None else _UNKNOWN_COMPATIBLE_PROVIDER
        )
        if pricing_provider is not None:
            resolved_pricing_provider = pricing_provider
        elif base_url is None and resolved_provider == "openai":
            resolved_pricing_provider = "openai"
        else:
            resolved_pricing_provider = None

        if (
            resolved_pricing_provider is not None
            and resolved_pricing_provider != resolved_provider
        ):
            raise ValueError(
                "pricing_provider must match provider_name until an explicit billing mapping is modeled"
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self._model_name = model_name
        self.retry_policy = retry_policy or RetryPolicy()
        self.cost_calculator = cost_calculator or CostCalculator()
        self.provider_protocol = _PROVIDER_PROTOCOL
        self.provider_name = resolved_provider
        self.pricing_provider = resolved_pricing_provider

    @property
    def model_name(self) -> str:
        return self._model_name

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference and preserve observed usage even when price evidence is unavailable."""
        start = perf_counter()

        messages = request.messages or [{"role": "user", "content": request.prompt}]
        call_result = await self._create_completion(messages, request)
        response = call_result.response

        elapsed_ms = int((perf_counter() - start) * 1000)
        completion = response.choices[0].message.content or ""
        usage = response.usage
        if usage is None:
            attempts = _mark_last_attempt_error(
                call_result.attempts,
                error_type="missing_usage",
            )
            raise replace(
                missing_usage_error(self.model_name),
                provider=self.provider_name,
                provider_attempts=attempts,
            )

        cached_tokens = _extract_cached_tokens(usage)
        quote = self._quote_usage_cost(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cached_input_tokens=cached_tokens,
        )
        final_attempt = _attempt_with_usage(
            call_result.attempts[-1],
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=cached_tokens,
            quote=quote,
        )
        attempts = (*call_result.attempts[:-1], final_attempt)

        return InferenceResponse(
            request_id=request.id,
            text=completion,
            finish_reason=response.choices[0].finish_reason or "stop",
            model_used=self.model_name,
            usage=UsageMetrics(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cached_tokens=cached_tokens,
                cost_usd=quote.amount_usd if quote is not None else None,
            ),
            cache_info=CacheInfo(hit=False),
            latency_ms=elapsed_ms,
            inference_time_ms=elapsed_ms,
            provider_attempts=attempts,
        )

    def _quote_usage_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
    ) -> PricingQuote | None:
        if self.pricing_provider is None:
            return None
        try:
            return self.cost_calculator.quote_provider_usage(
                provider=self.pricing_provider,
                model_name=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            )
        except UnknownModelPricingError:
            logger.warning(
                "provider_usage_unpriced",
                provider=self.provider_name,
                pricing_provider=self.pricing_provider,
                model=self.model_name,
                pricing_table_version=self.cost_calculator.pricing_table.version,
            )
            return None

    async def infer_batch(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        """Process batch sequentially (OpenAI-compatible chat path has no native batch here)."""
        results = []
        for req in requests:
            results.append(await self.infer(req))
        return results

    async def stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream tokens.

        Streaming usage/cost evidence is not yet exposed by this reference interface, so streaming
        must not be used to substantiate migration-cost claims.
        """
        messages = request.messages or [{"role": "user", "content": request.prompt}]

        call_result = await self._create_completion(messages, request, stream=True)
        stream = call_result.response
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def health_check(self) -> bool:
        """Check model availability without issuing an inference request.

        The client is configured with ``max_retries=0``. Health checking therefore uses model
        metadata retrieval and cannot silently create a billable Chat Completions attempt that is
        absent from the economic ledger. OpenAI-compatible endpoints that do not implement this
        capability fail the health check rather than falling back to an untracked inference probe.
        """
        try:
            await self.client.models.retrieve(self.model_name)
            return True
        except Exception as exc:
            logger.error(
                "openai_compatible_health_check_failed",
                provider=self.provider_name,
                error=str(exc),
            )
            return False

    async def _create_completion(
        self,
        messages: list[dict[str, str]],
        request: InferenceRequest,
        *,
        stream: bool = False,
    ) -> ProviderCallResult:
        import asyncio

        attempts: list[ProviderAttempt] = []
        last_error: ProviderError | None = None
        for attempt_index in range(1, self.retry_policy.max_attempts + 1):
            attempt_start = perf_counter()
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=cast(Any, messages),
                    max_tokens=request.parameters.max_tokens,
                    temperature=request.parameters.temperature,
                    top_p=request.parameters.top_p,
                    frequency_penalty=request.parameters.frequency_penalty,
                    presence_penalty=request.parameters.presence_penalty,
                    stop=request.parameters.stop_sequences or None,
                    stream=stream,
                )
                attempts.append(
                    ProviderAttempt(
                        attempt_index=attempt_index,
                        provider=self.provider_name,
                        model=self.model_name,
                        outcome=AttemptOutcome.SUCCEEDED,
                        latency_ms=int((perf_counter() - attempt_start) * 1000),
                    )
                )
                return ProviderCallResult(response=response, attempts=tuple(attempts))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                provider_error = replace(
                    classify_openai_error(exc),
                    provider=self.provider_name,
                )
                attempts.append(
                    ProviderAttempt(
                        attempt_index=attempt_index,
                        provider=self.provider_name,
                        model=self.model_name,
                        outcome=AttemptOutcome.FAILED,
                        latency_ms=int((perf_counter() - attempt_start) * 1000),
                        error_type=provider_error.error_type.value,
                        status_code=provider_error.status_code,
                    )
                )
                provider_error = replace(
                    provider_error,
                    provider_attempts=tuple(attempts),
                )
                last_error = provider_error
                if not provider_error.retryable or attempt_index >= self.retry_policy.max_attempts:
                    raise provider_error from exc
                await asyncio.sleep(self.retry_policy.backoff_seconds * attempt_index)

        if last_error is not None:
            raise last_error
        raise ProviderError(
            error_type=classify_openai_error(RuntimeError("provider call failed")).error_type,
            message="Provider call failed without a captured exception",
            provider=self.provider_name,
            retryable=False,
            provider_attempts=tuple(attempts),
        )


def _attempt_with_usage(
    attempt: ProviderAttempt,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cached_tokens: int,
    quote: PricingQuote | None,
) -> ProviderAttempt:
    if quote is None:
        return replace(
            attempt,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
        )
    if quote.provider != attempt.provider:
        raise ValueError("pricing quote provider must match observed execution provider")
    return replace(
        attempt,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        calculated_cost_usd=quote.amount_usd,
        cost_evidence=CostEvidenceKind.CALCULATED_FROM_USAGE,
        pricing_table_version=quote.pricing_table_version,
        pricing_record_id=quote.pricing_record_id,
        pricing_observed_at=quote.pricing_observed_at.isoformat(),
        pricing_source_url=quote.pricing_source_url,
    )


def _mark_last_attempt_error(
    attempts: tuple[ProviderAttempt, ...],
    *,
    error_type: str,
) -> tuple[ProviderAttempt, ...]:
    if not attempts:
        return attempts
    return (*attempts[:-1], replace(attempts[-1], error_type=error_type))


def _extract_cached_tokens(usage: Any) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    cached_tokens = getattr(details, "cached_tokens", 0)
    if isinstance(cached_tokens, int):
        return cached_tokens
    return 0
