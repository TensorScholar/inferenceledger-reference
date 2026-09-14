from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...domain.models.execution import ProviderAttempt


class ProviderErrorType(StrEnum):
    """Stable provider failure categories used by retries and telemetry."""

    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    MISSING_USAGE = "missing_usage"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderError(Exception):
    """Provider exception with a normalized failure type and observed attempt chain."""

    error_type: ProviderErrorType
    message: str
    provider: str
    retryable: bool
    status_code: int | None = None
    cause: Exception | None = None
    provider_attempt_count: int = 1
    provider_retry_count: int = 0
    provider_attempts: tuple[ProviderAttempt, ...] = ()

    def __post_init__(self) -> None:
        if self.provider_attempts:
            object.__setattr__(self, "provider_attempt_count", len(self.provider_attempts))
            object.__setattr__(self, "provider_retry_count", max(len(self.provider_attempts) - 1, 0))

    def __str__(self) -> str:
        return self.message


def classify_openai_error(error: Exception) -> ProviderError:
    """Map OpenAI SDK exceptions into stable provider error categories."""

    status_code = _extract_status_code(error)
    error_name = type(error).__name__.lower()
    message = str(error)

    if "timeout" in error_name or "timeout" in message.lower():
        return ProviderError(
            ProviderErrorType.TIMEOUT,
            message,
            provider="openai-compatible",
            retryable=True,
            status_code=status_code,
            cause=error,
        )
    if status_code in {401, 403} or "authentication" in error_name:
        return ProviderError(
            ProviderErrorType.AUTHENTICATION,
            message,
            provider="openai-compatible",
            retryable=False,
            status_code=status_code,
            cause=error,
        )
    if status_code == 429 or "ratelimit" in error_name or "rate_limit" in error_name:
        return ProviderError(
            ProviderErrorType.RATE_LIMIT,
            message,
            provider="openai-compatible",
            retryable=True,
            status_code=status_code,
            cause=error,
        )
    if status_code is not None and 400 <= status_code < 500:
        return ProviderError(
            ProviderErrorType.INVALID_REQUEST,
            message,
            provider="openai-compatible",
            retryable=False,
            status_code=status_code,
            cause=error,
        )
    if status_code is not None and status_code >= 500:
        return ProviderError(
            ProviderErrorType.SERVER_ERROR,
            message,
            provider="openai-compatible",
            retryable=True,
            status_code=status_code,
            cause=error,
        )
    return ProviderError(
        ProviderErrorType.UNKNOWN,
        message,
        provider="openai-compatible",
        retryable=False,
        status_code=status_code,
        cause=error,
    )


def missing_usage_error(model: str) -> ProviderError:
    return ProviderError(
        ProviderErrorType.MISSING_USAGE,
        f"Provider response for model '{model}' did not include token usage metadata",
        provider="openai-compatible",
        retryable=False,
    )


def _extract_status_code(error: Exception) -> int | None:
    raw = getattr(error, "status_code", None)
    if isinstance(raw, int):
        return raw

    response = getattr(error, "response", None)
    if response is None:
        return None

    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status

    if isinstance(response, dict):
        status = response.get("status_code")
        if isinstance(status, int):
            return status

    return None
