from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from ...utils.time import utc_now
from .execution import ProviderAttempt


@dataclass(frozen=True)
class CacheInfo:
    """Information about cache use for one response."""

    hit: bool
    source: str | None = None
    similarity_score: float | None = None
    tokens_saved: int = 0
    latency_saved_ms: int = 0


@dataclass(frozen=True)
class UsageMetrics:
    """Token usage and optional calculated cost for the final successful provider response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        if min(self.prompt_tokens, self.completion_tokens, self.total_tokens, self.cached_tokens) < 0:
            raise ValueError("usage token counts must be non-negative")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("cost_usd must be non-negative when known")

    @property
    def cache_hit_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.cached_tokens / self.total_tokens


@dataclass(frozen=True)
class InferenceResponse:
    """Canonical inference response plus the provider attempts that produced it."""

    request_id: UUID
    text: str
    model_used: str
    usage: UsageMetrics
    cache_info: CacheInfo
    latency_ms: int
    finish_reason: str = "stop"
    queue_time_ms: int = 0
    inference_time_ms: int = 0
    postprocess_time_ms: int = 0
    provider_attempt_count: int = 1
    provider_retry_count: int = 0
    provider_attempts: tuple[ProviderAttempt, ...] = ()
    timestamp: datetime = field(default_factory=utc_now)
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.provider_attempts:
            object.__setattr__(self, "provider_attempt_count", len(self.provider_attempts))
            object.__setattr__(self, "provider_retry_count", max(len(self.provider_attempts) - 1, 0))

    @property
    def total_cost_usd(self) -> float | None:
        """Calculated cost of the final successful response only, when pricing is known.

        Full execution-chain cost belongs to execution evidence because preceding retry/fallback
        attempts may have unknown or additional cost.
        """
        return self.usage.cost_usd
