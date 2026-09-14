from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from ...utils.time import utc_now

if TYPE_CHECKING:
    from .request import InferenceRequest


class CacheStrategy(StrEnum):
    EXACT = "exact"
    SEMANTIC = "semantic"
    PREFIX = "prefix"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class CacheKey:
    """Composite cache key."""

    content_hash: str
    model: str
    temperature: float
    max_tokens: int

    def to_string(self) -> str:
        return f"{self.content_hash}:{self.model}:{self.temperature}:{self.max_tokens}"

    @classmethod
    def from_request(cls, request: InferenceRequest) -> CacheKey:
        import hashlib
        import json

        content = request.prompt or json.dumps(request.messages, sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return cls(
            content_hash=content_hash,
            model=request.preferred_model or "default",
            temperature=request.parameters.temperature,
            max_tokens=request.parameters.max_tokens,
        )


@dataclass
class CacheEntry:
    """Entry in the response cache."""

    key: CacheKey
    prompt: str
    response: str
    embedding: object | None = None
    model_used: str = ""
    tokens_prompt: int = 0
    tokens_completion: int = 0
    cost_usd: float | None = None
    strategy: CacheStrategy = CacheStrategy.EXACT
    created_at: datetime = field(default_factory=utc_now)
    last_accessed: datetime = field(default_factory=utc_now)
    access_count: int = 0
    ttl_seconds: int | None = None
    confidence_score: float = 1.0
    id: UUID = field(default_factory=uuid4)

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        return (utc_now() - self.created_at).total_seconds()

    @property
    def cost_savings(self) -> float | None:
        if self.cost_usd is None:
            return None
        return self.cost_usd * self.access_count

    def touch(self) -> None:
        self.last_accessed = utc_now()
        self.access_count += 1


@dataclass(frozen=True)
class CacheMetrics:
    total_requests: int
    cache_hits: int
    cache_misses: int
    exact_hits: int = 0
    semantic_hits: int = 0
    prefix_hits: int = 0
    avg_similarity_score: float = 0.0
    total_tokens_saved: int = 0
    total_cost_saved_usd: float = 0.0
    avg_latency_saved_ms: float = 0.0
    cache_size: int = 0
    cache_memory_mb: float = 0.0
    evictions: int = 0
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


@dataclass(frozen=True)
class PrefixCacheEntry:
    prefix_hash: str
    prefix_text: str
    prefix_length: int
    kv_states: Any | None = None
    usage_count: int = 0
    last_used: datetime = field(default_factory=utc_now)
    tokens_saved_per_use: int = 0
    total_tokens_saved: int = 0


@dataclass(frozen=True)
class SemanticCacheConfig:
    enabled: bool = True
    similarity_threshold: float = 0.90
    max_distance: float = 0.15
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384
    index_type: str = "faiss"
    max_cache_size: int = 10000
