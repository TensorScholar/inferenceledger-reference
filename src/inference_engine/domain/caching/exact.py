from __future__ import annotations

import structlog

from ..models.cache import CacheEntry, CacheKey, CacheStrategy
from ..models.request import InferenceRequest
from ..models.response import CacheInfo, InferenceResponse, UsageMetrics
from .base import AbstractCache

logger = structlog.get_logger()


class ExactCache(AbstractCache):
    """In-memory exact-match cache used for deterministic Phase 0 behavior."""

    def __init__(self, max_entries: int = 10000) -> None:
        self.max_entries = max_entries
        self.cache: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    async def get(self, request: InferenceRequest) -> tuple[InferenceResponse, CacheInfo] | None:
        entry = self.cache.get(request.cache_key)
        if entry is None or entry.is_expired:
            self.misses += 1
            return None
        entry.touch()
        self.hits += 1
        cache_info = CacheInfo(
            hit=True,
            source="exact",
            similarity_score=1.0,
            tokens_saved=entry.tokens_prompt + entry.tokens_completion,
            latency_saved_ms=500,
        )
        response = InferenceResponse(
            request_id=request.id,
            text=entry.response,
            model_used=entry.model_used,
            usage=UsageMetrics(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cached_tokens=0,
                cost_usd=0.0,
            ),
            cache_info=cache_info,
            latency_ms=1,
            provider_attempt_count=0,
            provider_retry_count=0,
        )
        return response, cache_info

    async def set(self, request: InferenceRequest, response: InferenceResponse) -> None:
        entry = CacheEntry(
            key=CacheKey.from_request(request),
            prompt=request.input_text,
            response=response.text,
            model_used=response.model_used,
            tokens_prompt=response.usage.prompt_tokens,
            tokens_completion=response.usage.completion_tokens,
            cost_usd=response.usage.cost_usd,
            strategy=CacheStrategy.EXACT,
            ttl_seconds=request.cache_ttl_seconds,
        )
        self.cache[request.cache_key] = entry
        if len(self.cache) > self.max_entries:
            self._evict_lru()

    def _evict_lru(self) -> None:
        if not self.cache:
            return
        lru_key = min(self.cache, key=lambda key: self.cache[key].last_accessed)
        del self.cache[lru_key]

    async def invalidate(self, pattern: str | None = None) -> int:
        if pattern is None:
            count = len(self.cache)
            self.cache.clear()
            return count
        keys = [
            key
            for key, entry in self.cache.items()
            if pattern in entry.prompt or pattern in entry.response
        ]
        for key in keys:
            del self.cache[key]
        return len(keys)

    def get_metrics(self) -> dict[str, object]:
        total = self.hits + self.misses
        return {
            "cache_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }
