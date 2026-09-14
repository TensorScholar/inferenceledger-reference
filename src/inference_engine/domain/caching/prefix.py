from __future__ import annotations

from typing import Any

from ...utils.time import utc_now
from ..models.cache import PrefixCacheEntry
from ..models.request import InferenceRequest
from ..models.response import CacheInfo, InferenceResponse
from .base import AbstractCache


class PrefixCache(AbstractCache):
    """Prefix metadata cache for future prompt/KV-cache work."""

    def __init__(self, max_entries: int = 1000) -> None:
        self.max_entries = max_entries
        self.prefix_cache: dict[str, PrefixCacheEntry] = {}
        self.hits = 0
        self.misses = 0

    async def get_prefix(self, text: str) -> PrefixCacheEntry | None:
        for entry in self.prefix_cache.values():
            if text.startswith(entry.prefix_text):
                object.__setattr__(entry, "last_used", utc_now())
                object.__setattr__(entry, "usage_count", entry.usage_count + 1)
                self.hits += 1
                return entry
        self.misses += 1
        return None

    async def set_prefix(self, prefix_text: str, kv_states: Any | None = None) -> None:
        import hashlib

        prefix_hash = hashlib.sha256(prefix_text.encode()).hexdigest()[:16]
        self.prefix_cache[prefix_hash] = PrefixCacheEntry(
            prefix_hash=prefix_hash,
            prefix_text=prefix_text,
            prefix_length=len(prefix_text),
            kv_states=kv_states,
            tokens_saved_per_use=max(len(prefix_text) // 4, 1),
        )
        if len(self.prefix_cache) > self.max_entries:
            self._evict_lfu()

    def _evict_lfu(self) -> None:
        lfu_key = min(self.prefix_cache, key=lambda key: self.prefix_cache[key].usage_count)
        del self.prefix_cache[lfu_key]

    async def get(self, _request: InferenceRequest) -> tuple[InferenceResponse, CacheInfo] | None:
        return None

    async def set(self, _request: InferenceRequest, _response: InferenceResponse) -> None:
        return None

    async def invalidate(self, pattern: str | None = None) -> int:
        if pattern is None:
            count = len(self.prefix_cache)
            self.prefix_cache.clear()
            return count
        keys = [key for key, entry in self.prefix_cache.items() if pattern in entry.prefix_text]
        for key in keys:
            del self.prefix_cache[key]
        return len(keys)

    def get_metrics(self) -> dict[str, object]:
        total = self.hits + self.misses
        return {
            "cache_size": len(self.prefix_cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "total_tokens_saved": sum(entry.total_tokens_saved for entry in self.prefix_cache.values()),
        }
