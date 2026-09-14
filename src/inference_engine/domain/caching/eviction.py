from abc import ABC, abstractmethod

import structlog

from ...utils.time import utc_now
from ..models.cache import CacheEntry

logger = structlog.get_logger()


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""

    @abstractmethod
    def should_evict(self, entry: CacheEntry) -> bool:
        """Check if entry should be evicted."""
        raise NotImplementedError

    @abstractmethod
    def select_to_evict(self, entries: list[CacheEntry]) -> CacheEntry:
        """Select entry to evict from candidates."""
        raise NotImplementedError


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""

    def should_evict(self, _entry: CacheEntry) -> bool:
        return False

    def select_to_evict(self, entries: list[CacheEntry]) -> CacheEntry:
        if not entries:
            raise ValueError("Cannot evict from empty list")

        lru = min(entries, key=lambda e: e.last_accessed)
        logger.debug("lru_eviction", entry_id=str(lru.id), last_accessed=lru.last_accessed)
        return lru


class LFUEvictionPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""

    def should_evict(self, _entry: CacheEntry) -> bool:
        return False

    def select_to_evict(self, entries: list[CacheEntry]) -> CacheEntry:
        if not entries:
            raise ValueError("Cannot evict from empty list")

        lfu = min(entries, key=lambda e: e.access_count)
        logger.debug("lfu_eviction", entry_id=str(lfu.id), access_count=lfu.access_count)
        return lfu


class TTL_EvictionPolicy(EvictionPolicy):
    """Time To Live eviction policy."""

    def should_evict(self, entry: CacheEntry) -> bool:
        if entry.ttl_seconds is None:
            return False

        age = (utc_now() - entry.created_at).total_seconds()
        return age > entry.ttl_seconds

    def select_to_evict(self, entries: list[CacheEntry]) -> CacheEntry:
        if not entries:
            raise ValueError("Cannot evict from empty list")

        expired = [e for e in entries if self.should_evict(e)]
        if expired:
            oldest_expired = max(expired, key=lambda e: e.created_at)
            logger.debug("ttl_eviction", entry_id=str(oldest_expired.id), age=oldest_expired.age_seconds)
            return oldest_expired

        oldest = max(entries, key=lambda e: e.created_at)
        logger.debug("ttl_eviction_oldest", entry_id=str(oldest.id))
        return oldest


class CostAwareEvictionPolicy(EvictionPolicy):
    """Evict by measured cost benefit, falling back to LRU when economics are unknown."""

    def should_evict(self, _entry: CacheEntry) -> bool:
        return False

    def select_to_evict(self, entries: list[CacheEntry]) -> CacheEntry:
        if not entries:
            raise ValueError("Cannot evict from empty list")

        if any(entry.cost_savings is None for entry in entries):
            fallback = min(entries, key=lambda entry: entry.last_accessed)
            logger.debug(
                "cost_aware_eviction_fallback_lru",
                entry_id=str(fallback.id),
                reason="unknown_cost_evidence",
            )
            return fallback

        def score(entry: CacheEntry) -> float:
            benefit = entry.cost_savings
            assert benefit is not None
            if entry.age_seconds == 0:
                return float("inf")
            return benefit / entry.age_seconds

        worst = min(entries, key=score)
        logger.debug("cost_aware_eviction", entry_id=str(worst.id), score=score(worst))
        return worst
