
import numpy as np
import structlog

logger = structlog.get_logger()


class EmbeddingCache:
    """
    In-memory cache for computed embeddings.

    Avoids recomputing embeddings for identical text.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self.cache: dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, text: str) -> np.ndarray | None:
        """Get cached embedding if available."""
        embedding = self.cache.get(text)

        if embedding is not None:
            self.hits += 1
            logger.debug("embedding_cache_hit", text_hash=hash(text) % 10000)
            return embedding

        self.misses += 1
        logger.debug("embedding_cache_miss", text_hash=hash(text) % 10000)
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        """Cache embedding."""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()

        self.cache[text] = embedding

    def _evict_oldest(self) -> None:
        """Evict oldest entry (FIFO)."""
        if not self.cache:
            return

        oldest_key = next(iter(self.cache))
        del self.cache[oldest_key]

        logger.debug("embedding_evicted", cache_size=len(self.cache))

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

        logger.info("embedding_cache_cleared")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }

