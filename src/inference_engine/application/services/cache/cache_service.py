
import structlog

from ....domain.caching.exact import ExactCache
from ....domain.caching.prefix import PrefixCache
from ....domain.caching.semantic import SemanticCache
from ....domain.models.request import InferenceRequest
from ....domain.models.response import CacheInfo, InferenceResponse

logger = structlog.get_logger()


class CacheService:
    """
    Unified cache service coordinating all cache tiers.

    Checks cache hierarchy: exact → semantic → prefix
    """

    def __init__(
        self,
        exact_cache: ExactCache,
        semantic_cache: SemanticCache | None = None,
        prefix_cache: PrefixCache | None = None,
    ):
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.prefix_cache = prefix_cache

    async def get(
        self, request: InferenceRequest
    ) -> tuple[InferenceResponse, CacheInfo] | None:
        """
        Get cached response for request.

        Returns:
            Tuple of (response, cache_info) if hit, None otherwise
        """
        # Try exact cache first
        result = await self.exact_cache.get(request)
        if result is not None:
            logger.info("exact_cache_hit", request_id=str(request.id))
            return result

        # Try semantic cache
        if self.semantic_cache:
            result = await self.semantic_cache.get(request)
            if result is not None:
                logger.info("semantic_cache_hit", request_id=str(request.id))
                return result

        # Try prefix cache
        if self.prefix_cache:
            prefix_text = request.prompt or str(request.messages)
            entry = await self.prefix_cache.get_prefix(prefix_text)
            if entry is not None:
                logger.info("prefix_cache_hit", request_id=str(request.id))
                # Return a simplified response
                return None  # Prefix cache used for KV-cache, not full response

        logger.debug("cache_miss", request_id=str(request.id))
        return None

    async def set(self, request: InferenceRequest, response: InferenceResponse) -> None:
        """Store response in all applicable caches."""
        await self.exact_cache.set(request, response)

        if self.semantic_cache:
            await self.semantic_cache.set(request, response)

        # Prefix cache managed separately via set_prefix

        logger.debug("cache_set_complete", request_id=str(request.id))

    async def invalidate(self, pattern: str | None = None) -> int:
        """Invalidate caches."""
        count = await self.exact_cache.invalidate(pattern)

        if self.semantic_cache:
            count += await self.semantic_cache.invalidate(pattern)

        logger.info("cache_invalidated", pattern=pattern, count=count)
        return count

    def get_metrics(self) -> dict:
        """Get aggregated cache metrics."""
        metrics = self.exact_cache.get_metrics()

        if self.semantic_cache:
            sem_metrics = self.semantic_cache.get_metrics()
            metrics["semantic"] = sem_metrics

        return metrics

