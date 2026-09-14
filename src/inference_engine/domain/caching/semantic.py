from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..models.cache import SemanticCacheConfig
from ..models.request import InferenceRequest
from ..models.response import CacheInfo, InferenceResponse
from .base import AbstractCache, VectorStore


class SemanticCache(AbstractCache):
    """Disabled-by-default semantic cache shell.

    Phase 0 keeps this importable for service composition. Later phases should
    add correctness gates before semantic response reuse is enabled.
    """

    def __init__(
        self,
        config: SemanticCacheConfig,
        embedding_function: Callable[[str], Awaitable[object]],
        vector_store: VectorStore,
    ) -> None:
        self.config = config
        self.embedding_function = embedding_function
        self.vector_store = vector_store
        self.hits = 0
        self.misses = 0

    async def get(self, _request: InferenceRequest) -> tuple[InferenceResponse, CacheInfo] | None:
        self.misses += 1
        return None

    async def set(self, _request: InferenceRequest, _response: InferenceResponse) -> None:
        return None

    async def invalidate(self, _pattern: str | None = None) -> int:
        await self.vector_store.clear()
        return 0

    def get_metrics(self) -> dict[str, object]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }
