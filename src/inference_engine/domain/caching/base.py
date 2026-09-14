from __future__ import annotations

from abc import ABC, abstractmethod

from ..models.request import InferenceRequest
from ..models.response import CacheInfo, InferenceResponse


class AbstractCache(ABC):
    """Interface for response caches."""

    @abstractmethod
    async def get(self, request: InferenceRequest) -> tuple[InferenceResponse, CacheInfo] | None:
        raise NotImplementedError

    @abstractmethod
    async def set(self, request: InferenceRequest, response: InferenceResponse) -> None:
        raise NotImplementedError

    @abstractmethod
    async def invalidate(self, pattern: str | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self) -> dict[str, object]:
        raise NotImplementedError


class VectorStore(ABC):
    """Small vector-store interface for future semantic cache work."""

    @abstractmethod
    async def search(
        self, query_embedding: object, top_k: int, max_distance: float
    ) -> list[dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    async def add(self, id: str, embedding: object, metadata: dict[str, object]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> None:
        raise NotImplementedError

