from __future__ import annotations

from abc import ABC, abstractmethod

from ..models.batch import BatchRequest
from ..models.request import InferenceRequest


class AbstractBatcher(ABC):
    """Interface for batching strategies."""

    @abstractmethod
    async def add_request(self, request: InferenceRequest) -> None:
        raise NotImplementedError

    @abstractmethod
    async def collect_batch(self) -> BatchRequest | None:
        raise NotImplementedError

