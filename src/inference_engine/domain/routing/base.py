from __future__ import annotations

from abc import ABC, abstractmethod

from ..models.request import InferenceRequest
from ..models.routing import RoutingDecision


class AbstractRouter(ABC):
    """Interface for model routers."""

    @abstractmethod
    async def route(self, request: InferenceRequest) -> RoutingDecision:
        raise NotImplementedError

