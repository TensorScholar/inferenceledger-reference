from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import structlog

from ...domain.models.request import InferenceRequest
from ...domain.models.response import InferenceResponse

logger = structlog.get_logger()


class AbstractModelBackend(ABC):
    """Abstract interface for model inference backends."""

    @abstractmethod
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Run inference on a request.

        Args:
            request: Inference request

        Returns:
            Inference response with generated text and metadata
        """
        raise NotImplementedError

    @abstractmethod
    async def infer_batch(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        """
        Run inference on a batch of requests.

        Args:
            requests: List of inference requests

        Returns:
            List of inference responses
        """
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """
        Stream inference tokens.

        Args:
            request: Inference request

        Yields:
            Token strings
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if backend is healthy.

        Returns:
            True if healthy, False otherwise
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        raise NotImplementedError
