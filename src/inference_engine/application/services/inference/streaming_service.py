from collections.abc import AsyncIterator

import structlog

from ...dto.inference_dto import InferenceInputDTO
from .inference_service import InferenceService

logger = structlog.get_logger()


class StreamingService:
    """Service for streaming inference responses."""

    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service

    async def stream(self, request: InferenceInputDTO) -> AsyncIterator[str]:
        """
        Stream tokens for inference request.

        Args:
            request: Inference input DTO

        Yields:
            Token strings as they are generated
        """
        # For now, simple implementation: process and yield all at once
        # Real implementation would stream from model backend
        response = await self.inference_service.infer(request)

        # Yield in chunks
        chunk_size = 10
        for i in range(0, len(response.text), chunk_size):
            yield response.text[i : i + chunk_size]

        logger.info("stream_completed", text_length=len(response.text))

