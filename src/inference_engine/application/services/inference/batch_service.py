
import structlog

from ...dto.batch_dto import BatchInputDTO, BatchOutputDTO
from ...dto.inference_dto import InferenceOutputDTO
from .inference_service import InferenceService

logger = structlog.get_logger()


class BatchService:
    """Service for processing batch inference requests."""

    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service

    async def process_batch(self, batch_input: BatchInputDTO) -> BatchOutputDTO:
        """
        Process batch of inference requests.

        Args:
            batch_input: Batch input DTO with multiple requests

        Returns:
            Batch output DTO with responses
        """
        responses: list[InferenceOutputDTO] = []

        for request in batch_input.requests:
            response = await self.inference_service.infer(request)
            responses.append(response)

        logger.info("batch_processed", batch_size=len(responses))

        return BatchOutputDTO(responses=responses)

