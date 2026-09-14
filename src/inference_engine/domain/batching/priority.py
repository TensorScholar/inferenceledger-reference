
import structlog

from ..models.batch import BatchRequest, BatchStrategy
from ..models.request import InferenceRequest, RequestPriority
from .base import AbstractBatcher

logger = structlog.get_logger()


class PriorityBatcher(AbstractBatcher):
    """
    Priority-aware batcher that strictly enforces priority lanes.

    Ensures high-priority requests never wait behind lower-priority ones.
    """

    def __init__(self, strategy: BatchStrategy) -> None:
        self.strategy = strategy
        self.queues: dict[RequestPriority, list[InferenceRequest]] = {
            RequestPriority.EXPRESS: [],
            RequestPriority.STANDARD: [],
            RequestPriority.BATCH: [],
        }

    async def add_request(self, request: InferenceRequest) -> None:
        self.queues[request.priority].append(request)
        logger.debug(
            "priority_request_queued",
            request_id=str(request.id),
            priority=request.priority.value,
            queue_size=len(self.queues[request.priority]),
        )

    async def collect_batch(self) -> BatchRequest | None:
        # Strict priority: only collect from highest-priority non-empty queue
        for priority in [RequestPriority.EXPRESS, RequestPriority.STANDARD, RequestPriority.BATCH]:
            queue = self.queues[priority]
            if queue:
                return await self._collect_from_queue(queue, priority)

        return None

    async def _collect_from_queue(
        self, queue: list[InferenceRequest], priority: RequestPriority
    ) -> BatchRequest:
        # Collect based on priority-specific limits
        if priority == RequestPriority.EXPRESS:
            max_size = 4
        elif priority == RequestPriority.STANDARD:
            max_size = self.strategy.max_batch_size
        else:
            max_size = self.strategy.max_batch_size

        batch_requests = queue[:max_size]
        self.queues[priority] = queue[max_size:]

        logger.info(
            "priority_batch_collected",
            priority=priority.value,
            size=len(batch_requests),
        )

        return BatchRequest(requests=batch_requests, strategy=self.strategy)

    def get_queue_stats(self) -> dict:
        return {
            "express": len(self.queues[RequestPriority.EXPRESS]),
            "standard": len(self.queues[RequestPriority.STANDARD]),
            "batch": len(self.queues[RequestPriority.BATCH]),
            "total": sum(len(q) for q in self.queues.values()),
        }

