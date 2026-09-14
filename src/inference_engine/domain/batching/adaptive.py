from __future__ import annotations

from collections import deque

import structlog

from ...utils.time import utc_now
from ..models.batch import BatchMetrics, BatchRequest, BatchStrategy
from ..models.request import InferenceRequest, RequestPriority
from .base import AbstractBatcher

logger = structlog.get_logger()


class AdaptiveBatcher(AbstractBatcher):
    """Small adaptive batcher used as a Phase 0 domain primitive."""

    def __init__(self, strategy: BatchStrategy) -> None:
        self.strategy = strategy
        self.express_queue: deque[InferenceRequest] = deque()
        self.standard_queue: deque[InferenceRequest] = deque()
        self.batch_queue: deque[InferenceRequest] = deque()
        self.current_batch_size = strategy.min_batch_size
        self.recent_latencies: deque[int] = deque(maxlen=100)
        self.total_batches = 0
        self.total_requests = 0

    async def add_request(self, request: InferenceRequest) -> None:
        if request.priority == RequestPriority.EXPRESS:
            self.express_queue.append(request)
        elif request.priority == RequestPriority.BATCH:
            self.batch_queue.append(request)
        else:
            self.standard_queue.append(request)

    async def collect_batch(self) -> BatchRequest | None:
        if self.express_queue:
            return self._collect_express_batch()
        if len(self.standard_queue) >= self.current_batch_size:
            return self._collect_standard_batch()
        if len(self.batch_queue) >= self.strategy.max_batch_size:
            return self._collect_background_batch()
        return self._collect_timed_batch()

    def _collect_express_batch(self) -> BatchRequest:
        max_size = min(4, len(self.express_queue))
        requests = [self.express_queue.popleft() for _ in range(max_size)]
        return BatchRequest(
            requests=requests,
            strategy=BatchStrategy(min_batch_size=1, max_batch_size=4, max_wait_ms=10),
        )

    def _collect_standard_batch(self) -> BatchRequest:
        target_size = self._calculate_optimal_batch_size()
        requests = [
            self.standard_queue.popleft()
            for _ in range(min(target_size, len(self.standard_queue)))
        ]
        return BatchRequest(requests=requests, strategy=self.strategy)

    def _collect_background_batch(self) -> BatchRequest:
        requests = [
            self.batch_queue.popleft()
            for _ in range(min(self.strategy.max_batch_size, len(self.batch_queue)))
        ]
        return BatchRequest(requests=requests, strategy=self.strategy)

    def _collect_timed_batch(self) -> BatchRequest | None:
        if self._oldest_request_age_ms() < self.strategy.max_wait_ms:
            return None
        target_size = self._calculate_optimal_batch_size()
        requests: list[InferenceRequest] = []
        while len(requests) < target_size and self.standard_queue:
            requests.append(self.standard_queue.popleft())
        while len(requests) < target_size and self.batch_queue:
            requests.append(self.batch_queue.popleft())
        if not requests:
            return None
        return BatchRequest(requests=requests, strategy=self.strategy)

    def _calculate_optimal_batch_size(self) -> int:
        if not self.recent_latencies:
            return self.current_batch_size
        sorted_latencies = sorted(self.recent_latencies)
        p95_index = min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))
        p95_latency = sorted_latencies[p95_index]
        if p95_latency < self.strategy.target_latency_p95_ms * 0.8:
            self.current_batch_size = min(
                self.strategy.max_batch_size, max(self.current_batch_size + 1, int(self.current_batch_size * 1.2))
            )
        elif p95_latency > self.strategy.target_latency_p95_ms:
            self.current_batch_size = max(
                self.strategy.min_batch_size, int(self.current_batch_size * 0.8)
            )
        return self.current_batch_size

    def _oldest_request_age_ms(self) -> int:
        candidates = [
            queue[0]
            for queue in (self.express_queue, self.standard_queue, self.batch_queue)
            if queue
        ]
        if not candidates:
            return 0
        oldest = min(candidates, key=lambda request: request.timestamp)
        return int((utc_now() - oldest.timestamp).total_seconds() * 1000)

    async def record_batch_metrics(self, metrics: BatchMetrics) -> None:
        self.recent_latencies.append(metrics.processing_time_ms)
        self.total_batches += 1
        self.total_requests += metrics.size
        logger.debug("batch_metrics_recorded", batch_id=str(metrics.batch_id), size=metrics.size)

    def get_queue_stats(self) -> dict[str, int]:
        return {
            "express": len(self.express_queue),
            "standard": len(self.standard_queue),
            "batch": len(self.batch_queue),
            "total": len(self.express_queue) + len(self.standard_queue) + len(self.batch_queue),
            "current_batch_size": self.current_batch_size,
            "total_batches": self.total_batches,
            "total_requests": self.total_requests,
        }
