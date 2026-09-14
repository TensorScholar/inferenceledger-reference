from __future__ import annotations

from ..models.batch import BatchRequest, BatchStrategy
from ..models.request import InferenceRequest
from .base import AbstractBatcher


class SemanticBatcher(AbstractBatcher):
    """Placeholder-free minimal semantic batcher.

    Phase 0 only groups requests by exact leading text. Later phases can replace
    this with embedding-backed grouping once benchmark evidence requires it.
    """

    def __init__(self, strategy: BatchStrategy) -> None:
        self.strategy = strategy
        self.queue: list[InferenceRequest] = []

    async def add_request(self, request: InferenceRequest) -> None:
        self.queue.append(request)

    async def collect_batch(self) -> BatchRequest | None:
        if len(self.queue) < self.strategy.min_batch_size:
            return None
        seed = self.queue[0].input_text[:32]
        selected = [request for request in self.queue if request.input_text.startswith(seed)]
        selected = selected[: self.strategy.max_batch_size]
        selected_ids = {request.id for request in selected}
        self.queue = [request for request in self.queue if request.id not in selected_ids]
        return BatchRequest(requests=selected, strategy=self.strategy)

