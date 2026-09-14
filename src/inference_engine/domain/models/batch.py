from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from ...utils.time import utc_now
from .request import InferenceRequest, RequestPriority


@dataclass(frozen=True)
class BatchStrategy:
    """Configuration for request batching."""

    min_batch_size: int = 4
    max_batch_size: int = 64
    max_wait_ms: int = 50
    target_latency_p95_ms: int = 100
    enable_semantic_grouping: bool = True
    similarity_threshold: float = 0.85
    priority_lanes: bool = True
    express_max_wait_ms: int = 10

    def __post_init__(self) -> None:
        if self.min_batch_size < 1:
            raise ValueError("min_batch_size must be positive")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.min_batch_size > self.max_batch_size:
            raise ValueError("min_batch_size cannot exceed max_batch_size")


@dataclass
class BatchRequest:
    """A group of requests that can be processed together."""

    requests: list[InferenceRequest] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    strategy: BatchStrategy = field(default_factory=BatchStrategy)
    centroid_embedding: object | None = None
    common_prefix: str | None = None
    id: UUID = field(default_factory=uuid4)

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def priority(self) -> RequestPriority:
        priorities = [request.priority for request in self.requests]
        if RequestPriority.EXPRESS in priorities:
            return RequestPriority.EXPRESS
        if RequestPriority.STANDARD in priorities:
            return RequestPriority.STANDARD
        return RequestPriority.BATCH

    @property
    def estimated_tokens(self) -> int:
        return sum(request.estimated_input_tokens for request in self.requests)

    @property
    def age_ms(self) -> int:
        return int((utc_now() - self.created_at).total_seconds() * 1000)

    def can_add(self, request: InferenceRequest) -> bool:
        if self.size >= self.strategy.max_batch_size:
            return False
        return not (
            self.priority == RequestPriority.EXPRESS and request.priority != RequestPriority.EXPRESS
        )


@dataclass(frozen=True)
class BatchMetrics:
    """Metrics for a processed batch."""

    batch_id: UUID
    size: int
    total_tokens: int
    processing_time_ms: int
    wait_time_ms: int
    throughput_tokens_per_sec: float
    efficiency_score: float
    timestamp: datetime = field(default_factory=utc_now)
