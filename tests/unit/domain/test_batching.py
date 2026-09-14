"""Unit tests for batching strategies."""

import pytest

from inference_engine.domain.batching.adaptive import AdaptiveBatcher
from inference_engine.domain.batching.priority import PriorityBatcher
from inference_engine.domain.models.batch import BatchStrategy
from inference_engine.domain.models.request import (
    InferenceRequest,
    ModelParameters,
    RequestPriority,
)


@pytest.fixture
def sample_request() -> InferenceRequest:
    """Create a sample inference request."""
    return InferenceRequest(
        prompt="What is the capital of France?",
        parameters=ModelParameters(max_tokens=100),
        priority=RequestPriority.STANDARD,
    )


@pytest.fixture
def batch_strategy() -> BatchStrategy:
    """Create batch strategy configuration."""
    return BatchStrategy(
        min_batch_size=2,
        max_batch_size=8,
        max_wait_ms=100,
        target_latency_p95_ms=150,
    )


class TestAdaptiveBatcher:
    """Tests for AdaptiveBatcher."""

    @pytest.mark.asyncio
    async def test_add_request(self, batch_strategy, sample_request):
        """Test adding requests to batcher."""
        batcher = AdaptiveBatcher(batch_strategy)
        await batcher.add_request(sample_request)

        stats = batcher.get_queue_stats()
        assert stats["standard"] == 1

    @pytest.mark.asyncio
    async def test_collect_batch(self, batch_strategy):
        """Test batch collection."""
        batcher = AdaptiveBatcher(batch_strategy)

        # Add multiple requests
        for i in range(5):
            req = InferenceRequest(
                prompt=f"Question {i}",
                parameters=ModelParameters(),
                priority=RequestPriority.STANDARD,
            )
            await batcher.add_request(req)

        batch = await batcher.collect_batch()
        assert batch is not None
        assert batch.size >= 2
        assert batch.size <= batch_strategy.max_batch_size

    @pytest.mark.asyncio
    async def test_express_lane_priority(self, batch_strategy):
        """Test express lane takes priority."""
        batcher = AdaptiveBatcher(batch_strategy)

        # Add express request
        express_req = InferenceRequest(
            prompt="Express query",
            parameters=ModelParameters(),
            priority=RequestPriority.EXPRESS,
        )
        await batcher.add_request(express_req)

        batch = await batcher.collect_batch()
        assert batch is not None
        assert batch.priority == RequestPriority.EXPRESS


class TestPriorityBatcher:
    """Tests for PriorityBatcher."""

    @pytest.mark.asyncio
    async def test_strict_priority_lanes(self, batch_strategy):
        """Test priority lanes don't mix."""
        batcher = PriorityBatcher(batch_strategy)

        # Add requests across priorities
        await batcher.add_request(InferenceRequest(prompt="A", priority=RequestPriority.STANDARD))
        await batcher.add_request(InferenceRequest(prompt="B", priority=RequestPriority.EXPRESS))
        await batcher.add_request(InferenceRequest(prompt="C", priority=RequestPriority.BATCH))

        # Should collect from express first
        batch = await batcher.collect_batch()
        assert batch is not None
        assert batch.priority == RequestPriority.EXPRESS


class TestBatchStrategy:
    """Tests for BatchStrategy."""

    def test_validation_min_max(self):
        """Test validation of min/max batch sizes."""
        with pytest.raises(ValueError):
            BatchStrategy(min_batch_size=10, max_batch_size=5)

    def test_validation_passes(self):
        """Test valid configuration passes."""
        strategy = BatchStrategy(min_batch_size=2, max_batch_size=10)
        assert strategy.min_batch_size == 2
        assert strategy.max_batch_size == 10

