"""Integration tests for end-to-end inference pipeline."""
import pytest

from inference_engine.domain.batching.adaptive import AdaptiveBatcher
from inference_engine.domain.caching.exact import ExactCache
from inference_engine.domain.models.batch import BatchStrategy
from inference_engine.domain.models.request import (
    InferenceRequest,
    ModelParameters,
    RequestPriority,
)


@pytest.mark.integration
class TestInferencePipeline:
    """End-to-end inference pipeline tests."""

    @pytest.mark.asyncio
    async def test_request_with_caching(self):
        """Test request flow with caching."""
        cache = ExactCache()

        # First request - miss
        request1 = InferenceRequest(
            prompt="What is machine learning?",
            parameters=ModelParameters(),
        )

        # Simulate processing
        from inference_engine.domain.models.response import (
            CacheInfo,
            InferenceResponse,
            UsageMetrics,
        )
        response1 = InferenceResponse(
            request_id=request1.id,
            text="Machine learning is a subset of AI.",
            model_used="test-model",
            usage=UsageMetrics(prompt_tokens=5, completion_tokens=10, total_tokens=15, cost_usd=0.0001),
            cache_info=CacheInfo(hit=False),
            latency_ms=100,
        )

        await cache.set(request1, response1)

        # Second request - hit
        request2 = InferenceRequest(
            prompt="What is machine learning?",
            parameters=ModelParameters(),
        )

        result = await cache.get(request2)
        assert result is not None
        response2, cache_info = result
        assert cache_info.hit is True

    @pytest.mark.asyncio
    async def test_batching_pipeline(self):
        """Test batching multiple requests."""
        batcher = AdaptiveBatcher(BatchStrategy(min_batch_size=2, max_batch_size=10))

        # Add requests
        for i in range(5):
            request = InferenceRequest(
                prompt=f"Question {i}",
                parameters=ModelParameters(),
                priority=RequestPriority.STANDARD,
            )
            await batcher.add_request(request)

        # Collect batch
        batch = await batcher.collect_batch()
        assert batch is not None
        assert batch.size >= 2
        assert batch.size <= 10

