"""Unit tests for caching strategies."""

from datetime import UTC, datetime, timedelta

import pytest

from inference_engine.domain.caching.eviction import CostAwareEvictionPolicy
from inference_engine.domain.caching.exact import ExactCache
from inference_engine.domain.models.cache import CacheEntry, CacheKey
from inference_engine.domain.models.request import InferenceRequest, ModelParameters
from inference_engine.domain.models.response import CacheInfo, InferenceResponse, UsageMetrics
from inference_engine.infrastructure.telemetry.request_log import RequestTrace


@pytest.fixture
def sample_request() -> InferenceRequest:
    """Create sample inference request."""
    return InferenceRequest(
        prompt="What is 2+2?",
        parameters=ModelParameters(max_tokens=50),
    )


@pytest.fixture
def sample_response(sample_request: InferenceRequest) -> InferenceResponse:
    """Create sample inference response."""
    return InferenceResponse(
        request_id=sample_request.id,
        text="The answer is 4.",
        model_used="test-model",
        usage=UsageMetrics(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.0001,
        ),
        cache_info=CacheInfo(hit=False),
        latency_ms=50,
    )


class TestExactCache:
    """Tests for ExactCache."""

    @pytest.mark.asyncio
    async def test_set_and_get(self, sample_request, sample_response):
        """A cache hit is a successful local outcome with zero provider execution."""
        cache = ExactCache(max_entries=1000)

        await cache.set(sample_request, sample_response)
        result = await cache.get(sample_request)
        assert result is not None

        response, cache_info = result
        assert cache_info.hit is True
        assert cache_info.tokens_saved == 15
        assert response.text == sample_response.text
        assert response.provider_attempt_count == 0
        assert response.provider_retry_count == 0
        assert response.usage.prompt_tokens == 0
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == 0
        assert response.usage.cost_usd == 0.0

        trace = RequestTrace.from_response(provider="local-cache", response=response)
        assert trace.provider_attempt_count == 0
        assert trace.estimated_cost_usd == 0.0
        assert trace.cost_evidence_complete is True
        assert trace.pricing_table_version == "not_charged"

    @pytest.mark.asyncio
    async def test_cache_miss(self, sample_request):
        """Test cache miss for non-existent entry."""
        cache = ExactCache()

        result = await cache.get(sample_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ExactCache(max_entries=2)

        for i in range(3):
            req = InferenceRequest(
                prompt=f"Query {i}",
                parameters=ModelParameters(),
            )
            resp = InferenceResponse(
                request_id=req.id,
                text=f"Response {i}",
                model_used="test",
                usage=UsageMetrics(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                cache_info=CacheInfo(hit=False),
                latency_ms=10,
            )
            await cache.set(req, resp)

        stats = cache.get_metrics()
        assert stats["cache_size"] == 2

    @pytest.mark.asyncio
    async def test_invalidate_pattern(self, sample_request, sample_response):
        """Test cache invalidation by pattern."""
        cache = ExactCache()

        await cache.set(sample_request, sample_response)

        count = await cache.invalidate("France")
        assert count == 0

        count = await cache.invalidate("2+2")
        assert count == 1

        result = await cache.get(sample_request)
        assert result is None

    def test_cache_metrics(self):
        """Test cache metrics tracking."""
        cache = ExactCache()

        metrics = cache.get_metrics()
        assert "hits" in metrics
        assert "misses" in metrics
        assert "hit_rate" in metrics
        assert "cache_size" in metrics


def test_cost_aware_eviction_falls_back_to_lru_when_any_cost_is_unknown() -> None:
    policy = CostAwareEvictionPolicy()
    baseline = datetime(2026, 1, 1, tzinfo=UTC)
    known = CacheEntry(
        key=CacheKey(content_hash="known", model="test", temperature=0.0, max_tokens=1),
        prompt="known",
        response="known",
        cost_usd=0.01,
        access_count=5,
        last_accessed=baseline,
    )
    unknown = CacheEntry(
        key=CacheKey(content_hash="unknown", model="test", temperature=0.0, max_tokens=1),
        prompt="unknown",
        response="unknown",
        cost_usd=None,
        access_count=5,
        last_accessed=baseline + timedelta(seconds=1),
    )

    selected = policy.select_to_evict([known, unknown])

    assert selected is known
    assert unknown.cost_savings is None
