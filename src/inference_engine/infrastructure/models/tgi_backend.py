from collections.abc import AsyncIterator

import structlog

try:
    from text_generation import AsyncClient
except ImportError:
    AsyncClient = None

from ...domain.models.request import InferenceRequest
from ...domain.models.response import CacheInfo, InferenceResponse, UsageMetrics
from .base import AbstractModelBackend

logger = structlog.get_logger()


class TGIBackend(AbstractModelBackend):
    """Text-Generation-Inference (TGI) backend."""

    def __init__(self, base_url: str, model_name: str = "default"):
        if AsyncClient is None:
            raise ImportError("text_generation package not installed")

        self.client = AsyncClient(base_url)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference via TGI."""
        import time

        start = time.time()

        prompt = request.prompt or str(request.messages)

        response = await self.client.generate(
            prompt,
            max_new_tokens=request.parameters.max_tokens,
            temperature=request.parameters.temperature,
            top_p=request.parameters.top_p,
            top_k=request.parameters.top_k,
        )

        elapsed_ms = int((time.time() - start) * 1000)

        return InferenceResponse(
            request_id=request.id,
            text=response.generated_text,
            finish_reason="stop",
            model_used=self.model_name,
            usage=UsageMetrics(
                prompt_tokens=response.details.prefill_tokens,
                completion_tokens=response.details.generated_tokens,
                total_tokens=response.details.total_tokens,
                cost_usd=0.0,
            ),
            cache_info=CacheInfo(hit=False),
            latency_ms=elapsed_ms,
        )

    async def infer_batch(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        """Process batch via TGI."""
        prompts = [req.prompt or str(req.messages) for req in requests]

        responses = await self.client.generate_batch(
            prompts,
            max_new_tokens=requests[0].parameters.max_tokens,
            temperature=requests[0].parameters.temperature,
        )

        results = []
        for i, response in enumerate(responses):
            results.append(
                InferenceResponse(
                    request_id=requests[i].id,
                    text=response.generated_text,
                    finish_reason="stop",
                    model_used=self.model_name,
                    usage=UsageMetrics(
                        prompt_tokens=response.details.prefill_tokens,
                        completion_tokens=response.details.generated_tokens,
                        total_tokens=response.details.total_tokens,
                        cost_usd=0.0,
                    ),
                    cache_info=CacheInfo(hit=False),
                    latency_ms=0,  # Batch latency shared
                )
            )

        return results

    async def stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream tokens from TGI."""
        prompt = request.prompt or str(request.messages)

        async for chunk in self.client.generate_stream(
            prompt, max_new_tokens=request.parameters.max_tokens, temperature=request.parameters.temperature
        ):
            if chunk.token.special:
                continue
            yield chunk.token.text

    async def health_check(self) -> bool:
        """Check TGI backend health."""
        try:
            await self.client.health()
            return True
        except Exception as e:
            logger.error("tgi_health_check_failed", error=str(e))
            return False
