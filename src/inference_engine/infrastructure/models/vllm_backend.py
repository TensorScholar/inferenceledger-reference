from collections.abc import AsyncIterator

import structlog

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from ...domain.models.request import InferenceRequest
from ...domain.models.response import CacheInfo, InferenceResponse, UsageMetrics
from .base import AbstractModelBackend

logger = structlog.get_logger()


class vLLMBackend(AbstractModelBackend):
    """vLLM-based local model backend with continuous batching."""

    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        if LLM is None:
            raise ImportError("vllm package not installed")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,  # KV-cache reuse
        )
        self._model_name = model_path

    @property
    def model_name(self) -> str:
        return self._model_name

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference via vLLM."""
        import time

        start = time.time()

        prompt = request.prompt or str(request.messages)

        params = SamplingParams(
            temperature=request.parameters.temperature,
            max_tokens=request.parameters.max_tokens,
            top_p=request.parameters.top_p,
            top_k=request.parameters.top_k,
        )

        outputs = self.llm.generate([prompt], params)

        elapsed_ms = int((time.time() - start) * 1000)
        result = outputs[0]

        return InferenceResponse(
            request_id=request.id,
            text=result.outputs[0].text,
            finish_reason=result.outputs[0].finish_reason or "stop",
            model_used=self.model_name,
            usage=UsageMetrics(
                prompt_tokens=len(result.prompt_token_ids),
                completion_tokens=len(result.outputs[0].token_ids),
                total_tokens=len(result.prompt_token_ids) + len(result.outputs[0].token_ids),
                cost_usd=0.0,
            ),
            cache_info=CacheInfo(hit=False),
            latency_ms=elapsed_ms,
        )

    async def infer_batch(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        """Process batch efficiently with vLLM continuous batching."""
        import time

        start = time.time()

        prompts = [req.prompt or str(req.messages) for req in requests]

        params = SamplingParams(
            temperature=requests[0].parameters.temperature,
            max_tokens=requests[0].parameters.max_tokens,
        )

        outputs = self.llm.generate(prompts, params)

        elapsed_ms = int((time.time() - start) * 1000)

        results = []
        for i, output in enumerate(outputs):
            results.append(
                InferenceResponse(
                    request_id=requests[i].id,
                    text=output.outputs[0].text,
                    finish_reason=output.outputs[0].finish_reason or "stop",
                    model_used=self.model_name,
                    usage=UsageMetrics(
                        prompt_tokens=len(output.prompt_token_ids),
                        completion_tokens=len(output.outputs[0].token_ids),
                        total_tokens=len(output.prompt_token_ids)
                        + len(output.outputs[0].token_ids),
                        cost_usd=0.0,
                    ),
                    cache_info=CacheInfo(hit=False),
                    latency_ms=elapsed_ms // len(requests),  # Average
                )
            )

        return results

    async def stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream tokens from vLLM."""
        prompt = request.prompt or str(request.messages)

        params = SamplingParams(
            temperature=request.parameters.temperature,
            max_tokens=request.parameters.max_tokens,
        )

        for output in self.llm.generate(prompt, params):
            yield output.outputs[0].text

    async def health_check(self) -> bool:
        """Check vLLM backend health."""
        return self.llm is not None
