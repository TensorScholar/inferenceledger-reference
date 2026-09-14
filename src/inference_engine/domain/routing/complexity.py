from __future__ import annotations

from ..models.request import InferenceRequest
from ..models.routing import ComplexityEstimate


class ComplexityEstimator:
    """Heuristic request complexity estimator for early routing tests."""

    def __init__(self) -> None:
        self.reasoning_keywords = {
            "analyze",
            "explain",
            "compare",
            "evaluate",
            "reason",
            "synthesize",
            "step by step",
        }
        self.technical_domains = {
            "code",
            "programming",
            "algorithm",
            "mathematics",
            "science",
            "physics",
            "legal",
            "medical",
            "financial",
            "engineering",
            "quantum",
        }

    async def estimate(self, request: InferenceRequest) -> ComplexityEstimate:
        text = request.input_text.lower()
        factors: dict[str, float] = {}
        factors["length"] = min(1.0, len(text) / 2000)
        reasoning_count = sum(1 for keyword in self.reasoning_keywords if keyword in text)
        domain_count = sum(1 for domain in self.technical_domains if domain in text)
        factors["reasoning"] = min(1.0, reasoning_count / 3)
        factors["domain"] = min(1.0, domain_count / 2)
        factors["context"] = 0.5 if len(request.messages) > 2 else 0.0
        factors["output_length"] = min(1.0, request.parameters.max_tokens / 2000)
        weights = {
            "length": 0.2,
            "reasoning": 0.3,
            "domain": 0.2,
            "context": 0.15,
            "output_length": 0.15,
        }
        score = sum(factors[name] * weights[name] for name in factors)
        if reasoning_count and domain_count:
            score += 0.2
        if request.parameters.max_tokens >= 500 and reasoning_count:
            score += 0.05
        score = min(1.0, score)
        return ComplexityEstimate(
            score=score,
            factors=factors,
            input_length=len(text),
            estimated_reasoning_steps=reasoning_count,
            requires_context=len(request.messages) > 2,
            domain_specific=domain_count > 0,
        )
