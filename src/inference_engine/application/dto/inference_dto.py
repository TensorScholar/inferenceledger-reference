from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceInputDTO:
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7


@dataclass(frozen=True)
class InferenceOutputDTO:
    text: str
    model_used: str
    tokens_used: int
    cost_usd: float
    latency_ms: int
    cache_hit: bool

