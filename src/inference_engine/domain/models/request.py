from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from ...utils.time import utc_now


class RequestPriority(StrEnum):
    """Request priority levels affecting batching behavior."""

    EXPRESS = "express"
    STANDARD = "standard"
    BATCH = "batch"


@dataclass(frozen=True)
class RequestMetadata:
    """Metadata used for attribution and benchmark slicing."""

    user_id: str | None = None
    session_id: str | None = None
    feature_name: str | None = None
    experiment_id: str | None = None
    application: str = "default"
    environment: str = "development"
    custom_tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelParameters:
    """Generation parameters that affect model output and cache keys."""

    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class InferenceRequest:
    """Canonical inference request used by domain services."""

    prompt: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    parameters: ModelParameters = field(default_factory=ModelParameters)
    priority: RequestPriority = RequestPriority.STANDARD
    metadata: RequestMetadata = field(default_factory=RequestMetadata)
    timestamp: datetime = field(default_factory=utc_now)
    use_cache: bool = True
    cache_ttl_seconds: int | None = None
    preferred_model: str | None = None
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.prompt and not self.messages:
            raise ValueError("Either prompt or messages must be provided")

    @property
    def input_text(self) -> str:
        if self.prompt:
            return self.prompt
        return " ".join(message.get("content", "") for message in self.messages)

    @property
    def estimated_input_tokens(self) -> int:
        return max(len(self.input_text) // 4, 1)

    @property
    def cache_key(self) -> str:
        import hashlib
        import json

        content: dict[str, Any] = {
            "prompt": self.prompt,
            "messages": self.messages,
            "temperature": self.parameters.temperature,
            "max_tokens": self.parameters.max_tokens,
            "top_p": self.parameters.top_p,
            "preferred_model": self.preferred_model,
        }
        raw = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()
