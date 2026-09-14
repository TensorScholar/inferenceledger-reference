from __future__ import annotations

from ..models.request import InferenceRequest
from ..models.routing import ModelConfig


def required_context_tokens(request: InferenceRequest) -> int:
    """Conservative pre-execution context requirement used by routing eligibility checks."""
    return request.estimated_input_tokens + request.parameters.max_tokens


def supports_request_context(model: ModelConfig, request: InferenceRequest) -> bool:
    """Return whether the candidate can fit the estimated prompt plus maximum requested output."""
    return required_context_tokens(request) <= model.max_context_length
