from __future__ import annotations

from dataclasses import dataclass

from .inference_dto import InferenceInputDTO, InferenceOutputDTO


@dataclass(frozen=True)
class BatchInputDTO:
    requests: list[InferenceInputDTO]


@dataclass(frozen=True)
class BatchOutputDTO:
    responses: list[InferenceOutputDTO]

