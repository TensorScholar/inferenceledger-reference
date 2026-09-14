from __future__ import annotations

from ...dto.inference_dto import InferenceInputDTO, InferenceOutputDTO


class InferenceService:
    """Phase 0 inference service.

    Real provider execution is intentionally not implemented in Phase 0. Returning
    echo responses here would make benchmark and API claims dishonest.
    """

    async def infer(self, dto: InferenceInputDTO) -> InferenceOutputDTO:
        raise NotImplementedError(
            "This legacy application service has no provider implementation; use the reference provider path or benchmark application services."
        )

