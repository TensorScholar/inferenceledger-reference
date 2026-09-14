from __future__ import annotations

from fastapi import FastAPI

from .routes.inference import router as inference_router

app = FastAPI(
    title="InferenceLedger Reference Executor",
    version="0.1.0",
    description=(
        "Narrow OpenAI-compatible reference execution surface used for controlled "
        "InferenceLedger experiments. It is not the product control plane."
    ),
)

app.include_router(inference_router, prefix="/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "component": "reference-executor"}


@app.get("/health/ready")
async def ready() -> dict[str, bool]:
    return {"ready": True}


@app.get("/health/live")
async def live() -> dict[str, bool]:
    return {"alive": True}
