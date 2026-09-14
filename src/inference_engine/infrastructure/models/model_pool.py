
import structlog

from ...domain.models.routing import ModelConfig
from .base import AbstractModelBackend

logger = structlog.get_logger()


class ModelPool:
    """
    Manages pool of model backends with load balancing.

    Handles connection pooling, health checks, and failover.
    """

    def __init__(self) -> None:
        self.backends: dict[str, AbstractModelBackend] = {}
        self.configs: dict[str, ModelConfig] = {}

    def register_backend(self, model_id: str, backend: AbstractModelBackend, config: ModelConfig) -> None:
        """Register a model backend."""
        self.backends[model_id] = backend
        self.configs[model_id] = config

        logger.info("backend_registered", model_id=model_id, backend_type=type(backend).__name__)

    def get_backend(self, model_id: str) -> AbstractModelBackend | None:
        """Get backend by model ID."""
        return self.backends.get(model_id)

    def list_models(self) -> list[str]:
        """List all registered model IDs."""
        return list(self.backends.keys())

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all backends."""
        results = {}

        for model_id, backend in self.backends.items():
            try:
                results[model_id] = await backend.health_check()
            except Exception as e:
                logger.error("backend_health_check_failed", model_id=model_id, error=str(e))
                results[model_id] = False

        return results

    async def close_all(self) -> None:
        """Close all backends."""
        for model_id, backend in self.backends.items():
            try:
                if hasattr(backend, "close"):
                    await backend.close()
            except Exception as e:
                logger.error("backend_close_failed", model_id=model_id, error=str(e))

        self.backends.clear()
        self.configs.clear()

        logger.info("all_backends_closed")

