
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_max_connections: int = 50

    # Models
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    vllm_model_path: str | None = None
    tgi_base_url: str | None = None

    # Batching
    batch_min_size: int = 4
    batch_max_size: int = 64
    batch_max_wait_ms: int = 50
    batch_target_latency_p95: int = 100
    enable_semantic_grouping: bool = True
    priority_lanes: bool = True

    # Caching
    semantic_cache_enabled: bool = True
    cache_similarity_threshold: float = 0.90
    cache_max_size: int = 10000
    cache_ttl_seconds: int = 3600
    prefix_cache_enabled: bool = True

    # Routing
    routing_strategy: str = "cost_optimal"  # cost_optimal, latency_optimal, balanced
    cost_weight: float = 0.7
    fallback_enabled: bool = True

    # Monitoring
    log_level: str = "INFO"
    enable_tracing: bool = True
    prometheus_port: int = 9090

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4


settings = Settings()


