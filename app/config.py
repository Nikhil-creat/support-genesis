"""Centralized application configuration via pydantic-settings."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    postgres_dsn: str = "postgresql+asyncpg://support:support@localhost:5432/support_engine"
    postgres_checkpoint_dsn: str = "postgresql://support:support@localhost:5432/support_engine"

    # Redis / semantic cache
    redis_url: str = "redis://localhost:6379/0"
    semantic_cache_similarity_threshold: float = 0.92
    semantic_cache_ttl_seconds: int = 3600

    # LLM
    openai_api_key: str = "sk-changeme"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Guardrails
    high_value_refund_threshold_usd: float = 100.0
    max_tool_retry_attempts: int = 3

    # Observability
    otel_exporter_endpoint: str = "http://localhost:4317"
    langsmith_project: str = "autonomous-support-engine"
    service_name: str = "support-engine"

    # MCP
    mcp_server_endpoints: list[str] = ["http://localhost:8801/mcp"]

    # Computer vision (CNN)
    cnn_weights_path: str | None = None
    cnn_confidence_threshold: float = 0.75

    # Security
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"


@lru_cache
def get_settings() -> Settings:
    return Settings()
