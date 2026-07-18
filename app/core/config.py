from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. All tunables come from environment variables —
    no hardcoded values elsewhere in the codebase."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    qdrant_path: str = "./storage"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 20
    final_k: int = 5
    claude_model: str = "claude-sonnet-5"
    anthropic_api_key: str = ""
    max_context_tokens: int = 6000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
