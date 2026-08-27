from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    database_url: str = "sqlite:///quant_radar.db"
    llm_provider: str = "fake"
    llm_model: str = "fake-v1"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout_seconds: float = 30.0
    http_timeout_seconds: float = 20.0
    http_retries: int = Field(default=2, ge=0, le=5)
    arxiv_max_items: int = Field(default=10, ge=1, le=100)
    arxiv_lookback_days: int = Field(default=14, ge=1, le=90)
    repec_max_items: int = Field(default=10, ge=1, le=100)
    hyperliquid_assets: list[str] = ["BTC", "ETH", "SOL"]
    report_output_dir: str = "outputs"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def env_overrides(**values: Any) -> Settings:
    return Settings(**values)
