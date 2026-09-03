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
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_flash_model: str = "deepseek-v4-flash"
    llm_pro_model: str = "deepseek-v4-pro"
    llm_timeout_seconds: float = 30.0
    http_timeout_seconds: float = 20.0
    http_retries: int = Field(default=2, ge=0, le=5)
    max_pro_analyst_items: int = Field(default=3, ge=0)
    max_pro_critic_items: int = Field(default=3, ge=0)
    max_pro_weekly_reviews: int = Field(default=1, ge=0)
    arxiv_max_items: int = Field(default=10, ge=1, le=100)
    arxiv_lookback_days: int = Field(default=14, ge=1, le=90)
    repec_max_items: int = Field(default=10, ge=1, le=100)
    hyperliquid_assets: list[str] = ["BTC", "ETH", "SOL"]
    market_warmup_days: int = Field(default=33, ge=1, le=90)
    live_bootstrap_overlap_hours: int = Field(default=2, ge=0, le=24)
    live_incremental_max_hours: int = Field(default=30, ge=24, le=72)
    report_output_dir: str = "outputs"
    # --- Private delivery channels (all credentials stay in .env, never Git) ---
    email_enabled: bool = False
    smtp_host: str | None = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str | None = None
    smtp_password: str | None = None
    email_from: str | None = None
    email_to: list[str] | None = None
    discord_webhook_url: str | None = None
    # --- Public publishing ---
    publication_mode: str = "DRAFT_ONLY"  # DISABLED | DRAFT_ONLY | AUTO_PUBLISH
    publication_language: str = "ENGLISH"  # ENGLISH | CHINESE | BILINGUAL
    x_api_key: str | None = None
    x_api_secret: str | None = None
    x_access_token: str | None = None
    x_access_secret: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


def env_overrides(**values: Any) -> Settings:
    return Settings(**values)
