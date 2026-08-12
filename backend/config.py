"""Application configuration — loaded from .env via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment_name: str = "gpt-5.4-mini"
    azure_openai_api_version: str = "2024-12-01-preview"

    # Feature flags
    use_llm: bool = True

    # Search
    search_top_k: int = 10

    # Catalog
    dummyjson_base_url: str = "https://dummyjson.com"
    catalog_refresh_interval_seconds: int = 3600  # 1 hour


settings = Settings()
