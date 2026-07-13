from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    AI_SERVICE_HOST: str = "0.0.0.0"
    AI_SERVICE_PORT: int = 8000
    AI_SERVICE_DEBUG: bool = False
    AI_MODEL_PATH: str = "app/models/weights/eq_ka_gcn.pt"
    AI_MODEL_VERSION: str = "0.0.0-placeholder"
    LOG_LEVEL: str = "info"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
