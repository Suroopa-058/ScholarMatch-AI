from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")
    app_name: str = "ScholarMatch AI Backend"
    api_prefix: str = "/api"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    models_dir: Path = BACKEND_DIR / "models"
    sbert_model_path: str = str(BACKEND_DIR / "models" / "sbert" / "model")
    top_k: int = 5
    database_path: Path = BACKEND_DIR / "data" / "scholarmatch.db"
    auth_secret: str = "change-this-in-production-with-a-long-random-secret"
    auth_token_hours: int = 24

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
