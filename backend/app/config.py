from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SVD FaceVault API"
    cors_origins: str = "http://localhost:5173"
    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/facevault.db")
    svd_ranks: str = "5,10,20,30,50,100"
    recognition_threshold: float = 0.63
    # Live webcam frames are noisier than uploaded stills; a slightly more
    # forgiving bar keeps the same-person match from flickering in and out.
    live_recognition_threshold: float = 0.55

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def rank_values(self) -> list[int]:
        return [int(rank.strip()) for rank in self.svd_ranks.split(",") if rank.strip()]

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
