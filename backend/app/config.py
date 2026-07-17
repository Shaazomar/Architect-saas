"""Application settings. All values overridable via environment variables."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARCH_", env_file=".env", extra="ignore")

    # --- service ---
    data_dir: Path = Path("data")
    cors_origins: list[str] = ["http://localhost:3000"]
    # If set, every request must carry it in the X-API-Key header.
    api_key: str | None = None

    # --- upload limits (defense against oversized bodies / decompression bombs) ---
    max_upload_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 40_000_000
    min_image_side: int = 64

    # --- rate limiting (per client IP, token bucket) ---
    rate_limit_per_minute: int = 60

    # --- reconstruction defaults ---
    wall_height_m: float = 2.7
    slab_thickness_m: float = 0.10
    # Assumed real-world thickness of the walls detected in the drawing; used to
    # derive the pixel->meter scale when the plan carries no explicit scale.
    wall_thickness_m: float = 0.23


settings = Settings()
