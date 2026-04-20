"""Runtime configuration for the bank support application."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Environment-backed runtime settings."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    api_host: str = "127.0.0.1"
    api_port: int = 8000


def get_settings() -> Settings:
    """Load runtime settings from environment variables."""

    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
    )
