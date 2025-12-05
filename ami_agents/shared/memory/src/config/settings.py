"""Application settings and configuration.

This module defines global configuration for the RD4 signifier system.
"""

import logging
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Args:
        app_name: Application name
        version: API version
        storage_dir: Directory for file-based storage
        rdf_format: Default RDF serialization format
        enable_authoring_validation: Enable SHACL validation at ingest
        log_level: Logging level
        enabled_modules: List of enabled module versions
        latency_budgets_ms: Latency budgets per module
    """

    app_name: str = "RD4 Signifier System"
    version: str = "0.1.0"

    storage_dir: str = "./storage"
    rdf_format: str = "turtle"

    enable_authoring_validation: bool = False

    log_level: str = "INFO"

    enabled_modules: List[str] = ["sr:v1", "ms:v1", "rs:v1"]

    latency_budgets_ms: dict = {
        "total": 150,
        "im": 30,
        "sse": 20,
        "sv": 80,
        "rp": 10,
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Application settings
    """
    return Settings()


def setup_logging(settings: Settings) -> None:
    """Configure logging based on settings.

    Args:
        settings: Application settings
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
