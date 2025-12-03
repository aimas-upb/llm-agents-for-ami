"""Configuration settings for RD5 Plan Creation system.

This module defines all configuration options using Pydantic Settings
for environment variable support and validation.
"""

from functools import lru_cache
from typing import Optional, Set

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """RD5 system configuration.

    All settings can be overridden via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application settings
    app_name: str = "RD5 Plan Orchestrator"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # OpenAI settings
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model for code generation",
    )
    openai_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM responses (lower = more deterministic)",
    )
    openai_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens in LLM response",
    )

    # Weaviate settings
    weaviate_host: str = Field(default="localhost", description="Weaviate host")
    weaviate_port: int = Field(default=8081, description="Weaviate port")
    weaviate_grpc_port: int = Field(default=50051, description="Weaviate gRPC port")

    # RD4 Signifier API settings
    rd4_api_url: str = Field(
        default="http://localhost:8000",
        description="RD4 Signifier API base URL",
    )
    rd4_api_timeout: float = Field(
        default=30.0,
        description="Timeout for RD4 API requests in seconds",
    )

    # HMAS/Yggdrasil settings
    hmas_base_url: str = Field(
        default="http://localhost:8080",
        description="HMAS platform base URL",
    )

    # Sandbox execution settings
    sandbox_timeout: int = Field(
        default=30,
        description="Sandbox execution timeout in seconds",
    )
    sandbox_memory_limit: str = Field(
        default="128m",
        description="Docker sandbox memory limit",
    )
    sandbox_cpu_limit: float = Field(
        default=0.5,
        description="Docker sandbox CPU limit (0.5 = 50%)",
    )
    sandbox_network: str = Field(
        default="rd5_sandbox_net",
        description="Docker network for sandbox isolation",
    )

    # Code validation settings
    allowed_imports: Set[str] = Field(
        default={
            "aiohttp",
            "json",
            "datetime",
            "math",
            "networkx",
            "asyncio",
            "typing",
            "re",
            "collections",
            "itertools",
            "functools",
        },
        description="Whitelist of allowed Python imports in generated code",
    )

    blocked_patterns: Set[str] = Field(
        default={
            "os",
            "sys",
            "subprocess",
            "shutil",
            "eval",
            "exec",
            "compile",
            "__import__",
            "open",
            "file",
            "socket",
            "urllib",
            "requests",
            "globals",
            "locals",
            "getattr",
            "setattr",
            "delattr",
            "vars",
            "dir",
            "breakpoint",
            "input",
        },
        description="Blocked patterns in generated code",
    )

    # Workflow settings
    max_code_generation_retries: int = Field(
        default=3,
        description="Maximum retries for code generation on validation failure",
    )

    @property
    def weaviate_url(self) -> str:
        """Get full Weaviate URL."""
        return f"http://{self.weaviate_host}:{self.weaviate_port}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance with values from environment.
    """
    return Settings()
