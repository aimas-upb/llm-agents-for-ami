"""
Logging utilities for the AMI agent system.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class LoggerFactory:
    """Factory for creating configured loggers."""

    @staticmethod
    def create_logger(name: str, config: dict) -> logging.Logger:
        """
        Create a configured logger.

        Args:
            name: Logger name.
            config: Logging configuration from YAML.

        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger(name)

        # Prevent adding duplicate handlers
        if logger.hasHandlers():
            return logger

        # Set log level
        level = config.get("level", "INFO")
        logger.setLevel(getattr(logging, level.upper()))

        # Prevent propagation to avoid duplicate logs with root logger
        logger.propagate = False

        # Create formatter
        format_str = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        formatter = logging.Formatter(format_str)

        # Add file handler if enabled
        file_handler = LoggerFactory.create_file_handler(config)
        if file_handler:
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Add console handler if enabled
        console_handler = LoggerFactory.create_console_handler(config)
        if console_handler:
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    @staticmethod
    def create_file_handler(config: dict) -> Optional[RotatingFileHandler]:
        """
        Create a rotating file handler.

        Args:
            config: Logging configuration from YAML.

        Returns:
            RotatingFileHandler or None if disabled.
        """
        file_config = config.get("file_logging", {})

        if not file_config.get("enabled", False):
            return None

        log_file = file_config.get("path", "logs/ami_agents.log")
        max_bytes = file_config.get("max_bytes", 10485760)  # 10MB
        backup_count = file_config.get("backup_count", 5)

        # Create log directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )

        return handler

    @staticmethod
    def create_console_handler(config: dict) -> Optional[logging.StreamHandler]:
        """
        Create a console handler.

        Args:
            config: Console logging configuration.

        Returns:
            StreamHandler or None if disabled.
        """
        console_config = config.get("console_logging", {})

        if not console_config.get("enabled", True):  # Default to enabled
            return None

        handler = logging.StreamHandler(sys.stdout)
        return handler

    @staticmethod
    def merge_configs(global_config: dict, agent_config: Optional[dict] = None) -> dict:
        """
        Merge global and agent-specific logging configurations.

        Args:
            global_config: Global logging configuration from agents.yaml
            agent_config: Agent-specific logging configuration (optional)

        Returns:
            Merged configuration with agent-specific overrides
        """
        if not agent_config:
            return global_config

        merged = global_config.copy()

        # Override level if specified
        if "level" in agent_config:
            merged["level"] = agent_config["level"]

        # Override format if specified
        if "format" in agent_config:
            merged["format"] = agent_config["format"]

        # Merge file_logging settings
        if "file_logging" in agent_config:
            merged_file = merged.get("file_logging", {}).copy()
            merged_file.update(agent_config["file_logging"])
            merged["file_logging"] = merged_file

        # Merge console_logging settings
        if "console_logging" in agent_config:
            merged_console = merged.get("console_logging", {}).copy()
            merged_console.update(agent_config["console_logging"])
            merged["console_logging"] = merged_console

        return merged

    @staticmethod
    def get_logger(name: str, config: Optional[dict] = None) -> logging.Logger:
        """
        Get or create a logger with configuration.

        Args:
            name: Logger name.
            config: Logging configuration from YAML (optional).

        Returns:
            Configured logger instance.
        """
        if config:
            return LoggerFactory.create_logger(name, config)
        else:
            # Fallback to basic logger if no config provided
            return logging.getLogger(name)
