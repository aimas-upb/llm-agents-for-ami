"""
Utility functions for loading and validating configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """Utility class for loading YAML configurations."""

    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load a YAML file.

        Args:
            file_path: Path to the YAML file.

        Returns:
            Configuration dictionary.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config if config is not None else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")

    @staticmethod
    def load_with_env_vars(file_path: str) -> Dict[str, Any]:
        """
        Load YAML file with environment variable substitution.

        Args:
            file_path: Path to the YAML file.

        Returns:
            Configuration dictionary with env vars substituted.
        """
        import re

        config = ConfigLoader.load_yaml(file_path)

        def substitute_vars(obj):
            if isinstance(obj, str):
                # Replace ${VAR_NAME} or ${VAR_NAME:-default} with env var value
                pattern = r'\$\{([^}]+)\}'
                def replace_var(match):
                    var_expr = match.group(1)
                    # Handle bash-style default: VAR_NAME:-default
                    if ':-' in var_expr:
                        var_name, default_value = var_expr.split(':-', 1)
                        return os.environ.get(var_name, default_value)
                    else:
                        # Simple variable reference
                        return os.environ.get(var_expr, match.group(0))
                return re.sub(pattern, replace_var, obj)
            elif isinstance(obj, dict):
                return {k: substitute_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_vars(item) for item in obj]
            return obj

        return substitute_vars(config)

    @staticmethod
    def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against a schema.

        Args:
            config: Configuration dictionary.
            schema: Schema dictionary defining required fields.

        Returns:
            True if valid, False otherwise.
        """
        try:
            def validate_recursive(config_part, schema_part):
                if isinstance(schema_part, dict):
                    if not isinstance(config_part, dict):
                        return False
                    for key, value in schema_part.items():
                        if key not in config_part:
                            return False
                        if not validate_recursive(config_part[key], value):
                            return False
                elif isinstance(schema_part, type):
                    return isinstance(config_part, schema_part)
                return True

            return validate_recursive(config, schema)
        except Exception:
            return False

    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.

        Args:
            *configs: Variable number of config dictionaries.

        Returns:
            Merged configuration.
        """
        def deep_merge(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
            return target

        result = {}
        for config in configs:
            if config:
                deep_merge(result, config)
        return result
