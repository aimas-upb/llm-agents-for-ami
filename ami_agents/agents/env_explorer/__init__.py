"""
EnvExplorer agent package.

Refactored agent with proper separation of concerns:
- behaviors/: Extracted behavior classes
- utils/: Extracted utility functions
- config/: Configuration defaults
"""

from .env_explorer_agent import EnvExplorerAgent

__all__ = ['EnvExplorerAgent']