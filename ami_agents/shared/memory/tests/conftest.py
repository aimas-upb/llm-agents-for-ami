"""Make the engine importable as `src.*` wherever pytest is started from."""

from ami_agents.shared.memory import ensure_engine_on_path

ensure_engine_on_path()
