"""
Unit tests for the InteractionSolver LLM client factory: the
``planning.llm_planning`` section must be able to move the planner to its
own model/backend (NLU/planner role split, docs/multi_model_choice.md)
while the shared ``llm.providers`` block keeps serving the UserAssistant.
"""

import pytest

from ami_agents.agents.interaction_solver.utils.llm_client import build_llm_client


def _shared_provider(**overrides):
    provider = {
        "api_key": "sk-shared",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5-mini",
        "temperature": 1.0,
        "reasoning_effort": "low",
    }
    provider.update(overrides)
    return {"llm": {"default_provider": "openai", "providers": {"openai": provider}}}


class TestPlanningOverrides:
    def test_without_planning_section_uses_shared_provider(self):
        cfg = build_llm_client(_shared_provider())
        assert cfg.model == "gpt-5-mini"
        assert cfg.base_url == "https://api.openai.com/v1"
        assert cfg.reasoning_effort == "low"
        assert cfg.client.api_key == "sk-shared"

    def test_planning_section_moves_planner_to_its_own_backend(self):
        config = _shared_provider()
        config["planning"] = {
            "llm_planning": {
                "model": "qwen2.5-coder:3b",
                "base_url": "http://127.0.0.1:11500/v1",
                "api_key": "dummy",
                "reasoning_effort": "",
                "temperature": 0.5,
            }
        }
        cfg = build_llm_client(config)
        assert cfg.model == "qwen2.5-coder:3b"
        assert cfg.base_url == "http://127.0.0.1:11500/v1"
        assert cfg.client.api_key == "dummy"
        assert str(cfg.client.base_url).rstrip("/") == "http://127.0.0.1:11500/v1"
        assert cfg.temperature == 0.5
        # Empty planning-side effort is authoritative: nothing inherited from the provider.
        assert cfg.reasoning_effort is None

    def test_model_only_override_keeps_shared_backend(self):
        config = _shared_provider()
        config["planning"] = {"llm_planning": {"model": "gpt-4o"}}
        cfg = build_llm_client(config)
        assert cfg.model == "gpt-4o"
        assert cfg.base_url == "https://api.openai.com/v1"
        assert cfg.client.api_key == "sk-shared"
        assert cfg.reasoning_effort == "low"

    def test_absent_planning_effort_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")
        cfg = build_llm_client(_shared_provider(reasoning_effort=None))
        assert cfg.reasoning_effort == "medium"
