"""Every LLM task the agents run, and the model each one is configured with.

Each agent names its LLM tasks in its own section of agents.yaml: the
UserAssistant and the EnvExplorer under `llm:`, the InteractionSolver under
`planning:`. A task with no model of its own falls back the way the agent's
client builder does (`user_assistant/utils/llm_client.py`,
`interaction_solver/utils/llm_client.py`). At start, `log_task_llms` prints
one line per task, for example:

    LLM user_assistant.atomic_segmentation: model=gpt-6-luna reasoning_effort=low
    LLM env_explorer.query_response: model=gpt-4o-mini reasoning_effort=-

so a run shows what it calls before it calls anything. `override_task_llms`
is how a test harness applies OPENAI_MODEL / OPENAI_REASONING_EFFORT to every
task at once.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

from .demo_log import demo
from .llm_models import is_reasoning_model

# Where each agent keeps its LLM tasks in agents.yaml.
_LLM_SECTIONS = {"user_assistant": "llm", "env_explorer": "llm"}
_ISA_PLANNING_TASKS = ("llm_planning", "explicit_planning")


# -- the tasks ---------------------------------------------------------------

def task_llm_sections(config: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """(agent, task, section) for every LLM task agents.yaml configures.

    `config` is the merged config holding every agent's section, as main.py
    and the test harnesses load it.
    """
    out = []
    # 1. The UserAssistant's and the EnvExplorer's `llm:` tasks.
    for agent, key in _LLM_SECTIONS.items():
        for task, section in ((config.get(agent) or {}).get(key) or {}).items():
            if isinstance(section, dict):
                out.append((agent, task, section))
    # 2. The InteractionSolver's planners, under `planning:`.
    planning = (config.get("interaction_solver") or {}).get("planning") or {}
    for task in _ISA_PLANNING_TASKS:
        if isinstance(planning.get(task), dict):
            out.append(("interaction_solver", task, planning[task]))
    return out


def override_task_llms(config: Dict[str, Any], model: str = "", effort: str = "") -> None:
    """Replace every task's model and/or reasoning effort; empty leaves agents.yaml's."""
    for _, _, section in task_llm_sections(config):
        if model:
            section["model"] = model
        if effort:
            section["reasoning_effort"] = effort


# -- what each task resolves to ---------------------------------------------

def resolved_llm(agent: str, section: Dict[str, Any], config: Dict[str, Any]) -> Tuple[str, str]:
    """(model, reasoning effort) as the agent's client builder resolves them;
    the effort is "-" for a model that takes none."""
    # 1. The model: the task's own, then (InteractionSolver only) the
    #    provider's, then the shared default.
    llm_root = config.get("default_llm") or {}
    provider = ((llm_root.get("providers") or {})
                .get(llm_root.get("default_provider", "openai")) or {})
    model = section.get("model")
    if not model and agent == "interaction_solver":
        model = provider.get("model")
    model = str(model or llm_root.get("default_model") or "gpt-4o-mini")
    # 2. The effort: only reasoning models take one; "high" when none is set.
    if not is_reasoning_model(model):
        return model, "-"
    effort = (section.get("reasoning_effort")
              or (provider.get("reasoning_effort") if agent == "interaction_solver" else None)
              or os.getenv("OPENAI_REASONING_EFFORT") or "high")
    return model, str(effort)


def log_task_llms(config: Dict[str, Any], logger: logging.Logger) -> None:
    """One log line per LLM task: the model and effort it will be called with,
    and once the base URL every client uses."""
    llm_root = config.get("default_llm") or {}
    provider = ((llm_root.get("providers") or {})
                .get(llm_root.get("default_provider", "openai")) or {})
    logger.info(demo(f"LLM endpoint: {provider.get('base_url') or 'https://api.openai.com/v1'}"))
    for agent, task, section in task_llm_sections(config):
        model, effort = resolved_llm(agent, section, config)
        logger.info(demo(f"LLM {agent}.{task}: model={model} reasoning_effort={effort}"))
