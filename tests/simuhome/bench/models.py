"""Per-step model choices: a variant file -> one LLM client per step.

A variant names the model each reasoning step uses. Any OpenAI-compatible
endpoint works: no `base_url` means OpenAI; Ollama serves the same API at
`http://localhost:11434/v1`.

    name: mini-all
    segmentation:  {model: gpt-5-mini, reasoning_effort: low}
    structuring:   {model: gpt-5-mini, reasoning_effort: medium}
    planning:
      explicit:    {model: gpt-5-nano, reasoning_effort: low}
      general:     {model: gpt-5-mini, reasoning_effort: medium}
    response:      {model: gpt-5-mini, reasoning_effort: low}
    judge:         {model: gpt-4o-mini, temperature: 0.5}

Flow:
    load_variant(path)       YAML -> Variant (every step filled, defaults below)
    ModelSpec.ua_config()    the UA's LLMClientConfig for that step
    ModelSpec.isa_config()   the ISA's LLMClientConfig for that step
    ModelSpec.call_kwargs()  temperature, or reasoning effort + token budget

Reasoning models are recognised by name, as the agents do (`o*`, `gpt-5*`):
they get `reasoning_effort` and `max_completion_tokens`, the others
`temperature` and `max_tokens`. A local model whose name starts with "o" would
be misread; give it a tag that does not.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import yaml
from openai import AsyncOpenAI

from ami_agents.agents.interaction_solver.utils.llm_client import (
    LLMClientConfig as ISAClientConfig,
)
from ami_agents.agents.user_assistant.utils.llm_client import (
    LLMClientConfig as UAClientConfig,
    is_reasoning_model,
)

OPENAI_URL = "https://api.openai.com/v1"
# One request may take this long before it counts as failed (seconds).
REQUEST_TIMEOUT = 180.0

STEPS = ("segmentation", "structuring", "planning.explicit", "planning.general",
         "response", "judge")


@dataclass(frozen=True)
class ModelSpec:
    """One step's model and how it is called."""

    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0
    reasoning_effort: Optional[str] = None
    max_tokens: Optional[int] = None

    @property
    def reasoning(self) -> bool:
        return is_reasoning_model(self.model)

    @property
    def local(self) -> bool:
        return bool(self.base_url) and "openai.com" not in self.base_url

    def client(self) -> AsyncOpenAI:
        """A client for this spec. A local endpoint never inherits the
        environment's proxies or OpenAI key."""
        if self.local:
            return AsyncOpenAI(api_key=os.getenv(self.api_key_env or "", "") or "ollama",
                               base_url=self.base_url, timeout=REQUEST_TIMEOUT,
                               max_retries=0, http_client=httpx.AsyncClient(trust_env=False))
        return AsyncOpenAI(api_key=os.getenv(self.api_key_env or "OPENAI_API_KEY"),
                           base_url=self.base_url or OPENAI_URL, timeout=REQUEST_TIMEOUT)

    def call_kwargs(self) -> Dict[str, Any]:
        if self.reasoning:
            kwargs: Dict[str, Any] = {}
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            if self.max_tokens:
                kwargs["max_completion_tokens"] = self.max_tokens
            return kwargs
        kwargs = {"temperature": self.temperature}
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    def ua_config(self) -> UAClientConfig:
        return UAClientConfig(client=self.client(), model=self.model,
                              base_url=self.base_url or OPENAI_URL,
                              temperature=self.temperature,
                              reasoning_effort=self.reasoning_effort if self.reasoning else None,
                              max_completion_tokens=self.max_tokens if self.reasoning else None)

    def isa_config(self) -> ISAClientConfig:
        return ISAClientConfig(client=self.client(), model=self.model,
                               base_url=self.base_url or OPENAI_URL,
                               temperature=self.temperature,
                               reasoning_effort=self.reasoning_effort if self.reasoning else None,
                               max_tokens=None if self.reasoning else self.max_tokens,
                               max_completion_tokens=self.max_tokens if self.reasoning else None)

    def label(self) -> str:
        return self.model + (f"@{self.reasoning_effort}" if self.reasoning and
                             self.reasoning_effort else "")


# The agents' own defaults (`agents.yaml`), so a variant names only what it changes.
DEFAULTS: Dict[str, ModelSpec] = {
    "segmentation": ModelSpec("gpt-5-mini", reasoning_effort="low"),
    "structuring": ModelSpec("gpt-5-mini", reasoning_effort="medium"),
    "planning.explicit": ModelSpec("gpt-5-nano", reasoning_effort="low"),
    "planning.general": ModelSpec("gpt-5-mini", reasoning_effort="medium"),
    "response": ModelSpec("gpt-5-mini", reasoning_effort="low"),
    "judge": ModelSpec("gpt-4o-mini", temperature=0.5),
}


@dataclass(frozen=True)
class Variant:
    """A name and one model spec per step."""

    name: str
    specs: Dict[str, ModelSpec] = field(default_factory=dict)

    def __getitem__(self, step: str) -> ModelSpec:
        return self.specs[step]

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name,
                "steps": {step: {k: v for k, v in asdict(spec).items() if v is not None}
                          for step, spec in self.specs.items()}}

    def key(self, *steps: str) -> str:
        """A short hash of the specs of ``steps`` -- what a cached result of
        those steps depends on."""
        blob = json.dumps({s: asdict(self.specs[s]) for s in steps}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _spec(raw: Any, default: ModelSpec) -> ModelSpec:
    if raw is None:
        return default
    if isinstance(raw, str):
        return ModelSpec(raw)
    known = {k: raw[k] for k in ModelSpec.__dataclass_fields__ if k in raw}
    return ModelSpec(**known)


def variant_from_dict(raw: Dict[str, Any], name: Optional[str] = None) -> Variant:
    """A variant from its YAML mapping; unnamed steps take the defaults."""
    planning = raw.get("planning") or {}
    given = {
        "segmentation": raw.get("segmentation"),
        "structuring": raw.get("structuring"),
        "planning.explicit": planning.get("explicit") if isinstance(planning, dict) else None,
        "planning.general": planning.get("general") if isinstance(planning, dict) else None,
        "response": raw.get("response"),
        "judge": raw.get("judge"),
    }
    specs = {step: _spec(given[step], DEFAULTS[step]) for step in STEPS}
    return Variant(name=str(name or raw.get("name") or "default"), specs=specs)


def load_variant(path: Optional[str]) -> Variant:
    """The variant in ``path``, or the agents' defaults when none is given."""
    if not path:
        return variant_from_dict({}, "default")
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return variant_from_dict(raw, raw.get("name") or Path(path).stem)


def with_step(variant: Variant, step: str, spec: ModelSpec, name: str) -> Variant:
    """``variant`` with one step's spec replaced (for sweeps)."""
    return replace(variant, name=name, specs={**variant.specs, step: spec})
