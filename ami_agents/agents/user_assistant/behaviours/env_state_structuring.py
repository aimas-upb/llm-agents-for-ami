"""Behaviour: structure one ENV_STATE_REQUEST into ontology class identifiers.

Stage 2 of the request pipeline. The segmenter has already decided this is a
state request; this turns the natural-language fragment into the classes a
SPARQL query over the TD graph is built from -- a location, a device kind, and
either a device property or an environment variable -- and, for a device the
request calls by name, which of the home's devices has that name
(`utils/device_names.py`).

    1. the home's devices, from EnvExplorer (the goal context, unscoped by
       flags): what a name in the request is matched against
    2. one LLM call -> the classes, `device_reference` and `artifact_name`

One behaviour per intent, per CLAUDE.md 3.2: structuring one intent is
independent of structuring another, so several in a single utterance resolve
concurrently rather than one after the next.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from .. import pipeline
from ..prompts.intent_parsing_prompts import ENV_STATE_REQUEST_PARSER_PROMPT
from ..utils import loose_json_loads
from ..utils.llm_client import build_behaviour_llm_client, build_llm_call_kwargs
from ..utils.device_names import NO_DEVICES, device_slots
from ..utils.intent_slots import BRANCH_ROOTS, clean_class, clean_nested
from ..utils.ontology_context import get_ontology_context_json


class EnvStateStructuringBehaviour(OneShotBehaviour):
    """Parse ONE ENV_STATE_REQUEST intent into ontology class identifiers."""

    def __init__(self, intent_text: str, logger=None) -> None:
        super().__init__()
        self.intent_text = intent_text
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")
        # Populated by ``run`` -- a structured dict, or {} with ``error`` set.
        self.result: Dict[str, Any] = {}
        self.error: Optional[str] = None
        # Whether the home's devices were had: without them a device called by
        # name cannot be found, and the name slots are cleared.
        self._devices_known = False

    async def run(self) -> None:
        self.logger.info(demo(
            f"[LLM CALL] Structuring ENV_STATE_REQUEST: {self.intent_text[:80]!r}"
        ))
        # 1. The home's devices, so a device called by name can be found --
        #    or found missing. Without them the request is still parsed; the
        #    name slots are then cleared (`device_slots`).
        devices = await pipeline.fetch_goal_context(self.agent, self.logger,
                                                    self.intent_text)
        self._devices_known = bool(devices)
        # 2. The prompt, and the one call.
        prompt = ENV_STATE_REQUEST_PARSER_PROMPT.format(
            ontology_context=get_ontology_context_json(),
            home_devices=devices or NO_DEVICES,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": self.intent_text},
        ]

        try:
            llm_cfg = build_behaviour_llm_client(self.agent.config, "intent_parsing")
            response = await llm_cfg.client.chat.completions.create(
                model=llm_cfg.model,
                messages=messages,
                **build_llm_call_kwargs(llm_cfg),
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = loose_json_loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"expected a JSON object, got: {raw[:200]!r}")
            self.result = self._normalise(parsed)
            self.logger.info(demo(
                f"[ENV_STATE_REQUEST structured]\n{json.dumps(self.result, indent=2)}"
            ))
        except Exception as exc:
            self.error = str(exc)
            self.logger.warning("ENV_STATE structuring failed: %s", exc)
            # Enough for the caller to report the failure against the right
            # intent; every class slot stays absent.
            self.result = {
                "text_intent": self.intent_text,
                "location_class": None,
                "device_class": None,
                "device_property": None,
                "environment_variable": None,
            }

    def _normalise(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce the LLM's object into the shape callers may rely on."""
        result: Dict[str, Any] = {
            "text_intent": str(parsed.get("text_intent") or self.intent_text),
            "location_class": clean_class(parsed.get("location_class")),
            "device_class": clean_class(parsed.get("device_class")),
            **device_slots(parsed, getattr(self, "_devices_known", False)),
            "device_property": clean_nested(
                parsed.get("device_property"), "parent_class"),
            "environment_variable": clean_nested(
                parsed.get("environment_variable"), "measurement_quantity"),
            "reason": str(parsed.get("reason") or ""),
        }

        # A branch root is not an answer: it matches everything under it, so the
        # resolver would return a mixed bag of unrelated properties. Drop it and
        # let the request read as underdetermined, which it is.
        for slot in ("device_property", "environment_variable"):
            entry = result[slot]
            if entry and entry["class"] in BRANCH_ROOTS:
                self.logger.warning(
                    "Discarding branch root %s from %s -- too generic to identify "
                    "a property", entry["class"], slot,
                )
                result[slot] = None

        return result
