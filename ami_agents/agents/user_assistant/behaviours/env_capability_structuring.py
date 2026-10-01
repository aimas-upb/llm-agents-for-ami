"""Behaviour: structure one ENV_CAPABILITIES_REQUEST into ontology class identifiers.

Stage 2 of the request pipeline. The segmenter has already decided this is a
capability question; this turns the natural-language fragment into the classes
EnvExplorer resolves against the TD graph -- a location, a device kind, a device
property or an environment variable, a command -- plus the performative that
says whether a yes/no or a list is wanted.

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
from ..prompts.intent_parsing_prompts import ENV_CAPABILITIES_REQUEST_PARSER_PROMPT
from ..utils import loose_json_loads
from ..utils.intent_slots import BRANCH_ROOTS, clean_class, clean_nested
from ..utils.llm_client import build_behaviour_llm_client, build_llm_call_kwargs
from ..utils.ontology_context import get_capability_ontology_context_json

PERFORMATIVES = ("query", "query_if")
DEFAULT_PERFORMATIVE = "query"

# The command tree's root. Too generic on its own, but meaningful beside an
# environment variable: "can anything cool the kitchen" asks for any action
# with an effect on that variable.
COMMAND_ROOT = "saref:Command"

SLOTS = ("location_class", "device_class", "device_property",
         "environment_variable", "command")


class EnvCapabilityStructuringBehaviour(OneShotBehaviour):
    """Parse ONE ENV_CAPABILITIES_REQUEST intent into ontology class identifiers."""

    def __init__(self, intent_text: str, logger=None) -> None:
        super().__init__()
        self.intent_text = intent_text
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")
        # Populated by ``run`` -- a structured dict, or empty slots with
        # ``error`` set.
        self.result: Dict[str, Any] = {}
        self.error: Optional[str] = None

    async def run(self) -> None:
        self.logger.info(demo(
            f"[LLM CALL] Structuring ENV_CAPABILITIES_REQUEST: {self.intent_text[:80]!r}"
        ))
        prompt = ENV_CAPABILITIES_REQUEST_PARSER_PROMPT.format(
            ontology_context=get_capability_ontology_context_json()
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
            self.result = normalise_capability_slots(
                parsed, self.intent_text, self.logger)
            self.logger.info(demo(
                f"[ENV_CAPABILITIES_REQUEST structured]\n"
                f"{json.dumps(self.result, indent=2)}"
            ))
        except Exception as exc:
            self.error = str(exc)
            self.logger.warning("ENV_CAPABILITIES structuring failed: %s", exc)
            # Enough for the caller to report the failure against the right
            # intent; every class slot stays absent.
            self.result = {
                "text_intent": self.intent_text,
                "request_performative": DEFAULT_PERFORMATIVE,
                **{slot: None for slot in SLOTS},
            }


def normalise_capability_slots(parsed: Dict[str, Any], intent_text: str,
                               logger=None) -> Dict[str, Any]:
    """Coerce the LLM's object into the shape callers may rely on.

    A plain function so it is testable without an LLM or a SPADE agent.
    """
    performative = clean_class(parsed.get("request_performative"))
    if performative not in PERFORMATIVES:
        performative = DEFAULT_PERFORMATIVE

    result: Dict[str, Any] = {
        "text_intent": str(parsed.get("text_intent") or intent_text),
        "request_performative": performative,
        "location_class": clean_class(parsed.get("location_class")),
        "device_class": clean_class(parsed.get("device_class")),
        "device_property": clean_nested(parsed.get("device_property"), "parent_class"),
        "environment_variable": clean_nested(
            parsed.get("environment_variable"), "measurement_quantity"),
        "command": clean_nested(parsed.get("command"), "parent_class"),
        "reason": str(parsed.get("reason") or ""),
    }

    # A branch root is not an answer: it matches everything under it, so the
    # resolver would return a mixed bag of unrelated properties. Drop it and let
    # the request read as underdetermined, which it is.
    for slot in ("device_property", "environment_variable"):
        entry = result[slot]
        if entry and entry["class"] in BRANCH_ROOTS:
            _warn(logger, "Discarding branch root %s from %s", entry["class"], slot)
            result[slot] = None

    # The two are mutually exclusive; keep the device's own property, which is
    # the narrower reading, if the parser filled both.
    if result["device_property"] and result["environment_variable"]:
        _warn(logger, "Both device_property and environment_variable set; "
              "keeping device_property")
        result["environment_variable"] = None

    command = result["command"]
    if command and command["class"] == COMMAND_ROOT and not result["environment_variable"]:
        _warn(logger, "Discarding %s: only meaningful with an environment "
              "variable", COMMAND_ROOT)
        result["command"] = None

    return result


def _warn(logger, message: str, *args: Any) -> None:
    if logger is not None:
        logger.warning(message, *args)
