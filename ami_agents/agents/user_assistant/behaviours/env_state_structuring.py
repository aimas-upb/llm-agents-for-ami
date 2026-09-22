"""Behaviour: structure one ENV_STATE_REQUEST into ontology class identifiers.

Stage 2 of the request pipeline. The segmenter has already decided this is a
state request; this turns the natural-language fragment into the classes a
SPARQL query over the TD graph is built from -- a location, a device kind, and
either a device property or an environment variable.

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
from ..prompts.intent_parsing_prompts import ENV_STATE_REQUEST_PARSER_PROMPT
from ..utils import loose_json_loads
from ..utils.llm_client import build_behaviour_llm_client, build_llm_call_kwargs
from ..utils.ontology_context import get_capabilities_context_json

# The parser must answer with a class specific enough to identify something.
# These are the taxonomy's roots: they match every property beneath them, so an
# answer naming one resolves to everything and therefore to nothing.
BRANCH_ROOTS = frozenset({
    "homeont:ActuatableDeviceProperty",
    "homeont:DeviceStateProperty",
    "homeont:DeviceCapabilityProperty",
    "sosa:ObservableProperty",
})


def _clean(value: Any) -> Optional[str]:
    """A class identifier, or None.

    Older prompts used "NA" as a sentinel; treat any such leftover as absent so
    a caller never has to test for a magic string.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NONE", "NULL", "UNKNOWN"}:
        return None
    return text


def _clean_nested(value: Any, *keys: str) -> Optional[Dict[str, str]]:
    """A `{class, ...}` object, or None if it carries no usable class."""
    if not isinstance(value, dict):
        return None
    cls = _clean(value.get("class"))
    if cls is None:
        return None
    out: Dict[str, str] = {"class": cls}
    for key in keys:
        extra = _clean(value.get(key))
        if extra is not None:
            out[key] = extra
    return out


class EnvStateStructuringBehaviour(OneShotBehaviour):
    """Parse ONE ENV_STATE_REQUEST intent into ontology class identifiers."""

    def __init__(self, intent_text: str, logger=None) -> None:
        super().__init__()
        self.intent_text = intent_text
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")
        # Populated by ``run`` -- a structured dict, or {} with ``error`` set.
        self.result: Dict[str, Any] = {}
        self.error: Optional[str] = None

    async def run(self) -> None:
        self.logger.info(demo(
            f"[LLM CALL] Structuring ENV_STATE_REQUEST: {self.intent_text[:80]!r}"
        ))
        prompt = ENV_STATE_REQUEST_PARSER_PROMPT.format(
            capabilities_context=get_capabilities_context_json()
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
            "location_class": _clean(parsed.get("location_class")),
            "device_class": _clean(parsed.get("device_class")),
            "device_property": _clean_nested(
                parsed.get("device_property"), "parent_class"),
            "environment_variable": _clean_nested(
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
