"""Typed intent dataclasses for goal requests.

Represents the structured output of per-span intent parsing via the three
LLM prompts: GOAL_REQUEST_PARSER_PROMPT, ENV_STATE_REQUEST_PARSER_PROMPT,
and ENV_CAPABILITIES_REQUEST_PARSER_PROMPT.
"""

from dataclasses import dataclass
from typing import Optional, Union


def _na_to_none(v: Optional[str]) -> Optional[str]:
    """Normalize the LLM's 'NA' sentinel to None."""
    return None if v in (None, "NA") else v


@dataclass
class Intent:
    """Generic intent carrying only the user's verbatim text.

    The fallback when a request arrives as bare text and no structure has been
    parsed out of it -- see ``goal_request.py``, which builds one of these from
    a plain string. Both the UserAssistant and the InteractionSolver use it, so
    it lives here alongside the structured intents rather than in either agent.

    For goal requests with parsed structure, use ImplicitGoalIntent or
    ExplicitGoalIntent instead.
    """

    intent_text: str

    def to_query_string(self) -> str:
        """Return the verbatim user text as the query key."""
        return self.intent_text


@dataclass
class GoalAction:
    """Structured action specification from GOAL_REQUEST_PARSER_PROMPT."""

    affordance_type: Optional[str]  # ontology class (e.g. "ex:TurnOnCommand") or None
    parameter: Optional[str]        # payload param name (e.g. "brightness") or None
    value: Optional[str]            # target/delta value as string or None
    verb: str                       # "set" | "modify"


@dataclass
class GoalTarget:
    """Structured target specification from GOAL_REQUEST_PARSER_PROMPT."""

    artifact_name: Optional[str]    # e.g. "light308" or None
    artifact_type: Optional[str]    # ontology class (e.g. "ex:Light") or None
    workspace_type: Optional[str]   # ontology class (e.g. "ex:Kitchen") or None
    workspace_name: Optional[str]   # e.g. "Lab 308" or None


@dataclass
class ImplicitGoalIntent:
    """Parser output when a goal cannot be resolved to explicit action+target.

    Implicit goals are candidates for signifier-based experience retrieval:
    the system must infer the right device from context.

    The subtype indicates the reason for implicitness:
    - conditioned_actions: action/target clear but tied to conditions/constraints
    - composite_actions: multiple actions joined by sequencing
    - implicit_intent: vague/underspecified without resolvable device/command
    """

    text_intent: str                # verbatim user text for this intent
    reason: str                     # LLM justification for implicit classification
    subtype: str = "implicit_intent"  # "conditioned_actions" | "composite_actions" | "implicit_intent"
    affected_env_vars: Optional[list] = None  # [{"variable": str, "direction": str}, ...] inferred by LLM

    def to_query_string(self) -> str:
        """Return the verbatim user text as the query key."""
        return self.text_intent

    def to_wire_dict(self) -> dict:
        """Serialize to the wire format for inter-agent communication."""
        return {
            "category": "implicit",
            "subtype": self.subtype,
            "text_intent": self.text_intent,
            "reason": self.reason,
            "affected_env_vars": self.affected_env_vars,
        }

    @property
    def intent_type(self) -> str:
        """Intent type for routing and signifier matching."""
        return "implicit"


@dataclass
class ExplicitGoalIntent:
    """Parser output when a goal is directly resolvable to action+target.

    Explicit goals have full specification from the user and do NOT produce
    signifiers — there is no context-dependent experience to store.
    """

    text_intent: str                # verbatim user text for this intent
    action: GoalAction              # structured action specification
    target: GoalTarget              # structured target specification

    def to_query_string(self) -> str:
        """Return the verbatim user text as the query key."""
        return self.text_intent

    def to_wire_dict(self) -> dict:
        """Serialize to the wire format for inter-agent communication."""
        return {
            "category": "explicit",
            "text_intent": self.text_intent,
            "action": {
                "affordance_type": self.action.affordance_type,
                "parameter": self.action.parameter,
                "value": self.action.value,
                "verb": self.action.verb,
            },
            "target": {
                "artifact_name": self.target.artifact_name,
                "artifact_type": self.target.artifact_type,
                "workspace_type": self.target.workspace_type,
                "workspace_name": self.target.workspace_name,
            },
        }

    @property
    def intent_type(self) -> str:
        """Intent type for routing and signifier matching."""
        return "explicit"


def goal_intent_from_dict(data: dict) -> Union[ImplicitGoalIntent, ExplicitGoalIntent]:
    """Deserialize LLM parser output into a typed goal intent.

    Normalizes the "NA" sentinel to None for Optional fields.
    """
    if data.get("category") == "explicit":
        raw_action = data.get("action", {})
        raw_target = data.get("target", {})
        return ExplicitGoalIntent(
            text_intent=data.get("text_intent", ""),
            action=GoalAction(
                affordance_type=_na_to_none(raw_action.get("affordance_type")),
                parameter=raw_action.get("parameter"),  # already null from LLM
                value=raw_action.get("value"),          # already null from LLM
                verb=raw_action.get("verb", "set"),
            ),
            target=GoalTarget(
                artifact_name=_na_to_none(raw_target.get("artifact_name")),
                artifact_type=_na_to_none(raw_target.get("artifact_type")),
                workspace_type=_na_to_none(raw_target.get("workspace_type")),
                workspace_name=_na_to_none(raw_target.get("workspace_name")),
            ),
        )
    return ImplicitGoalIntent(
        text_intent=data.get("text_intent", ""),
        reason=data.get("reason", ""),
        subtype=data.get("subtype", "implicit_intent"),
        affected_env_vars=data.get("affected_env_vars"),
    )
