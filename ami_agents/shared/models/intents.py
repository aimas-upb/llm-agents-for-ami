"""The plain-text intent: a request known only by its words.

Structured goals are `goal_structure.GoalStructure`. `Intent` is what is left
for a request that arrives as bare text, or that is handed to the general
planning workflow, which plans from the words (`goal_request.py` builds one
from a plain string, or from a structure's text). Both the UserAssistant and
the InteractionSolver use it, so it lives here rather than in either agent.
"""

from dataclasses import dataclass


@dataclass
class Intent:
    """Generic intent carrying only the user's verbatim text."""

    intent_text: str

    def to_query_string(self) -> str:
        """Return the verbatim user text as the query key."""
        return self.intent_text

    def to_wire_dict(self) -> dict:
        """Serialize for an envelope; the words are all there is."""
        return {"text_intent": self.intent_text}
