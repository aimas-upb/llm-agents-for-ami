from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
from ami_agents.agents.user_assistant.models import Intent


def test_infer_observable_properties_accepts_serialized_structured_intents():
    properties = InteractionSolverAgent._infer_observable_properties_from_intents(
        [
            {
                "action": "modify",
                "artifact": "light",
                "parameter": "brightness",
                "intent_text": "reduce glare",
            }
        ]
    )

    assert "glare" in properties
    assert "luminosity" in properties


def test_infer_observable_properties_still_accepts_intent_objects():
    properties = InteractionSolverAgent._infer_observable_properties_from_intents(
        [
            Intent(
                action="set",
                artifact="window",
                parameter="open_close",
                value=True,
                intent_text="the air is stuffy in here",
            )
        ]
    )

    assert "air_quality" in properties
