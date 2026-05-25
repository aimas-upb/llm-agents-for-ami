from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
from ami_agents.agents.user_assistant.models import Intent


def test_infers_air_quality_from_stuffy_air_phrase():
    intents = [
        Intent(
            action="unknown",
            artifact="unknown",
            intent_text="the air is so stuffy in here",
        )
    ]

    inferred = InteractionSolverAgent._infer_observable_properties_from_intents(intents)

    assert "air_quality" in inferred


def test_infers_air_quality_from_ventilation_and_fresh_air_phrases():
    intents = [
        Intent(action="open", artifact="window", intent_text="please improve ventilation"),
        Intent(action="unknown", artifact="unknown", intent_text="i need some fresh air"),
    ]

    inferred = InteractionSolverAgent._infer_observable_properties_from_intents(intents)

    assert "air_quality" in inferred
