"""A window covering's position says which way its scale runs.

Matter's lift percentage counts closure (0 fully open, 100 fully closed), but
the number alone does not say so, and planners guessed: "fully closed" was
sent as 0 in the gpt-6-luna SimuHome runs. The TD now states it in the
device's own value schema (`js:description`), from `SCALE_MEANINGS`:

    blinds   position / targetPosition (action input and property output)
    schema   "Percent closed: 0 is fully open, 100 is fully closed."

It is on the instance's schema, not the homeont class, because another
deployment may count the other way (Home Assistant's cover position is
percent open).
"""

import json
from pathlib import Path

import pytest

from ami_agents.shared.utils.namespaces import JS, TD

REPO = Path(__file__).resolve().parents[2]
EPISODE = REPO.parent / "SimuHome" / "data" / "benchmark" / "qt3_feasible_seed_26.json"
BLINDS = "living_room_window_covering_controller_1"
MEANING = "Percent closed: 0 is fully open, 100 is fully closed."


@pytest.fixture(scope="module")
def graph():
    """The living-room blinds' TD, built offline from a benchmark home."""
    from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD
    if not EPISODE.exists():
        pytest.skip(f"SimuHome benchmark not found at {EPISODE}")
    config = json.loads(EPISODE.read_text())["initial_home_config"]
    return SimuHomeTD("http://localhost:8097", "test", config).artifact("living_room", BLINDS)


def _descriptions(graph, title):
    """The `js:description` of every value schema of the affordances titled
    ``title``: action inputs and property outputs, unwrapped from an object
    schema when the input is one."""
    out = []
    # 1. Each affordance with that title, and its input and output schemas.
    for affordance in graph.subjects(TD.title, None):
        if str(graph.value(affordance, TD.title)) != title:
            continue
        for predicate in (TD.hasInputSchema, TD.hasOutputSchema):
            schema = graph.value(affordance, predicate)
            if schema is None:
                continue
            # 2. A direct-value input may be wrapped in an object schema.
            for member in list(graph.objects(schema, JS.properties)) or [schema]:
                out.append(str(graph.value(member, JS.description) or ""))
    return out


class TestCoveringScale:
    @pytest.mark.parametrize("title", ["position", "targetPosition"])
    def test_every_position_schema_states_the_direction(self, graph, title):
        """Both position affordances, read and write, say 100 is closed."""
        descriptions = _descriptions(graph, title)
        assert descriptions, f"no {title} schemas on {BLINDS}"
        assert all(MEANING in d for d in descriptions)

    def test_other_numbers_get_no_direction(self, graph):
        """The sentence is the covering's: no other schema of the device has it."""
        # 1. Every titled affordance other than the two positions.
        titles = {str(graph.value(a, TD.title)) for a in graph.subjects(TD.title, None)}
        others = titles - {"position", "targetPosition"}
        # 2. None of their schemas carries the covering's sentence.
        assert not any(MEANING in d for t in others for d in _descriptions(graph, t))
