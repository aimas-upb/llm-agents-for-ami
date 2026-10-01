"""Where an artifact's state is read from: the TD's names, then the alias.

HASP names one state property per entity (`sensorState`, `coverState`), so
consumers take the URLs EnvExplorer's snapshot carries rather than building
`{artifact}/properties/state`.
"""

from types import SimpleNamespace

from ami_agents.agents.env_explorer.behaviors.environment_snapshot_behaviour import (
    environment_snapshot,
    state_property_urls,
)
from ami_agents.shared.models.environment import AffordanceType
from ami_agents.shared.utils.state_urls import state_urls

ART = "http://hasp/workspaces/lab/artifacts/temp#artifact"


class _Logger:
    def info(self, *a, **k): pass


def _affordance(name, kind=AffordanceType.PROPERTY, href=None):
    return SimpleNamespace(name=name, affordance_type=kind, artifact_id=ART,
                           form=SimpleNamespace(href=href or f"http://hasp/x/{name}"))


def _agent(affordances):
    engine = SimpleNamespace(get_affordances_for_artifact=lambda aid: affordances)
    artifact = SimpleNamespace(name="temp", workspace_id="lab", current_state={"v": 1})
    return SimpleNamespace(artifacts={ART: artifact}, integration_engine=engine,
                           logger=_Logger())


class TestFromTheTD:
    def test_state_named_properties_only(self):
        agent = _agent([_affordance("sensorState"), _affordance("brightness"),
                        _affordance("turnOn", AffordanceType.ACTION)])
        assert state_property_urls(agent, ART) == ["http://hasp/x/sensorState"]

    def test_legacy_name_is_recognised(self):
        assert state_property_urls(_agent([_affordance("state")]), ART) == ["http://hasp/x/state"]

    def test_no_engine_means_none(self):
        agent = SimpleNamespace(artifacts={})
        assert state_property_urls(agent, ART) == []

    def test_the_snapshot_carries_them(self):
        snapshot = environment_snapshot(_agent([_affordance("sensorState")]))
        assert snapshot["artifacts"][ART]["state_property_urls"] == ["http://hasp/x/sensorState"]


class TestConsumers:
    def test_the_snapshot_urls_win(self):
        assert state_urls(ART, {"state_property_urls": ["http://hasp/x/sensorState"]}) == [
            "http://hasp/x/sensorState"]

    def test_falls_back_to_the_alias(self):
        assert state_urls(ART, {}) == [
            "http://hasp/workspaces/lab/artifacts/temp/properties/state"]

    def test_nothing_for_an_empty_id(self):
        assert state_urls("", {}) == []
