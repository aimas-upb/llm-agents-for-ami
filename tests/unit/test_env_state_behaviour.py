"""ENV_STATE and ENV_SNAPSHOT: two questions, two behaviours.

ENV_STATE asks for a named property of a named kind of device, and resolves
ontology classes to answer it. ENV_SNAPSHOT asks for everything and names
nothing. These pin that each answers its own question, and pin the snapshot's
shape -- consumers index `artifacts` by id and filter it by `workspace_id`, so
a list would silently give them nothing.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from ami_agents.agents.env_explorer.behaviors.environment_snapshot_behaviour import (
    EnvironmentSnapshotBehaviour,
    environment_snapshot,
)
from ami_agents.agents.env_explorer.behaviors.environment_state_behaviour import (
    CLASS_KEYS,
    EnvironmentStateBehaviour,
)
from ami_agents.agents.env_explorer.behaviors.value_retrieval_behaviour import (
    ValueRetrievalBehaviour,
)
from ami_agents.agents.env_explorer.utils.state_resolution import ResolvedAffordance
from ami_agents.shared.models.messages import MessageType


class _Logger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


def _agent(artifacts):
    return SimpleNamespace(artifacts=artifacts, environment_map={},
                           logger=_Logger())


def _artifact(name, workspace, state):
    return SimpleNamespace(name=name, workspace_id=workspace,
                           current_state=state)


@pytest.fixture
def agent():
    return _agent({
        "art/light": _artifact("light_1", "ws/kitchen", {"onOff": True}),
        "art/fan": _artifact("fan_1", "ws/kitchen", {"onOff": False,
                                                     "speed": 3}),
    })


class TestSnapshot:
    def test_artifacts_is_a_dict_keyed_by_id(self, agent):
        """The guard every consumer applies before using the payload."""
        snapshot = environment_snapshot(agent)
        assert isinstance(snapshot["artifacts"], dict)
        assert set(snapshot["artifacts"]) == {"art/light", "art/fan"}

    def test_each_entry_carries_name_workspace_and_state(self, agent):
        entry = environment_snapshot(agent)["artifacts"]["art/fan"]
        assert entry["name"] == "fan_1"
        assert entry["workspace_id"] == "ws/kitchen"
        assert entry["state"] == {"onOff": False, "speed": 3}

    def test_summary_counts_artifacts_and_properties(self, agent):
        summary = environment_snapshot(agent)["summary"]
        assert summary["total_artifacts"] == 2
        assert summary["total_properties"] == 3
        assert summary["workspaces"] == ["ws/kitchen"]

    def test_state_is_copied_not_shared(self, agent):
        """Mutating the response must not reach back into the agent."""
        snapshot = environment_snapshot(agent)
        snapshot["artifacts"]["art/light"]["state"]["onOff"] = "tampered"
        assert agent.artifacts["art/light"].current_state["onOff"] is True

    def test_empty_environment(self):
        snapshot = environment_snapshot(_agent({}))
        assert snapshot["artifacts"] == {}
        assert snapshot["summary"]["total_artifacts"] == 0


class TestClassKeys:
    def test_the_structuring_stage_slots(self):
        """The contract with the UA parser; drifting apart breaks dispatch."""
        assert set(CLASS_KEYS) == {
            "location_class", "device_class", "device_property",
            "environment_variable",
        }


class _Reply:
    """Captures what the behaviour would have sent."""

    def __init__(self):
        self.body = None
        self.thread = None
        self.metadata = {}

    def set_metadata(self, key, value):
        self.metadata[key] = value


class _Msg:
    def __init__(self, body, msg_type, reply):
        self.body = body
        self.thread = None
        self.sender = "test"
        self._type = msg_type
        self._reply = reply

    def get_metadata(self, key):
        return self._type if key == "type" else None

    def make_reply(self):
        return self._reply


def drive(behaviour, body, msg_type):
    """Run one behaviour tick with a canned message; return the reply body."""
    reply = _Reply()
    msg = _Msg(body, msg_type, reply)

    async def _receive(timeout=1):
        return msg

    async def _send(_):
        return None

    behaviour.receive = _receive
    behaviour.send = _send
    asyncio.run(behaviour.run())
    return json.loads(reply.body)


class TestTheTwoQuestionsStaySeparate:
    """ENV_STATE names a property. It never answers the other question."""

    def test_an_empty_state_request_is_an_error(self, agent):
        """Not a snapshot: that is ENV_SNAPSHOT_REQUEST, and the error says so."""
        beh = EnvironmentStateBehaviour()
        beh.agent = agent
        response = drive(beh, "{}", MessageType.ENV_STATE_REQUEST.value)

        assert response["error"] == "no_property_named"
        assert MessageType.ENV_SNAPSHOT_REQUEST.value in response["detail"]

    def test_a_snapshot_request_needs_no_classes(self, agent):
        """The other behaviour answers it, and names nothing to do so."""
        beh = EnvironmentSnapshotBehaviour()
        beh.agent = agent
        response = drive(beh, "{}", MessageType.ENV_SNAPSHOT_REQUEST.value)

        assert isinstance(response["artifacts"], dict)
        assert response["summary"]["total_artifacts"] == 2

    def test_a_state_request_naming_a_property_is_answered(self, agent,
                                                           monkeypatch):
        from ami_agents.agents.env_explorer.behaviors import (
            environment_state_behaviour as module)
        from ami_agents.agents.env_explorer.utils.state_resolution import (
            StateOutcome, StateResolution)

        monkeypatch.setattr(
            module, "resolve_payload",
            lambda a, p: StateResolution(outcome=StateOutcome.NONE,
                                         detail="nothing senses that"))
        beh = EnvironmentStateBehaviour()
        beh.agent = agent
        response = drive(
            beh,
            json.dumps({"location_class": "homeont:Kitchen",
                        "environment_variable": {"class": "homeont:Illuminance"}}),
            MessageType.ENV_STATE_REQUEST.value)

        assert "error" not in response
        assert response["outcome"] == "no_affordance"


def _affordance(target, name="onOff"):
    return ResolvedAffordance(
        artifact="art/x", artifact_name="x_1", artifact_type="homeont:Fan",
        workspace_name="Kitchen", affordance_name=name,
        affordance_type="homeont:FanOnOff", target=target,
    )


class _Response:
    def __init__(self, payload=None, fail=False):
        self._payload, self._fail = payload, fail

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    def raise_for_status(self):
        if self._fail:
            raise RuntimeError("HTTP 500")

    async def json(self, content_type=None):
        return self._payload


class _Session:
    """Stands in for aiohttp: one canned response per target."""

    def __init__(self, by_target):
        self.by_target = by_target
        self.requested = []

    def get(self, target):
        self.requested.append(target)
        return self.by_target[target]


async def _run(behaviour, session):
    """Drive the read loop without a SPADE runtime."""
    await asyncio.gather(*[
        behaviour._read(session, affordance)
        for affordance in behaviour.affordances
    ])
    behaviour.result = behaviour.affordances


class TestValueRetrieval:
    def test_a_scalar_value(self):
        affordance = _affordance("http://x/onOff")
        beh = ValueRetrievalBehaviour([affordance], logger=_Logger())
        session = _Session({"http://x/onOff": _Response(True)})
        asyncio.run(_run(beh, session))
        assert affordance.value is True
        assert affordance.has_value is True
        assert affordance.as_dict()["value"] is True

    def test_an_object_valued_property_stays_a_dictionary(self):
        """The whole reading, keys included -- not flattened or summarised."""
        channel = {"CallSign": "BBC1", "Name": "BBC One", "MajorNumber": 1}
        affordance = _affordance("http://x/currentChannel", "currentChannel")
        beh = ValueRetrievalBehaviour([affordance], logger=_Logger())
        session = _Session({"http://x/currentChannel": _Response(channel)})
        asyncio.run(_run(beh, session))
        assert affordance.value == channel

    def test_a_failed_read_keeps_the_resolution(self):
        """The caller still learns what would answer, and where."""
        affordance = _affordance("http://x/onOff")
        beh = ValueRetrievalBehaviour([affordance], logger=_Logger())
        session = _Session({"http://x/onOff": _Response(fail=True)})
        asyncio.run(_run(beh, session))
        assert affordance.has_value is False
        as_dict = affordance.as_dict()
        assert "value" not in as_dict
        assert "read failed" in as_dict["detail"]
        assert as_dict["target"] == "http://x/onOff"

    def test_one_failure_does_not_sink_the_others(self):
        good = _affordance("http://x/a")
        bad = _affordance("http://x/b")
        beh = ValueRetrievalBehaviour([good, bad], logger=_Logger())
        session = _Session({"http://x/a": _Response(42),
                            "http://x/b": _Response(fail=True)})
        asyncio.run(_run(beh, session))
        assert good.value == 42
        assert bad.has_value is False

    def test_every_affordance_is_read(self):
        affordances = [_affordance(f"http://x/{i}") for i in range(4)]
        beh = ValueRetrievalBehaviour(affordances, logger=_Logger())
        session = _Session({f"http://x/{i}": _Response(i) for i in range(4)})
        asyncio.run(_run(beh, session))
        assert len(session.requested) == 4
        assert [a.value for a in affordances] == [0, 1, 2, 3]

    def test_nothing_to_read(self):
        beh = ValueRetrievalBehaviour([], logger=_Logger())
        asyncio.run(beh.run())
        assert beh.result == []
        assert beh.error is None


class TestMismatchIsReadToo:
    """A mismatch is read like any other resolution.

    "Is anything still on in the kitchen" matches several different on/off
    properties. It is only answerable with their values, so the outcome
    describes what was found -- it does not decide whether to look.
    """

    def test_answer_by_class_reads_a_mismatched_resolution(self, monkeypatch):
        from ami_agents.agents.env_explorer.behaviors import (
            environment_state_behaviour as module)
        from ami_agents.agents.env_explorer.utils.state_resolution import (
            StateOutcome, StateResolution)

        affordances = [_affordance("http://x/a"), _affordance("http://x/b")]
        resolution = StateResolution(
            outcome=StateOutcome.MISMATCHED,
            affordances=affordances,
            property_classes=["homeont:FanOnOff", "homeont:OnOffLightOnOff"],
            detail="two types",
        )
        monkeypatch.setattr(module, "resolve_payload",
                            lambda agent, payload: resolution)

        read = []

        class _Retrieval:
            def __init__(self, affs, logger=None):
                self.affordances, self.error = affs, None

            async def join(self):
                for affordance in self.affordances:
                    affordance.value, affordance.has_value = True, True
                    read.append(affordance.target)

        monkeypatch.setattr(module, "ValueRetrievalBehaviour", _Retrieval)

        beh = EnvironmentStateBehaviour()
        beh.agent = SimpleNamespace(
            artifacts={}, environment_map={}, logger=_Logger(),
            add_behaviour=lambda b: None)

        response = asyncio.run(beh._answer_by_class(
            {"location_class": "homeont:Kitchen",
             "device_property": {"class": "homeont:OnOff"}}))

        assert response["outcome"] == "mismatched_affordance"
        assert read == ["http://x/a", "http://x/b"]
        assert all(entry["value"] is True for entry in response["affordances"])

    def test_no_affordance_reads_nothing(self, monkeypatch):
        from ami_agents.agents.env_explorer.behaviors import (
            environment_state_behaviour as module)
        from ami_agents.agents.env_explorer.utils.state_resolution import (
            StateOutcome, StateResolution)

        monkeypatch.setattr(
            module, "resolve_payload",
            lambda agent, payload: StateResolution(
                outcome=StateOutcome.NONE, detail="nothing senses that"))

        def _fail(*a, **k):
            raise AssertionError("must not read when nothing resolved")

        monkeypatch.setattr(module, "ValueRetrievalBehaviour", _fail)

        beh = EnvironmentStateBehaviour()
        beh.agent = SimpleNamespace(artifacts={}, environment_map={},
                                    logger=_Logger())
        response = asyncio.run(beh._answer_by_class({"text_intent": "x"}))
        assert response["outcome"] == "no_affordance"
        assert response["affordances"] == []
