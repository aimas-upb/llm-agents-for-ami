"""What the phrasing model is shown, and what happens when it cannot be reached.

The API is never called here. These pin the payload -- labels rather than
identifiers, values intact -- and the dispatch that decides whether a call is
made at all.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from ami_agents.agents.user_assistant.behaviours.state_answer import (
    StateAnswerBehaviour,
    readings_for_prompt,
)


def aff(device, prop, value, room="Kitchen", name="device_1"):
    return {"artifact_name": name, "artifact_type": device,
            "affordance_type": prop, "workspace_name": room, "value": value,
            "artifact": "http://host/workspaces/test/kitchen/artifacts/device_1#artifact",
            "target": "http://host/.../properties/onOff",
            "parameter_name": None}


class TestPromptInput:
    def test_classes_arrive_as_labels(self):
        reading = readings_for_prompt([
            aff("homeont:OnOffLight", "homeont:OnOffLightOnOff", True)])[0]
        assert reading["device"] == "On/Off Light"
        assert reading["property"] == "On/Off Light On/Off"
        assert reading["room"] == "Kitchen"

    def test_no_identifier_or_url_is_shown_to_the_model(self):
        """What reaches the prompt can reach the answer."""
        payload = json.dumps(readings_for_prompt([
            aff("homeont:Tv", "homeont:ChannelCurrentChannel", {"Name": "BBC One"})]))
        assert "homeont:" not in payload
        assert "http" not in payload
        assert "#artifact" not in payload

    def test_a_dictionary_value_survives_intact(self):
        """Choosing one of its fields is the reason for the call."""
        channel = {"CallSign": "BBC1", "Name": "BBC One", "MajorNumber": 1}
        reading = readings_for_prompt([
            aff("homeont:Tv", "homeont:ChannelCurrentChannel", channel)])[0]
        assert reading["value"] == channel

    def test_one_entry_per_affordance(self):
        readings = readings_for_prompt([
            aff("homeont:Fan", "homeont:FanOnOff", True),
            aff("homeont:AirPurifier", "homeont:AirPurifierOnOff", False)])
        assert [r["device"] for r in readings] == ["Fan", "Air Purifier"]

    def test_a_failed_read_is_passed_on_as_a_note(self):
        affordance = aff("homeont:Fan", "homeont:FanOnOff", None)
        affordance["detail"] = "read failed: timeout"
        assert readings_for_prompt([affordance])[0]["note"] == "read failed: timeout"

    def test_an_unnamed_device_falls_back_to_its_instance_name(self):
        affordance = aff(None, "homeont:OnOff", True, name="odd_device_3")
        assert readings_for_prompt([affordance])[0]["device"] == "odd_device_3"


class _Logger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


class TestFailureIsNotFatal:
    def test_an_unreachable_model_sets_error_not_an_exception(self, monkeypatch):
        """A failed call must leave `error` set, never raise into the caller."""
        from ami_agents.agents.user_assistant.behaviours import state_answer

        def _boom(config, key):
            raise RuntimeError("no client")

        monkeypatch.setattr(state_answer, "build_behaviour_llm_client", _boom)

        beh = StateAnswerBehaviour(
            "is anything on", "mismatched_affordance",
            [aff("homeont:Fan", "homeont:FanOnOff", True)], logger=_Logger())
        beh.agent = SimpleNamespace(config={})
        asyncio.run(beh.run())
        assert beh.result is None
        assert "no client" in beh.error

    def test_a_malformed_reply_is_a_failure_not_an_answer(self, monkeypatch):
        """Prose where JSON was asked for must fall back, not reach the user."""
        from ami_agents.agents.user_assistant.behaviours import state_answer

        class _Message:
            content = "sure, the fan is on"

        class _Completions:
            async def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=_Message())])

        monkeypatch.setattr(
            state_answer, "build_behaviour_llm_client",
            lambda config, key: SimpleNamespace(
                client=SimpleNamespace(
                    chat=SimpleNamespace(completions=_Completions())),
                model="gpt-5-nano"))
        monkeypatch.setattr(state_answer, "build_llm_call_kwargs", lambda cfg: {})

        beh = StateAnswerBehaviour(
            "is anything on", "mismatched_affordance",
            [aff("homeont:Fan", "homeont:FanOnOff", True)], logger=_Logger())
        beh.agent = SimpleNamespace(config={})
        asyncio.run(beh.run())
        assert beh.result is None
        assert beh.error

    def test_a_well_formed_reply_is_the_answer(self, monkeypatch):
        from ami_agents.agents.user_assistant.behaviours import state_answer

        class _Message:
            content = '{"answer": "Yes - the Fan is on."}'

        class _Completions:
            async def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=_Message())])

        monkeypatch.setattr(
            state_answer, "build_behaviour_llm_client",
            lambda config, key: SimpleNamespace(
                client=SimpleNamespace(
                    chat=SimpleNamespace(completions=_Completions())),
                model="gpt-5-nano"))
        monkeypatch.setattr(state_answer, "build_llm_call_kwargs", lambda cfg: {})

        beh = StateAnswerBehaviour(
            "is anything on", "mismatched_affordance",
            [aff("homeont:Fan", "homeont:FanOnOff", True)], logger=_Logger())
        beh.agent = SimpleNamespace(config={})
        asyncio.run(beh.run())
        assert beh.result == "Yes - the Fan is on."
        assert beh.error is None


class TestDispatch:
    """Which branch `_phrase` takes, without touching the network."""

    @staticmethod
    def _behaviour():
        return SimpleNamespace(logger=_Logger(), agent=SimpleNamespace())

    def test_no_affordance_is_answered_without_a_model(self):
        from ami_agents.agents.user_assistant import queries

        response = {"outcome": "no_affordance", "affordances": [],
                    "query": {"location_class": "homeont:Kitchen",
                              "property_class": "homeont:Illuminance"}}
        answer = asyncio.run(queries._phrase(self._behaviour(), response, "x"))
        assert "Illuminance" in answer

    def test_a_basic_resolution_is_answered_without_a_model(self):
        from ami_agents.agents.user_assistant import queries

        response = {"outcome": "resolved_affordance", "affordances": [
            aff("homeont:Freezer", "homeont:CompartmentTemperature", -18.0)]}
        answer = asyncio.run(queries._phrase(self._behaviour(), response, "x"))
        assert "-18.0" in answer

    def test_an_error_response_is_reported(self):
        from ami_agents.agents.user_assistant import queries

        response = {"error": "state_resolution_failed", "detail": "boom"}
        answer = asyncio.run(queries._phrase(self._behaviour(), response, "x"))
        assert "boom" in answer
