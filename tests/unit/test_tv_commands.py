"""A SimuHome TV can be tuned and played, not only switched on and dimmed.

Matter drives a TV's channel and playback by commands (ChangeChannel,
ChangeChannelByNumber; Play, Pause, Stop) that write read-only attributes. With
those links in `command_targets.yaml`, both attributes classify as actuatable,
so the TD offers actions for them, and SHTD translates each value:

    currentChannel  homeont:SetModeCommand          "MBC" / {MajorNumber: 3} / "3.0"
                                                    -> write CurrentChannel = the MBC entry
    playbackState   homeont:SetPlaybackStateCommand 0 / "Playing"   -> Play
                                                    1 / "paused"    -> Pause
                                                    2               -> Stop

The channel is written, not commanded: SimuHome's ChangeChannel fails on every
home loaded from an episode (its ChannelList entries stay dicts), so SHTD
writes the entry the command would have selected (`shtd._channel_write`).

The channel's schema lists the TV's own channels ("MBC = 3.0"), so a request
naming a channel can be planned. A plan's action node sends an object input
as named fields, which SHTD and the bench's scheduler take as the value.
"""

import json
import sys
from pathlib import Path

import pytest
from rdflib.namespace import RDF

from ami_agents.shared.utils.namespaces import HOMEONT, JS, TD

REPO = Path(__file__).resolve().parents[1].parent
SIMUHOME = REPO / "ami_agents" / "environment" / "integration" / "SimuHome"
EPISODE = REPO.parent / "SimuHome" / "data" / "benchmark" / "qt4-2_feasible_seed_29.json"
TV = "living_room_tv_1"


@pytest.fixture(scope="module")
def td():
    """The benchmark home's TD builder (offline: no simulator)."""
    if not EPISODE.exists():
        pytest.skip(f"SimuHome benchmark not found at {EPISODE}")
    # SHTD imports its siblings as top-level modules.
    if str(SIMUHOME) not in sys.path:
        sys.path.insert(0, str(SIMUHOME))
    from ami_agents.environment.integration.SimuHome.td_builder import SimuHomeTD
    config = json.loads(EPISODE.read_text())["initial_home_config"]
    return SimuHomeTD("http://localhost:8097", "test", config)


def _action(graph, title):
    """The action affordance titled ``title``, or None."""
    return next((a for a in graph.subjects(RDF.type, TD.ActionAffordance)
                 if str(graph.value(a, TD.title)) == title), None)


# -- the TD ---------------------------------------------------------------------

class TestTvDescription:
    def test_channel_and_playback_are_actions_with_their_command_classes(self, td):
        """Both are actions now, typed by what they do."""
        graph = td.artifact("living_room", TV)
        channel, playback = _action(graph, "currentChannel"), _action(graph, "playbackState")
        assert (channel, RDF.type, HOMEONT.SetModeCommand) in graph
        assert (playback, RDF.type, HOMEONT.SetPlaybackStateCommand) in graph

    def test_the_channel_input_lists_this_tvs_channels(self, td):
        """The input says which number each channel name is."""
        graph = td.artifact("living_room", TV)
        schema = graph.value(_action(graph, "currentChannel"), TD.hasInputSchema)
        description = str(graph.value(schema, JS.description))
        assert "Channels on this device:" in description
        assert "MBC = 3.0" in description and "KBS1 = 1.0" in description


# -- SHTD's dispatch ---------------------------------------------------------------

class TestTvDispatch:
    @pytest.mark.parametrize("value", [
        "MBC", "mbc", {"Name": "MBC"}, {"MajorNumber": 3, "MinorNumber": 0}, "3.0", 3])
    def test_a_channel_by_name_or_number_is_written_whole(self, td, value):
        """Any way of naming MBC writes the TV's own MBC entry to CurrentChannel."""
        from ami_agents.environment.integration.SimuHome import shtd
        call = shtd._dispatch_call(td, "living_room", TV, "currentChannel", value)
        # 1. A write, not SimuHome's failing ChangeChannel command.
        assert call["tool"] == "write_attribute"
        args = call["args"]
        assert (args["cluster_id"], args["attribute_id"]) == ("Channel", "CurrentChannel")
        # 2. The whole entry, as the evaluator reads it (MajorNumber 3 is MBC).
        assert args["value"]["Name"] == "MBC" and args["value"]["MajorNumber"] == 3

    def test_a_channel_the_tv_does_not_have_is_a_400_naming_its_channels(self, td):
        """An unknown channel is refused, and the refusal lists what there is."""
        from fastapi import HTTPException
        from ami_agents.environment.integration.SimuHome import shtd
        with pytest.raises(HTTPException) as caught:
            shtd._dispatch_call(td, "living_room", TV, "currentChannel", "CNN")
        assert caught.value.status_code == 400 and "MBC (3.0)" in caught.value.detail

    @pytest.mark.parametrize("value, command", [
        (0, "Play"), ("Playing", "Play"), (1, "Pause"), ("paused", "Pause"),
        (2, "Stop"), ("not_playing", "Stop")])
    def test_playback_by_the_state_to_reach(self, td, value, command):
        """The enum number or its name picks the command; no arguments."""
        from ami_agents.environment.integration.SimuHome import shtd
        call = shtd._dispatch_call(td, "living_room", TV, "playbackState", value)
        assert (call["args"]["command_id"], call["args"]["args"]) == (command, {})

    def test_buffering_is_not_commanded(self, td):
        """Buffering is reported, never reached by a command: a 400."""
        from fastapi import HTTPException
        from ami_agents.environment.integration.SimuHome import shtd
        with pytest.raises(HTTPException):
            shtd._dispatch_call(td, "living_room", TV, "playbackState", 3)


# -- an object input sent as named fields -----------------------------------------

class TestObjectInputs:
    def test_the_scheduler_takes_named_fields_as_the_value(self):
        """A plan's action node posts an object input as its fields; the
        scheduler passes them on whole, and a `value` still wins."""
        from tests.simuhome.bench.scheduling import action_value
        assert action_value({"MajorNumber": 3, "MinorNumber": 0}) == \
            {"MajorNumber": 3, "MinorNumber": 0}
        assert action_value({"value": 2}) == 2
        assert action_value({}) is None and action_value(None) is None


# -- a structure in place of a value -------------------------------------------------

class TestStructuredValues:
    def test_a_schema_shaped_value_is_refused_not_read_as_off(self, td):
        """Read as a value, {"type": "boolean", "value": true} would send Off."""
        from fastapi import HTTPException
        from ami_agents.environment.integration.SimuHome import shtd
        with pytest.raises(HTTPException) as caught:
            shtd._dispatch_call(td, "living_room", TV, "onOff",
                                {"type": "boolean", "value": True})
        assert caught.value.status_code == 400 and "single value" in caught.value.detail

    def test_a_channel_still_takes_its_named_fields(self, td):
        """The one action whose input is an object."""
        from ami_agents.environment.integration.SimuHome import shtd
        call = shtd._dispatch_call(td, "living_room", TV, "currentChannel",
                                   {"MajorNumber": 3, "MinorNumber": 0})
        assert call["args"]["value"]["Name"] == "MBC"
