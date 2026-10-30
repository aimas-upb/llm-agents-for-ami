"""A device called by name in a state or capability question.

The goals' rule (a name fits only the device that has it) for questions: the
parser says how the device is picked out and which of the home's devices has
the name; a named device the home lacks is listed against, never read in its
place (`ami_agents/agents/user_assistant/utils/device_names.py`).
"""

from ami_agents.agents.user_assistant.utils.device_names import (
    device_slots,
    missing_device_answer,
    named_device_missing,
    state_lookup,
)

LIGHT_1 = {"device_class": "homeont:OnOffLight", "location_class": "homeont:Bedroom",
           "device_property": {"class": "homeont:OnOffState"},
           "environment_variable": None}


class TestSlots:
    def test_a_named_device_keeps_its_name(self):
        assert device_slots({"device_reference": "name", "artifact_name": "bedroom_light_1"},
                            devices_known=True) == {
            "device_reference": "name", "artifact_name": "bedroom_light_1"}

    def test_a_described_device_has_no_name(self):
        """A name given for a description is dropped: "the light" names none."""
        assert device_slots({"device_reference": "description", "artifact_name": "x"},
                            devices_known=True)["artifact_name"] is None

    def test_without_the_homes_devices_nothing_is_kept(self):
        """The parser could not have found a name; "missing" would be a guess."""
        assert device_slots({"device_reference": "name", "artifact_name": None},
                            devices_known=False) == {
            "device_reference": None, "artifact_name": None}

    def test_an_unknown_reference_is_dropped(self):
        assert device_slots({"device_reference": "label"}, True)["device_reference"] is None


class TestMissing:
    def test_a_named_device_with_no_match_is_missing(self):
        assert named_device_missing({**LIGHT_1, "device_reference": "name"})
        assert not named_device_missing({**LIGHT_1, "device_reference": "name",
                                         "artifact_name": "bedroom_on_off_light_1"})
        assert not named_device_missing({**LIGHT_1, "device_reference": "description"})

    def test_the_lookup_for_a_missing_device_reads_nothing(self):
        """Only the devices of its kind are asked for: no property, no read."""
        lookup = state_lookup({**LIGHT_1, "device_reference": "name"})
        assert lookup["device_property"] is None and lookup["environment_variable"] is None
        assert lookup["device_class"] == "homeont:OnOffLight"
        # A found or described device is asked about as parsed.
        found = {**LIGHT_1, "device_reference": "name", "artifact_name": "b_1"}
        assert state_lookup(found) == found

    def test_the_answer_lists_what_the_room_has(self):
        answer = missing_device_answer(LIGHT_1, ["bedroom_dimmable_light_1", None])
        assert answer == ("I can't find the on/off light you named in the bedroom. "
                          "It has: bedroom_dimmable_light_1.")
        assert missing_device_answer(LIGHT_1, []).endswith("in the bedroom.")
