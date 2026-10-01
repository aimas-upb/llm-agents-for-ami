"""Which actions the GOAL_REQUEST parser is shown.

Actions are typed from two vocabularies -- homeont (`SetModeCommand`) and SAREF
(`OnCommand`, `SetAbsoluteLevelCommand`) -- and both describe what the action
does. Only protocol actions (WebSub, artifact CRUD) are left out.
"""

from ami_agents.agents.user_assistant.pipeline import build_capabilities_hierarchical_text
from ami_agents.shared.utils.namespaces import action_types


def _summary(*actions):
    return {
        "discovery_complete": True,
        "workspaces": [{
            "name": "lab308",
            "artifacts": [{
                "name": "lights_308",
                "affordances": [
                    {"type": "action_affordance", "name": name, "semantic_types": types}
                    for name, types in actions
                ] + [{"type": "property_affordance", "name": "brightness",
                      "semantic_types": ["td:PropertyAffordance"]}],
            }],
        }],
    }


def test_saref_and_homeont_commands_are_listed():
    text = build_capabilities_hierarchical_text(_summary(
        ("LightTurnOn", ["td:ActionAffordance", "saref:OnCommand"]),
        ("SetHvacMode", ["td:ActionAffordance", "homeont:SetModeCommand"]),
        ("SetTemperature", ["https://saref.etsi.org/core/SetAbsoluteLevelCommand"]),
    ))
    assert "action: LightTurnOn" in text
    assert "action: SetHvacMode" in text
    assert "action: SetTemperature" in text


def test_protocol_actions_are_left_out():
    text = build_capabilities_hierarchical_text(_summary(
        ("subscribeToArtifact", ["td:ActionAffordance", "https://purl.org/hmas/websub/subscribeToArtifact"]),
        ("getArtifactRepresentation", ["td:ActionAffordance", "https://purl.org/hmas/jacamo/PerceiveArtifact"]),
    ))
    assert "subscribeToArtifact" not in text
    assert "getArtifactRepresentation" not in text
    assert "property: brightness" in text


def test_action_types_keeps_both_vocabularies():
    assert action_types(["td:ActionAffordance", "saref:OnCommand",
                         "http://example.org/homeont/SetModeCommand",
                         "http://example.org/StatusCommand"]) == [
        "saref:OnCommand", "homeont:SetModeCommand"]
