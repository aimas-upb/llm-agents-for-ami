"""SHTD serves scaled Matter types in human units; writes must scale back."""

from ami_agents.environment.integration.SimuHome.td_builder import report_value, wire_value


def test_writes_invert_reads():
    assert wire_value("temperature", 21.5) == 2150          # 0.01 C on the wire
    assert wire_value("percent100ths", 80) == 8000          # 0.01 % on the wire
    for matter_type, stored in (("temperature", 2150), ("percent100ths", 8000)):
        assert wire_value(matter_type, report_value(matter_type, stored)) == stored


def test_other_types_pass_through():
    assert wire_value("uint8", 3) == 3
    assert wire_value("temperature", True) is True
    assert wire_value(None, "Heavy") == "Heavy"
