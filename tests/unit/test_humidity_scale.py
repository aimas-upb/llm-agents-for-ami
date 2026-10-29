"""A device's humidity reading is reported in %, as the room's is.

Matter's RelativeHumidityMeasurement carries 100 x the relative humidity
(5404 is 54.04 %), like temperature's hundredths of a degree, but the Matter
registry types it as a plain uint16. A dehumidifier's reading was served as
5404 beside the room sensor's 54.04, and a state answer averaged the two
("On average, 2729.02"). `tables.SCALED_ATTRIBUTE_TYPES` types it as
`percent100ths`, so `report_value` scales it:

    2.RelativeHumidityMeasurement.MeasuredValue   5404 -> 54.04
"""

from ami_agents.environment.integration.SimuHome.classify import classify_device
from ami_agents.environment.integration.SimuHome.td_builder import report_value

PATH = "2.RelativeHumidityMeasurement.MeasuredValue"


def test_a_dehumidifiers_humidity_is_reported_in_percent():
    """The record carries the spec's scale, and the reported value is in %."""
    records = classify_device({PATH: 5404, "1.OnOff.OnOff": True})
    record = records[PATH]
    assert record["type"] == "percent100ths"
    assert report_value(record["type"], record["value"]) == 54.04
