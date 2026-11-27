"""One artifact per HA device, served end to end.

The fake home is one area, `invertoare`, holding the SolarAssistant Effekta
inverter (one device, three entities: a battery reading and two `select`s
that HA's old per-entity routing could not tell apart) and a device-less
template sensor. The tests check that:
  * the device is one artifact, the device-less entity another;
  * its TD names affordances from the entities, and every form targets
    HASP's own `/properties/<name>` and `/actions/<name>` routes -- no HA
    service path and no entity id;
  * each action reaches its own entity, each property reads its own state;
  * a name the artifact does not have is a 404 listing the names it has;
  * a state change is published on the device's topic with the property name;
  * declared a device group by the home, the area has no place and no
    environment, its devices are located nowhere and observe nothing, and
    answers still name the group;
  * a 2-in-1 device (a second fake home: the Blue Air's purifier fan and
    dehumidifier) is one artifact typed with both classes, each half typed by
    its own rules.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient
from rdflib import Graph, URIRef

import hasp as appmod
from hasp_namespaces import HCTL, HMAS, HOMEONT, SOSA, TD, TDSOSA
from rdflib.namespace import RDF, RDFS

AREA = "invertoare"
BASE = f"http://testserver/workspaces/{AREA}"
CHARGER = "select.solarassistant_effekta_charger_source_priority"
OUTPUT = "select.solarassistant_effekta_output_source_priority"
SOC = "sensor.solarassistant_effekta_battery_state_of_charge"
TEMPERATURE = "sensor.solarassistant_effekta_temperature"


# -- the fake home ----------------------------------------------------------------

class FakeHAWS:
    """HA's registries: one area, one inverter device, one device-less sensor."""

    async def get_areas(self):
        return [{"area_id": AREA, "name": "Invertoare"}]

    async def get_devices(self, area_id=None):
        return [{"id": "effekta", "name": "SolarAssistant Effekta", "area_id": AREA,
                 "manufacturer": "Effekta", "model": "AX M1 5000-48"}]

    async def get_entities(self):
        return [
            {"entity_id": SOC, "device_id": "effekta", "area_id": None},
            {"entity_id": CHARGER, "device_id": "effekta", "area_id": None},
            {"entity_id": OUTPUT, "device_id": "effekta", "area_id": None},
            {"entity_id": TEMPERATURE, "device_id": "effekta", "area_id": None},
            {"entity_id": "sensor.total_power", "device_id": None, "area_id": AREA,
             "original_name": "Total power"},
        ]

    async def close(self):
        return None


class FakeHAREST:
    """HA's states and services; records every service call."""

    def __init__(self):
        self.calls = []

    async def get_states(self):
        options = {
            CHARGER: ["Solar first", "Solar and utility simultaneously", "Solar only"],
            OUTPUT: ["Solar/Battery/Utility", "Utility/Solar/Battery"],
        }
        return [
            {"entity_id": SOC, "state": "75",
             "attributes": {"device_class": "battery", "unit_of_measurement": "%"}},
            {"entity_id": CHARGER, "state": "Solar first", "attributes": {"options": options[CHARGER]}},
            {"entity_id": OUTPUT, "state": "Solar/Battery/Utility", "attributes": {"options": options[OUTPUT]}},
            # A °C reading the defaults would take for the room's air.
            {"entity_id": TEMPERATURE, "state": "29.1",
             "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"}},
            {"entity_id": "sensor.total_power", "state": "1033",
             "attributes": {"device_class": "power", "unit_of_measurement": "W"}},
        ]

    async def get_services(self):
        return [{"domain": "select", "services": {"select_option": {
            "target": {"entity": [{"domain": ["select"]}]},
            "fields": {"option": {"required": True, "selector": {"text": {}}}},
        }}}]

    async def call_service(self, domain, service, data):
        self.calls.append((domain, service, data))
        return [{"entity_id": data["entity_id"], "state": data.get("option"), "attributes": {}}]

    async def close(self):
        return None


@pytest.fixture()
def client(monkeypatch):
    """A HASP app over the fake home, with no area filter and no startup task."""
    monkeypatch.setattr(appmod, "AREAS", set(), raising=False)
    monkeypatch.setattr(appmod, "ha_client", FakeHAWS())
    monkeypatch.setattr(appmod, "ha_rest", FakeHAREST())
    monkeypatch.setattr(appmod.app.state, "graph_cache", None, raising=False)
    appmod.subscriptions.clear()
    return TestClient(appmod.app)


def _graph(client, path):
    """A HASP document, parsed."""
    response = client.get(path)
    assert response.status_code == 200, response.text
    return Graph().parse(data=response.text, format="turtle")


# -- artifacts ----------------------------------------------------------------------

def test_the_device_is_one_artifact(client):
    """Three entities, one device: one artifact; the device-less sensor its own."""
    listing = _graph(client, f"/workspaces/{AREA}/artifacts")
    contained = {str(o) for o in listing.objects(URIRef(f"{BASE}#workspace"), HMAS.contains)}
    assert contained == {f"{BASE}/artifacts/solarassistant_effekta#artifact",
                         f"{BASE}/artifacts/total_power#artifact"}


def test_the_td_names_affordances_from_the_entities(client):
    """The device prefix is stripped; actions are `<key>_<service>`."""
    td = _graph(client, f"/workspaces/{AREA}/artifacts/solarassistant_effekta")
    names = {str(n) for n in td.objects(None, TD.name)}
    assert {"battery_state_of_charge", "charger_source_priority", "output_source_priority",
            "charger_source_priority_select_option",
            "output_source_priority_select_option"} <= names


def test_every_form_targets_hasps_own_routes(client):
    """No HA service path and no entity id in any form target."""
    td = _graph(client, f"/workspaces/{AREA}/artifacts/solarassistant_effekta")
    targets = [str(t) for t in td.objects(None, HCTL.hasTarget)]
    own = [t for t in targets if t.startswith(f"{BASE}/artifacts/solarassistant_effekta/")]
    assert own, targets
    assert all("/ha/" not in t and "select." not in t and "sensor." not in t for t in targets)
    assert f"{BASE}/artifacts/solarassistant_effekta/actions/charger_source_priority_select_option" in own


# -- routing ------------------------------------------------------------------------

def test_each_action_reaches_its_own_entity(client):
    """Two selects on one device: each action calls HA on its own entity."""
    response = client.post(
        f"/workspaces/{AREA}/artifacts/solarassistant_effekta/actions/charger_source_priority_select_option",
        json={"option": "Solar only"})
    assert response.status_code == 200, response.text
    assert appmod.ha_rest.calls == [
        ("select", "select_option", {"entity_id": CHARGER, "option": "Solar only"})]


def test_each_property_reads_its_own_state(client):
    """The battery reading, as a number; a select's option, as text."""
    path = f"/workspaces/{AREA}/artifacts/solarassistant_effekta/properties"
    assert client.get(f"{path}/battery_state_of_charge").json() == 75
    assert client.get(f"{path}/output_source_priority").json() == "Solar/Battery/Utility"


def test_an_unknown_name_lists_the_names_the_artifact_has(client):
    """404, with the artifact's own names, and no call to HA."""
    response = client.post(f"/workspaces/{AREA}/artifacts/solarassistant_effekta/actions/turn_on")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error"] == "unknown_action"
    assert "charger_source_priority_select_option" in detail["available"]
    assert appmod.ha_rest.calls == []


def test_an_action_is_not_readable_and_a_property_not_invocable(client):
    """The two routes keep to their own kind of affordance."""
    path = f"/workspaces/{AREA}/artifacts/solarassistant_effekta"
    assert client.get(f"{path}/properties/charger_source_priority_select_option").status_code == 404
    assert client.post(f"{path}/actions/battery_state_of_charge").status_code == 404


def test_an_old_entity_link_finds_the_device(client):
    """A request naming an entity reaches the device's artifact."""
    response = client.get(f"/workspaces/{AREA}/artifacts/{CHARGER}")
    assert response.status_code == 200
    assert "solarassistant_effekta#artifact" in response.text


# -- notifications ------------------------------------------------------------------

def test_a_state_change_is_published_on_the_devices_topic(client):
    """The entity's context is its device's artifact, with the property that
    reads its state."""
    # 1. Build the cache through a request, then ask it about one entity.
    client.get(f"/workspaces/{AREA}/artifacts")
    cache = appmod.app.state.graph_cache
    contexts = asyncio.run(cache.get_entity_contexts(SOC))

    # 2. One context: the device's topic and the property name.
    assert len(contexts) == 1
    assert contexts[0]["artifact_topic"].endswith("/artifacts/solarassistant_effekta#artifact")
    assert contexts[0]["affordance"] == "battery_state_of_charge"


# -- a device group -----------------------------------------------------------------

@pytest.fixture()
def group_client(client, tmp_path, monkeypatch):
    """The same home, with Invertoare declared a device group and the
    inverter's temperature named a device reading (as homelab.yaml does)."""
    # 1. The home config.
    path = tmp_path / "home.yaml"
    path.write_text(
        "semantic:\n"
        "  areas:\n"
        "    invertoare: {group_class: 'homeont:DeviceGroup'}\n"
        "  devices:\n"
        "    SolarAssistant Effekta:\n"
        "      device_class: 'homeont:HybridInverter'\n"
        "      properties:\n"
        "        temperature: 'homeont:DeviceInternalTemperature'\n")
    monkeypatch.setenv("SEMANTIC_CONFIG", str(path))
    appmod.reload_semantics()
    yield client
    # 2. Back to the defaults for the next test.
    monkeypatch.setenv("SEMANTIC_CONFIG", "")
    appmod.reload_semantics()


def test_a_device_group_has_no_place_and_no_environment(group_client):
    """The workspace stands for a group: typed, named, linked both ways."""
    workspace = _graph(group_client, f"/workspaces/{AREA}")
    group = URIRef(f"{BASE}#group")
    ws = URIRef(f"{BASE}#workspace")
    assert (group, RDF.type, HOMEONT.DeviceGroup) in workspace
    assert (group, HOMEONT.isGroupOfWorkspace, ws) in workspace
    assert (ws, HOMEONT.hasGroup, group) in workspace
    assert str(workspace.value(group, RDFS.label)) == "Invertoare"
    assert (URIRef(f"{BASE}#place"), None, None) not in workspace
    assert (None, HOMEONT.hasEnvironment, None) not in workspace


def test_a_group_device_is_located_nowhere_and_observes_nothing(group_client):
    """The inverter's temperature is its own reading, not anyone's air."""
    td = _graph(group_client, f"/workspaces/{AREA}/artifacts/solarassistant_effekta")
    art = URIRef(f"{BASE}/artifacts/solarassistant_effekta#artifact")
    assert (art, HOMEONT.isLocatedIn, None) not in td
    assert (art, SOSA.observes, None) not in td
    assert (art, RDF.type, HOMEONT.HybridInverter) in td
    (prop,) = [p for p in td.objects(art, TD.hasPropertyAffordance)
               if str(td.value(p, TD.name)) == "temperature"]
    assert (prop, RDF.type, HOMEONT.DeviceInternalTemperature) in td


def test_answers_name_the_group(group_client):
    """EnvExplorer's answer label for a group's device is the group's name."""
    from ami_agents.agents.env_explorer.utils.capability_resolution import workspace_label
    graph = (_graph(group_client, f"/workspaces/{AREA}")
             + _graph(group_client, f"/workspaces/{AREA}/artifacts/solarassistant_effekta"))
    art = URIRef(f"{BASE}/artifacts/solarassistant_effekta#artifact")
    assert workspace_label(graph, art) == "Invertoare"


# -- a 2-in-1 device ----------------------------------------------------------------

GARAGE = "garaj"
GARAGE_BASE = f"http://testserver/workspaces/{GARAGE}"
PURIFIER = "fan.blue_air_garage"
DEHUMIDIFIER = "humidifier.blue_air_garage_dehumidifier"


class TwoInOneHAWS(FakeHAWS):
    """A garage holding one device with two halves: a purifier fan and a
    dehumidifier (the Blue Air Garage)."""

    async def get_areas(self):
        return [{"area_id": GARAGE, "name": "Garaj"}]

    async def get_devices(self, area_id=None):
        return [{"id": "blueair", "name": "Blue Air Garage", "area_id": GARAGE,
                 "manufacturer": "Blueair", "model": "2-in-1"}]

    async def get_entities(self):
        return [{"entity_id": PURIFIER, "device_id": "blueair", "area_id": None},
                {"entity_id": DEHUMIDIFIER, "device_id": "blueair", "area_id": None}]


class TwoInOneHAREST(FakeHAREST):
    """The two halves' states, and the services that drive them."""

    async def get_states(self):
        return [
            {"entity_id": PURIFIER, "state": "off",
             "attributes": {"preset_modes": ["auto", "night"], "percentage": 0}},
            {"entity_id": DEHUMIDIFIER, "state": "off",
             "attributes": {"device_class": "dehumidifier", "humidity": 50,
                            "min_humidity": 30, "max_humidity": 80}},
        ]

    async def get_services(self):
        target = {"target": {"entity": [{"domain": ["humidifier"]}]}, "fields": {}}
        return [{"domain": "humidifier", "services": {"turn_on": target, "turn_off": target}}]


@pytest.fixture()
def two_in_one(monkeypatch, tmp_path):
    """HASP over the 2-in-1 home, with the Blue Air typed both ways by the
    home config (as homelab.yaml does)."""
    # 1. The fake home.
    monkeypatch.setattr(appmod, "AREAS", set(), raising=False)
    monkeypatch.setattr(appmod, "ha_client", TwoInOneHAWS())
    monkeypatch.setattr(appmod, "ha_rest", TwoInOneHAREST())
    monkeypatch.setattr(appmod.app.state, "graph_cache", None, raising=False)
    appmod.subscriptions.clear()
    # 2. The home config: one device, two classes.
    path = tmp_path / "home.yaml"
    path.write_text(
        "semantic:\n"
        "  areas:\n"
        "    garaj: {location_class: 'homeont:Garage'}\n"
        "  devices:\n"
        "    Blue Air Garage:\n"
        "      device_class: ['homeont:AirPurifier', 'homeont:Dehumidifier']\n")
    monkeypatch.setenv("SEMANTIC_CONFIG", str(path))
    appmod.reload_semantics()
    yield TestClient(appmod.app)
    # 3. Back to the defaults for the next test.
    monkeypatch.setenv("SEMANTIC_CONFIG", "")
    appmod.reload_semantics()


def test_a_2_in_1_is_one_artifact_of_both_classes(two_in_one):
    """Both classes on the artifact; each half's affordance typed by its own
    rules; the dehumidifier's turn_on dries the garage's air."""
    # 1. The artifact is both things.
    td = _graph(two_in_one, f"/workspaces/{GARAGE}/artifacts/blue_air_garage")
    art = URIRef(f"{GARAGE_BASE}/artifacts/blue_air_garage#artifact")
    assert (art, RDF.type, HOMEONT.AirPurifier) in td
    assert (art, RDF.type, HOMEONT.Dehumidifier) in td
    # 2. The dehumidifier half's state is a dehumidifier's on/off.
    props = {str(td.value(p, TD.name)): p for p in td.objects(art, TD.hasPropertyAffordance)}
    assert (props["dehumidifier"], RDF.type, HOMEONT.DehumidifierOnOff) in td
    # 3. Its turn_on lowers the garage's relative humidity.
    actions = {str(td.value(a, TD.name)): a for a in td.objects(art, TD.hasActionAffordance)}
    (actuation,) = td.objects(actions["dehumidifier_turn_on"], TDSOSA.hasEffectActuation)
    (variable,) = td.objects(actuation, TDSOSA.decreasesObservableProperty)
    assert str(variable).endswith("relative_humidity")
