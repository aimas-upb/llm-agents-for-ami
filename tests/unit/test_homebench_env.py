"""HomeBench environment (P2): mapping tables, ontology, TDs, engine app, HBTD.

No LLM and no server process: the engine app and HBTD run in-process under
FastAPI's ``TestClient`` (HBTD with a stub client wrapping an in-process engine). The
end-to-end gate (a real engine + HBTD pair crawled with ``YggdrasilIntegration``) is
``HomeBench/verify_phase2.py``; the tests here cover the same rules without processes:

* mapping tables: settled, the physics tables empty, ``command_for`` and
  ``action_for_command`` inverse, the gold string a function of
  ``(device, action, value)``; the generator, with approved rows preserved;
* ontology: the six new device classes and their commands and properties, curtains
  and blinds kept apart by the type closure;
* the TD: one property per attribute, one action per settable attribute, the schemas,
  no environment and no TD-SOSA effect;
* the engine app and ``HomeBenchClient``; HBTD: 400 for a missing or wrongly typed
  value, 422 for a refused one, the WebSub payload with ``propertyUri`` and ``value``;
* the gold round trip over the real dataset (skipped when ``../HomeBench`` is absent).
"""

import copy
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from rdflib import Graph, Namespace, RDF, RDFS, URIRef

from ami_agents.environment.integration.HomeBench import (
    engine, hbtd, simulator, verify_phase2,
)
from ami_agents.environment.integration.HomeBench.engine import (
    EngineError, HomeBenchEngine,
)
from ami_agents.environment.integration.HomeBench.mappings import (
    MappingTableError, load_mappings,
)
from ami_agents.environment.integration.HomeBench.sim_client import (
    HomeBenchClient, SimuHomeError,
)
from ami_agents.environment.integration.HomeBench.td_builder import HomeBenchTD
from ami_agents.shared.utils.vocabulary import close_types
from tests.homebench import convert, mapping_tables

ROOT = Path(__file__).resolve().parents[2]
HOMEBENCH = ROOT.parent / "HomeBench"
HAVE_DATA = (HOMEBENCH / convert.TEST_DATA).is_file()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="../HomeBench absent")

HOMEONT = Namespace("http://example.org/homeont/")
SAREF = Namespace("https://saref.etsi.org/core/")
TD = Namespace("https://www.w3.org/2019/wot/td#")
JS = Namespace("https://www.w3.org/2019/wot/json-schema#")
TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
ONTOLOGY = ROOT / "ami_agents" / "shared" / "ontologies" / "homeont.ttl"
BASE = "http://hb.test"
HOME_ID = "home7"


# -- a synthetic home with every device type -----------------------------------------------------


def _num(value, low=0, high=100):
    """A HomeBench numeric attribute block."""
    return {"value": value, "lowest": str(low), "highest": str(high)}


def _enum(value, options):
    """A HomeBench enum attribute block."""
    return {"value": value, "options": options}


LEVELS = ["auto", "low", "medium", "high"]


def raw_home():
    """A two-room HomeBench home holding all 15 device types; a plain light and a
    brightness-only light give the three light classes (with the colour light)."""
    return {
        "living_room": {
            "room_name": "living_room",
            "light": {"state": "on", "attributes": {
                "brightness": _num(40), "color": {"value": [1, 2, 3]}}},
            "air_conditioner": {"state": "off", "attributes": {
                "temperature": _num(24, 16, 30), "mode": _enum(
                    "cool", ["cool", "heat", "fan_only", "dry"]),
                "fan_speed": _enum("auto", LEVELS),
                "swing": _enum("auto", ["auto", "up", "middle", "down"])}},
            "heating": {"state": "on", "attributes": {
                "temperature": _num(25, 30, 100),
                "mode": _enum("heat", ["heat", "fan_only"]),
                "fan_speed": _enum("auto", LEVELS)}},
            "fan": {"state": "off", "attributes": {
                "speed": _enum("low", LEVELS),
                "swing": _enum("auto", ["auto", "up", "middle", "down"])}},
            "air_purifiers": {"state": "on", "attributes": {
                "mode": _enum("auto", ["auto", "sleep"]),
                "fan_speed": _enum("auto", LEVELS)}},
            "humidifier": {"state": "on", "attributes": {
                "mode": _enum("auto", ["auto", "sleep"]),
                "intensity": _num(8), "tank": _num(50)}},
            "dehumidifiers": {"state": "off", "attributes": {
                "mode": _enum("sleep", ["auto", "sleep"]),
                "intensity": _num(8), "tank": _num(50)}},
            "aromatherapy": {"state": "off", "attributes": {
                "intensity": _num(5), "interval": _num(25, 10, 60)}},
            "water_heater": {"state": "on", "attributes": {
                "temperature": _num(60, 30, 100),
                "mode": _enum("eco", ["heating", "eco"])}},
            "curtain": {"state": "open", "attributes": {" degree": _num(0)}},
            "blinds": {"state": "closed", "attributes": {}},
            "garage_door": {"state": "closed", "attributes": {}},
            "media_player": {"state": "paused", "attributes": {"volume": _num(5)}},
            "trash": {"state": "full", "attributes": {}},
        },
        "kitchen": {
            "room_name": "kitchen",
            "light": {"state": "off", "attributes": {"brightness": _num(10)}},
            "curtain": {"state": "closed", "attributes": {" degree": _num(30)}},
        },
        "bathroom": {"room_name": "bathroom",
                     "light": {"state": "on", "attributes": {}}},
        "VacuumRobot": {"state": "on", "attributes": {
            "battery": _num(80), "mode": _enum("auto", ["auto", "strong", "sleep"])}},
    }


@pytest.fixture()
def home():
    """The synthetic home as an ``initial_home_config``."""
    return convert.convert_home({"home_id": 7, "home_status": raw_home()})


@pytest.fixture()
def machine(home):
    """An engine holding the synthetic home."""
    return HomeBenchEngine(home)


@pytest.fixture()
def td(machine):
    """The TD builder over the engine's current state."""
    return HomeBenchTD(BASE, HOME_ID, machine.home_state())


def graph_of(g):
    """Parse a built graph back from Turtle, as an agent receives it."""
    out = Graph()
    out.parse(data=g.serialize(format="turtle"), format="turtle")
    return out


def affordances(g, art, predicate):
    """``{td:name: node}`` of a Thing's property or action affordances."""
    return {str(g.value(n, TD.name)): n for n in g.objects(art, predicate)
            if (n, RDF.type, TD.PropertyAffordance) in g
            or (n, RDF.type, TD.ActionAffordance) in g}


def thing(td, room, device):
    """``(graph, artifact node)`` of a device's TD."""
    g = graph_of(td.artifact(room, device))
    return g, URIRef(f"{td.artifact_path(room, device)}#artifact")


# -- mapping tables ----------------------------------------------------------------------------------


class TestMappings:
    def test_tables_are_settled_and_the_physics_tables_empty(self):
        m = load_mappings()
        assert m.unsettled() == {}
        assert (m.observable_properties, m.ambient_sensors, m.actuation_effects) == \
            ({}, {}, [])
        assert set(m.device_types) == set(engine.DEVICE_STATE_COMMANDS)
        assert set(m.rooms) == set(engine.ROOMS) | {engine.HOUSEHOLD_ROOM}

    def test_every_attribute_has_an_affordance_and_settable_ones_an_action(self, machine):
        """Each attribute of each device is one property; each settable one is also
        an action, and the actions together reach exactly the home's commands."""
        m = load_mappings()
        for device_id in machine.device_ids():
            device = machine.device(device_id)
            rows = m.affordances(device["device_type"], device["attributes"])
            assert {r["attribute"] for r in rows} == set(device["attributes"])
            commands = set()
            for row in (r for r in rows if r["actuatable"]):
                for value in self.values_of(row, device):
                    commands.add(m.command_for(device["device_type"],
                                               row["affordance_name"], value)[0])
            assert commands == set(machine.exposed_commands(device_id)), device_id

    @staticmethod
    def values_of(row, device):
        """Values to try on an action row: both booleans, each option, a few numbers."""
        kind = row["value_kind"]
        if kind == "boolean":
            return [True, False]
        if kind == "none":
            return [None]
        if kind == "rgb":
            return [[1, 2, 3]]
        if row.get("values"):
            return list(row["values"])
        rule = device["constraints"][row["attribute"]]
        if kind == "enum":
            return list(rule["options"])
        return [rule["lowest"], rule["highest"]]

    def test_command_for_and_action_for_command_are_inverse(self, machine):
        m = load_mappings()
        checked = 0
        for device_id in machine.device_ids():
            device = machine.device(device_id)
            dtype = device["device_type"]
            for row in (r for r in m.affordances(dtype, device["attributes"])
                        if r["actuatable"]):
                for value in self.values_of(row, device):
                    command, arg = m.command_for(dtype, row["affordance_name"], value)
                    action, back = m.action_for_command(dtype, command, arg)
                    assert (action, back) == (row["affordance_name"], value), (
                        dtype, command, arg)
                    checked += 1
        assert checked > 100

    @pytest.mark.parametrize("device_id,dtype,action,value,gold", [
        ("living_room_light", "light", "onOff", True, "living_room.light.turn_on()"),
        ("living_room_light", "light", "onOff", False, "living_room.light.turn_off()"),
        ("kitchen_light", "light", "brightness", 20, "kitchen.light.set_brightness(20)"),
        ("living_room_curtain", "curtain", "openClose", "closed",
         "living_room.curtain.close()"),
        ("living_room_curtain", "curtain", "degree", 30, "living_room.curtain.set_degree(30)"),
        ("living_room_media_player", "media_player", "playback", "playing",
         "living_room.media_player.play()"),
        ("living_room_trash", "trash", "pack", None, "living_room.trash.pack()"),
        ("living_room_fan", "fan", "swing", "up", "living_room.fan.set_swing(up)"),
        ("living_room_air_conditioner", "air_conditioner", "fanSpeed", "auto",
         "living_room.air_conditioner.set_fan_speed(auto)"),
        ("household_vacuum_robot", "vacuum_robot", "mode", "auto",
         "vacuum_robot.set_mode(auto)"),
        ("household_vacuum_robot", "vacuum_robot", "cleaningArea", "kitchen",
         "vacuum_robot.set_cleaning_area(kitchen)"),
    ])
    def test_gold_string_is_a_function_of_device_action_value(
            self, device_id, dtype, action, value, gold):
        assert load_mappings().gold_string(device_id, dtype, action, value) == gold

    def test_unknown_action_or_value_is_an_error(self):
        m = load_mappings()
        with pytest.raises(MappingTableError):
            m.command_for("light", "nope", 1)
        with pytest.raises(MappingTableError):
            m.command_for("light", "onOff", "on")        # a string, not a boolean
        with pytest.raises(MappingTableError):
            m.action_for_command("light", "open", None)  # a curtain command
        with pytest.raises(MappingTableError):
            m.gold_string("kitchen_curtain", "light", "onOff", True)  # id/type mismatch

    def test_vacuum_commands_the_home_does_not_offer_have_no_action(self):
        for command in ("start", "stop", "pause", "charge"):
            with pytest.raises(MappingTableError):
                load_mappings().action_for_command("vacuum_robot", command, None)

    def test_read_and_write_attribute_are_inverse(self):
        m = load_mappings()
        row = m.affordance("light", "onOff")
        assert m.read_value(row, "on") is True and m.read_value(row, "off") is False
        assert m.write_attribute(row, True) == "on" and m.write_attribute(row, False) == "off"
        assert m.read_value(m.affordance("light", "brightness"), 40) == 40
        with pytest.raises(MappingTableError):
            m.read_value(row, "dim")

    def test_light_class_follows_its_attributes(self):
        m = load_mappings()
        assert m.device_family("light", {"state": "on", "brightness": 1, "color": [1, 2, 3]}
                               )["td_types"][0] == "homeont:ColorLight"
        assert m.device_family("light", {"state": "on", "brightness": 1}
                               )["td_types"][0] == "homeont:DimmableLight"
        assert m.device_family("light", {"state": "on"})["td_types"][0] == "homeont:OnOffLight"


# -- the generator -------------------------------------------------------------------------------------


class TestGenerator:
    def test_tables_from_homes(self, home):
        tables = mapping_tables.generate_tables([home])
        assert set(tables) == {"device_type_map", "room_map", "property_classes",
                               "actuatable_properties", "command_map",
                               *mapping_tables.EMPTY_TABLES}
        assert all(tables[name] == [] for name in mapping_tables.EMPTY_TABLES)
        assert {r["key"] for r in tables["room_map"]} == {
            "living_room", "kitchen", "bathroom", "household"}
        assert all(r["status"] == "auto" for rows in tables.values() for r in rows)
        keys = {r["key"] for r in tables["command_map"]}
        assert {"light.onOff", "light.brightness", "trash.pack"} <= keys
        assert "vacuum_robot.state" not in keys        # read only: no command

    def test_a_device_type_attribute_or_room_without_a_specification_fails(self, home):
        for edit in (
            lambda h: h["rooms"]["kitchen"]["devices"].append(
                {"device_id": "kitchen_toaster", "device_type": "toaster",
                 "attributes": {"state": "on"}, "constraints": {}}),
            lambda h: h["rooms"]["kitchen"]["devices"][0]["attributes"].update(foo=1),
            lambda h: h["rooms"].update(attic={"state": {}, "devices": []}),
        ):
            bad = copy.deepcopy(home)
            edit(bad)
            with pytest.raises(mapping_tables.MappingError):
                mapping_tables.generate_tables([bad])

    def test_approved_rows_survive_regeneration(self, home, tmp_path):
        tables = mapping_tables.generate_tables([home])
        mapping_tables.write_tables(tables, tmp_path)
        # A reviewer edits one row and approves it, and adds one the generator lacks.
        path = tmp_path / "command_map.yaml"
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        edited = next(r for r in doc["rows"] if r["key"] == "light.brightness")
        edited.update(gold_op="set_brightness_v2({value})", status="approved")
        doc["rows"].append({"key": "light.extra", "device_type": "light", "action": "x",
                            "gold_op": "x()", "status": "approved"})
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        mapping_tables.write_tables(mapping_tables.generate_tables([home]), tmp_path)
        again = yaml.safe_load(path.read_text(encoding="utf-8"))["rows"]
        kept = {r["key"]: r for r in again}
        assert kept["light.brightness"]["gold_op"] == "set_brightness_v2({value})"
        assert "light.extra" in kept                     # not dropped by the generator
        assert kept["light.onOff"]["status"] == "auto"   # the rest regenerated

    @needs_data
    def test_the_tables_on_disk_are_what_the_generator_writes(self):
        records = convert.read_jsonl(HOMEBENCH / convert.HOME_DATA)
        tables = mapping_tables.generate_tables(
            [convert.convert_home(r) for r in records])
        for name, rows in tables.items():
            on_disk = yaml.safe_load(
                (mapping_tables.MAPPINGS_DIR / f"{name}.yaml").read_text(
                    encoding="utf-8"))["rows"]
            approved = {r["key"]: r for r in on_disk if r.get("status") == "approved"}
            assert [approved.get(r["key"], r) for r in rows] == \
                [r for r in on_disk if r["key"] in {x["key"] for x in rows}], name


# -- the ontology -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def onto():
    """The live homeont.ttl."""
    g = Graph()
    g.parse(str(ONTOLOGY), format="turtle")
    return g


class TestOntology:
    @pytest.mark.parametrize("cls,parent", [
        ("Curtain", "WindowCoveringController"), ("Blinds", "WindowCoveringController"),
        ("Aromatherapy", None), ("GarageDoor", None), ("WaterHeater", None),
        ("Trash", None),
    ])
    def test_new_device_classes_sit_under_a_saref_family(self, onto, cls, parent):
        node = HOMEONT[cls]
        assert (node, RDF.type, OWL.Class) in onto
        assert list(onto.objects(node, RDFS.label))
        parents = set(onto.objects(node, RDFS.subClassOf))
        if parent:
            assert HOMEONT[parent] in parents
        else:
            assert parents & {SAREF.Appliance, SAREF.Actuator}

    def test_new_words_are_alt_labels(self, onto):
        skos = Namespace("http://www.w3.org/2004/02/skos/core#")
        words = {str(o).lower() for cls in ("Trash", "WaterHeater", "Aromatherapy",
                                            "GarageDoor", "Curtain", "Blinds")
                 for o in onto.objects(HOMEONT[cls], skos.altLabel)}
        assert {"trash can", "boiler", "aroma diffuser", "garage door opener",
                "drapes", "venetian blinds"} <= words

    def test_every_class_the_tables_name_is_defined(self, onto):
        defined = {str(s) for s in onto.subjects(RDF.type, OWL.Class)}
        m = load_mappings()
        curies = set(m.property_classes)
        curies |= {r["action_class"] for rows in m._affordances.values()
                   for r in rows if r.get("action_class")}
        curies |= {t for row in m.device_types.values() for t in row["td_types"]}
        curies |= {t for row in m.device_types.values()
                   for v in row.get("variants") or [] for t in v["td_types"]}
        curies |= {r["homeont_class"] for r in m.rooms.values()}
        for curie in curies:
            prefix, local = curie.split(":", 1)
            ns = {"homeont": str(HOMEONT), "saref": str(SAREF)}[prefix]
            if prefix == "homeont":
                assert ns + local in defined, curie

    def test_command_classes_specialise_their_saref_commands(self, onto):
        assert set(onto.objects(HOMEONT.SetOpenCloseCommand, RDFS.subClassOf)) == {
            SAREF.OpenCommand, SAREF.CloseCommand}
        assert SAREF.Command in set(onto.objects(HOMEONT.PackCommand, RDFS.subClassOf))
        assert SAREF.Command in set(onto.objects(HOMEONT.SetColorCommand, RDFS.subClassOf))

    def test_nothing_existing_was_removed(self, onto):
        """The root file this one is checked against is the committed one."""
        for cls in ("WindowCoveringController", "SetOnOffCommand", "ColorLight",
                    "Heater", "MediaPlayer", "OperationalState"):
            assert (HOMEONT[cls], RDF.type, OWL.Class) in onto


# -- the TD ----------------------------------------------------------------------------------------------------


class TestTD:
    def test_the_room_lists_every_device_and_the_vacuum_has_its_workspace(self, td, machine):
        listed = set()
        for room in td.room_ids():
            g = graph_of(td.room_workspace(room))
            ws = URIRef(f"{td.room_path(room)}#workspace")
            listed |= {str(a).split("/artifacts/")[1].split("#")[0]
                       for a in g.objects(ws, URIRef("https://purl.org/hmas/contains"))
                       if "/artifacts/" in str(a)}
        assert listed == set(machine.device_ids())
        assert "household" in td.room_ids()
        household = graph_of(td.room_workspace("household"))
        assert any("household_vacuum_robot" in str(s) for s in household.subjects())

    def test_light_classes_by_attribute(self, td):
        types = lambda room, dev: {  # noqa: E731
            str(t) for t in thing(td, room, dev)[0].objects(
                thing(td, room, dev)[1], RDF.type)}
        assert str(HOMEONT.ColorLight) in types("living_room", "living_room_light")
        assert str(HOMEONT.DimmableLight) in types("kitchen", "kitchen_light")
        assert str(HOMEONT.OnOffLight) in types("bathroom", "bathroom_light")

    def test_actions_exist_only_for_attributes_the_device_has(self, td):
        g, art = thing(td, "bathroom", "bathroom_light")
        assert set(affordances(g, art, TD.hasActionAffordance)) >= {"onOff"}
        assert "brightness" not in affordances(g, art, TD.hasPropertyAffordance)
        assert "brightness" not in affordances(g, art, TD.hasActionAffordance)
        g, art = thing(td, "kitchen", "kitchen_light")
        assert "brightness" in affordances(g, art, TD.hasActionAffordance)

    def test_state_affordances(self, td):
        def shape(room, device, name, predicate, slot):
            g, art = thing(td, room, device)
            node = affordances(g, art, predicate)[name]
            schema = g.value(node, slot)
            return g, node, schema

        # onOff: a boolean, typed with the generic classes, acting upon its property.
        g, action, schema = shape("living_room", "living_room_fan", "onOff",
                                  TD.hasActionAffordance, TD.hasInputSchema)
        assert (schema, RDF.type, JS.BooleanSchema) in g
        assert (action, RDF.type, HOMEONT.SetOnOffCommand) in g
        prop = g.value(action, SAREF.actsUpon)
        assert (prop, RDF.type, HOMEONT.OnOff) in g
        assert (prop, RDF.type, TDSOSA.ActuatablePropertyAffordance) in g
        # openClose: an enum of the two positions.
        g, action, schema = shape("living_room", "living_room_blinds", "openClose",
                                  TD.hasActionAffordance, TD.hasInputSchema)
        assert {str(o) for o in g.objects(schema, JS.enum)} == {"open", "closed"}
        assert (action, RDF.type, HOMEONT.SetOpenCloseCommand) in g
        # playback: the three states, on the existing operational-state command.
        g, action, schema = shape("living_room", "living_room_media_player", "playback",
                                  TD.hasActionAffordance, TD.hasInputSchema)
        assert {str(o) for o in g.objects(schema, JS.enum)} == {
            "playing", "paused", "stopped"}
        assert (action, RDF.type, HOMEONT.SetOperationalStateCommand) in g
        # pack: no input at all; the property reads the three fill states.
        g, action, schema = shape("living_room", "living_room_trash", "pack",
                                  TD.hasActionAffordance, TD.hasInputSchema)
        assert schema is None and (action, RDF.type, HOMEONT.PackCommand) in g
        g, prop, schema = shape("living_room", "living_room_trash", "pack",
                                TD.hasPropertyAffordance, TD.hasOutputSchema)
        assert {str(o) for o in g.objects(schema, JS.enum)} == {
            "empty", "not_full", "full"}
        # The vacuum's state is readable and has no action.
        g, art = thing(td, "household", "household_vacuum_robot")
        assert "state" in affordances(g, art, TD.hasPropertyAffordance)
        assert "state" not in affordances(g, art, TD.hasActionAffordance)
        assert {"battery", "tank"} & set(affordances(g, art, TD.hasActionAffordance)) == set()

    def test_numeric_schema_carries_the_devices_own_range_and_unit(self, td):
        g, art = thing(td, "living_room", "living_room_heating")
        action = affordances(g, art, TD.hasActionAffordance)["temperature"]
        schema = g.value(action, TD.hasInputSchema)
        # The data declares 30-100 for the heater; the home file holds the correction.
        assert (int(g.value(schema, JS.minimum)), int(g.value(schema, JS.maximum))) == (16, 30)
        assert (schema, RDF.type, JS.IntegerSchema) in g
        assert g.value(schema, URIRef("http://qudt.org/schema/qudt/unit")) is not None
        assert (action, RDF.type, SAREF.SetAbsoluteLevelCommand) in g
        assert (g.value(action, SAREF.actsUpon), RDF.type, HOMEONT.HeaterHeatingSetpoint) in g

    def test_enum_options_are_the_devices_own(self, td):
        g, art = thing(td, "household", "household_vacuum_robot")
        schema = g.value(affordances(g, art, TD.hasActionAffordance)["mode"],
                         TD.hasInputSchema)
        assert {str(o) for o in g.objects(schema, JS.enum)} == {"auto", "strong", "sleep"}
        assert "auto" in str(g.value(schema, JS.description))
        g, art = thing(td, "living_room", "living_room_light")
        colour = g.value(affordances(g, art, TD.hasActionAffordance)["color"],
                         TD.hasInputSchema)
        assert (colour, RDF.type, JS.ArraySchema) in g
        assert int(g.value(colour, JS.minItems)) == int(g.value(colour, JS.maxItems)) == 3

    def test_no_physics_is_claimed(self, td):
        """No environment resource, no room variables, no sensors, no effects."""
        for room in td.room_ids():
            g = graph_of(td.room_workspace(room))
            assert not list(g.subjects(RDF.type, HOMEONT.Environment))
            assert not list(g.subjects(RDF.type, SOSA.FeatureOfInterest))
            assert td.ambient_sensors(room) == []
            for device in td.rooms[room]["devices"]:
                dg = td.artifact(room, device["device_id"])
                assert not list(dg.subject_objects(TDSOSA.hasEffectActuation))
                assert not list(dg.subject_objects(SOSA.hasFeatureOfInterest))
                assert not list(dg.subject_objects(SOSA.observes))

    def test_names_are_unique_and_every_homeont_term_is_defined(self, td, onto):
        defined = {str(s) for kind in (OWL.Class, OWL.DatatypeProperty, OWL.ObjectProperty)
                   for s in onto.subjects(RDF.type, kind)}
        emitted = set()
        for room in td.room_ids():
            for device in td.rooms[room]["devices"]:
                g, art = thing(td, room, device["device_id"])
                names = [str(g.value(n, TD.name)) for n in g.objects(art, TD.hasPropertyAffordance)]
                assert len(names) == len(set(names)), device["device_id"]
                for _s, p, o in g:
                    for term in (p, o if p == RDF.type else None):
                        if term is not None and str(term).startswith(str(HOMEONT)):
                            emitted.add(str(term))
        assert emitted and emitted <= defined, sorted(emitted - defined)

    def test_curtains_and_blinds_are_told_apart_by_the_type_closure(self, td):
        curtain, c_art = thing(td, "living_room", "living_room_curtain")
        blinds, b_art = thing(td, "living_room", "living_room_blinds")
        close_types(curtain)
        close_types(blinds)
        assert (c_art, RDF.type, HOMEONT.Curtain) in curtain
        assert (c_art, RDF.type, HOMEONT.Blinds) not in curtain
        assert (b_art, RDF.type, HOMEONT.Blinds) in blinds
        assert (b_art, RDF.type, HOMEONT.Curtain) not in blinds
        for g, art in ((curtain, c_art), (blinds, b_art)):
            assert (art, RDF.type, HOMEONT.WindowCoveringController) in g

    def test_unknown_device_is_none_and_the_platform_is_titled_for_homebench(self, td):
        assert td.artifact("kitchen", "kitchen_toaster") is None
        platform = graph_of(td.platform())
        assert "HomeBench" in str(platform.value(td.platform_uri(), TD.title))


# -- the engine app and its client -----------------------------------------------------------------------------


@pytest.fixture()
def app_client(home):
    """The engine app with no home loaded, and a TestClient on it."""
    simulator._engine = None
    yield TestClient(simulator.app)
    simulator._engine = None


class TestEngineApp:
    def test_no_home_is_409_and_reset_loads_one(self, app_client, home):
        assert app_client.get("/api/home/state").status_code == 409
        assert app_client.get("/api/__health__").json()["data"] == {"ok": True}
        reply = app_client.post("/api/simulation/reset", json=home)
        body = reply.json()
        assert reply.status_code == 200 and body["error"] is None
        assert body["status"] == {"code": 200, "message": "OK"}
        assert set(body["data"]["rooms"]) == set(home["rooms"])

    def test_state_time_and_commands(self, app_client, home):
        app_client.post("/api/simulation/reset", json=home)
        assert app_client.get("/api/time").json()["data"] == {"now": home["base_time"]}
        commands = app_client.get("/api/devices/kitchen_light/commands").json()["data"]
        assert commands["commands"] == ["turn_off", "turn_on", "set_brightness"]
        attrs = app_client.get("/api/devices/kitchen_light/attributes").json()["data"]
        assert attrs["brightness"] == 10

    def test_apply_command_validates_like_the_engine(self, app_client, home):
        app_client.post("/api/simulation/reset", json=home)
        url = "/api/devices/kitchen_light/commands"
        ok = app_client.post(url, json={"command_id": "set_brightness", "args": {"value": 20}})
        assert ok.json()["data"] == {"device_id": "kitchen_light", "command": "set_brightness",
                                     "status": "applied", "attribute": "brightness",
                                     "value": 20}
        out_of_range = app_client.post(url, json={"command_id": "set_brightness",
                                                  "args": {"value": 101}})
        assert out_of_range.status_code == 400
        assert out_of_range.json()["status"]["code"] == 400 and out_of_range.json()["data"] is None
        assert app_client.post(url, json={"command_id": "dance"}).status_code == 400
        assert app_client.post(url, json={"args": {}}).status_code == 400
        assert app_client.post("/api/devices/nowhere/commands",
                               json={"command_id": "turn_on"}).status_code == 404
        # The refused calls left the accepted one in place.
        state = app_client.get("/api/home/state").json()["data"]
        light = next(d for d in state["rooms"]["kitchen"]["devices"]
                     if d["device_id"] == "kitchen_light")
        assert light["attributes"]["brightness"] == 20

    def test_unreachable_vacuum_command_is_reported_not_applied(self, app_client, home):
        app_client.post("/api/simulation/reset", json=home)
        reply = app_client.post("/api/devices/household_vacuum_robot/commands",
                                json={"command_id": "start"}).json()["data"]
        assert reply["status"] == "unreachable" and reply["attribute"] is None

    def test_reset_restores_the_home_and_rejects_a_bad_body(self, app_client, home):
        app_client.post("/api/simulation/reset", json=home)
        app_client.post("/api/devices/kitchen_light/commands",
                        json={"command_id": "turn_on"})
        again = app_client.post("/api/simulation/reset", json=home).json()["data"]
        light = next(d for d in again["rooms"]["kitchen"]["devices"]
                     if d["device_id"] == "kitchen_light")
        assert light["attributes"]["state"] == "off"
        assert app_client.post("/api/simulation/reset", json={"x": 1}).status_code == 400
        assert app_client.post("/api/simulation/reset", content=b"nope").status_code == 400

    def test_fast_forward_answers_the_state(self, app_client, home):
        app_client.post("/api/simulation/reset", json=home)
        reply = app_client.post("/api/simulation/fast_forward_to").json()["data"]
        assert reply["current_tick"] == 0 and "rooms" in reply


class TestClient_:
    """``HomeBenchClient`` sends HomeBench's calls over SimuHome's transport."""

    def test_methods_map_to_routes(self, monkeypatch):
        sent = []
        client = HomeBenchClient("http://x/api")
        monkeypatch.setattr(client, "_request",
                            lambda method, path, payload=None: sent.append(
                                (method, path, payload)) or {})
        client.execute_command("kitchen_light", "set_brightness", {"value": 5})
        client.write_attribute("kitchen_light", "brightness", 7)
        client.execute_command("kitchen_light", "turn_on")
        client.device_commands("kitchen_light")
        assert sent == [
            ("POST", "/devices/kitchen_light/commands",
             {"command_id": "set_brightness", "args": {"value": 5}}),
            ("POST", "/devices/kitchen_light/commands",
             {"command_id": "set_brightness", "args": {"value": 7}}),
            ("POST", "/devices/kitchen_light/commands",
             {"command_id": "turn_on", "args": {}}),
            ("GET", "/devices/kitchen_light/commands", None),
        ]

    def test_it_is_the_simuhome_client_with_the_same_public_methods(self):
        from ami_agents.environment.integration.SimuHome.sim_client import SimuHomeClient
        for name in ("health", "reset", "home_state", "current_time", "execute_command",
                     "write_attribute", "reset_simulation", "fast_forward_to"):
            assert callable(getattr(HomeBenchClient, name))
            assert hasattr(SimuHomeClient, name)

    def test_an_error_envelope_raises(self):
        with pytest.raises(SimuHomeError):
            HomeBenchClient._unwrap(
                {"status": {"code": 400, "message": "Bad Request"}, "data": None,
                 "error": "x"}, "POST /x")


# -- HBTD ----------------------------------------------------------------------------------------------------------


class StubClient:
    """What HBTD needs of ``HomeBenchClient``, over an in-process engine."""

    base_url = "http://stub/api"

    def __init__(self, machine):
        self.machine = machine
        self.calls = []

    def home_state(self):
        return self.machine.home_state()

    def current_time(self):
        return {"now": self.machine.home_state()["current_time"]}

    def execute_command(self, device_id, command_id, args=None):
        self.calls.append((device_id, command_id, args))
        try:
            result = self.machine.apply(device_id, command_id, (args or {}).get("value"))
        except EngineError as exc:
            raise SimuHomeError(str(exc)) from exc
        return {"status": result.status}


@pytest.fixture()
def server(machine, monkeypatch):
    """HBTD in-process over a stub client; yields ``(TestClient, stub)``."""
    stub = StubClient(machine)
    monkeypatch.setattr(hbtd, "_client", stub)
    monkeypatch.setattr(hbtd, "_home_id", HOME_ID)
    monkeypatch.setattr(hbtd, "_base_url", "http://testserver")
    return TestClient(hbtd.app), stub


def action(room, device, name):
    """The action URL of a device."""
    return f"/workspaces/{HOME_ID}/{room}/artifacts/{device}/actions/{name}"


def prop(room, device, name):
    """The property URL of a device."""
    return f"/workspaces/{HOME_ID}/{room}/artifacts/{device}/properties/{name}"


class TestHBTD:
    def test_discovery_routes_serve_turtle(self, server):
        client, _stub = server
        for path in ("/", "/workspaces", f"/workspaces/{HOME_ID}",
                     f"/workspaces/{HOME_ID}/kitchen", f"/workspaces/{HOME_ID}/kitchen/artifacts",
                     f"/workspaces/{HOME_ID}/kitchen/artifacts/kitchen_light"):
            reply = client.get(path)
            assert reply.status_code == 200 and reply.headers["content-type"].startswith(
                "text/turtle"), path
            assert Graph().parse(data=reply.text, format="turtle")
        assert client.get("/workspaces/otherhome").status_code == 404
        assert client.get(f"/workspaces/{HOME_ID}/attic").status_code == 404
        assert client.get(f"/workspaces/{HOME_ID}/kitchen/artifacts/ghost").status_code == 404

    def test_property_reads_are_bare_json_in_the_schemas_form(self, server):
        client, _stub = server
        assert client.get(prop("kitchen", "kitchen_light", "onOff")).json() is False
        assert client.get(prop("living_room", "living_room_light", "onOff")).json() is True
        assert client.get(prop("kitchen", "kitchen_light", "brightness")).json() == 10
        assert client.get(prop("living_room", "living_room_light", "color")).json() == [1, 2, 3]
        assert client.get(prop("living_room", "living_room_curtain", "openClose")).json() == "open"
        assert client.get(prop("household", "household_vacuum_robot", "state")).json() == "on"
        # cleaning_area starts as null in the home file.
        assert client.get(prop("household", "household_vacuum_robot",
                               "cleaningArea")).json() is None
        assert client.get(prop("bathroom", "bathroom_light", "brightness")).status_code == 404

    def test_a_valid_invocation_moves_the_engine_and_acknowledges(self, server, machine):
        client, stub = server
        reply = client.post(action("kitchen", "kitchen_light", "brightness"), json={"value": 20})
        assert reply.status_code == 200
        assert reply.json()["dispatched"] == {
            "call": "execute_command", "command": "set_brightness", "args": {"value": 20}}
        assert machine.device("kitchen_light")["attributes"]["brightness"] == 20
        client.post(action("kitchen", "kitchen_light", "onOff"), json={"value": True})
        assert machine.device("kitchen_light")["attributes"]["state"] == "on"
        client.post(action("living_room", "living_room_trash", "pack"), json={})
        assert machine.device("living_room_trash")["attributes"]["state"] == "empty"
        # `input` and `payload` carry the value as they do for SHTD.
        client.post(action("kitchen", "kitchen_light", "brightness"), json={"input": 30})
        assert machine.device("kitchen_light")["attributes"]["brightness"] == 30
        assert [c[1] for c in stub.calls] == [
            "set_brightness", "turn_on", "pack", "set_brightness"]

    @pytest.mark.parametrize("room,device,name,body,status", [
        # missing value
        ("kitchen", "kitchen_light", "onOff", {}, 400),
        ("kitchen", "kitchen_light", "onOff", {"value": None}, 400),
        ("kitchen", "kitchen_light", "brightness", {}, 400),
        ("living_room", "living_room_air_conditioner", "mode", {}, 400),
        # wrongly typed value
        ("kitchen", "kitchen_light", "onOff", {"value": "on"}, 400),
        ("kitchen", "kitchen_light", "onOff", {"value": 1}, 400),
        ("kitchen", "kitchen_light", "brightness", {"value": "20"}, 400),
        ("kitchen", "kitchen_light", "brightness", {"value": True}, 400),
        ("kitchen", "kitchen_light", "brightness", {"value": 20.5}, 400),
        ("living_room", "living_room_curtain", "openClose", {"value": 5}, 400),
        ("living_room", "living_room_light", "color", {"value": [1, 2]}, 400),
        ("living_room", "living_room_light", "color", {"value": "red"}, 400),
        ("living_room", "living_room_trash", "pack", {"value": True}, 400),
        # well typed, refused by the device
        ("kitchen", "kitchen_light", "brightness", {"value": 101}, 422),
        ("living_room", "living_room_curtain", "openClose", {"value": "ajar"}, 422),
        ("living_room", "living_room_air_conditioner", "mode", {"value": "frost"}, 422),
        ("living_room", "living_room_heating", "temperature", {"value": 80}, 422),
        ("living_room", "living_room_light", "color", {"value": [1, 2, 300]}, 422),
        # no such thing
        ("kitchen", "kitchen_light", "swing", {"value": "up"}, 404),
        ("bathroom", "bathroom_light", "brightness", {"value": 5}, 404),
        ("household", "household_vacuum_robot", "state", {"value": "off"}, 404),
        ("kitchen", "kitchen_ghost", "onOff", {"value": True}, 404),
    ])
    def test_bad_invocations_are_rejected_and_change_nothing(
            self, server, machine, room, device, name, body, status):
        client, stub = server
        before = machine.home_state()
        reply = client.post(action(room, device, name), json=body)
        assert reply.status_code == status, reply.text
        assert machine.home_state() == before

    def test_a_missing_on_off_value_does_not_switch_the_device_off(self, server, machine):
        """The SHTD defect: ``_truthy(None)`` is False, so a bare POST powers it off."""
        client, _stub = server
        assert machine.device("living_room_light")["attributes"]["state"] == "on"
        assert client.post(action("living_room", "living_room_light", "onOff"),
                           json={}).status_code == 400
        assert client.post(action("living_room", "living_room_light", "onOff"),
                           content=b"not json").status_code == 400
        assert machine.device("living_room_light")["attributes"]["state"] == "on"

    def test_dispatch_returns_the_call_without_executing_it(self, server, machine):
        client, stub = server
        reply = client.post(action("kitchen", "kitchen_light", "brightness") + "/dispatch",
                            json={"value": 55})
        assert reply.json() == {"tool": "execute_command", "args": {
            "device_id": "kitchen_light", "command_id": "set_brightness",
            "args": {"value": 55}}}
        assert stub.calls == []
        assert machine.device("kitchen_light")["attributes"]["brightness"] == 10

    def test_hub_refuses_unknown_topics_and_bad_modes(self, server):
        client, _stub = server
        body = {"hub.mode": "subscribe", "hub.topic": "http://testserver/workspaces/x",
                "hub.callback": "http://localhost:1/webhook"}
        assert client.post("/hub/", json=body).status_code == 404
        assert client.post("/hub/", json={**body, "hub.mode": "peek"}).status_code == 400
        assert client.post("/hub/", json={"hub.mode": "subscribe"}).status_code == 400
        assert client.get("/_shtd/subscriptions").json()["count"] == 0

    def test_status_and_time(self, server, machine):
        client, _stub = server
        status = client.get("/_shtd/status").json()
        assert status["devices"] == len(machine.device_ids())
        assert status["unsettledMappingRows"] == {} and "phase" not in status
        assert client.get("/_shtd/time").json() == {"now": machine.home_state()["current_time"]}

    def test_the_payload_carries_property_uri_and_value(self):
        home_topic = "http://127.0.0.1:9/workspaces/home7"
        payload = hbtd.change_payload(home_topic, "kitchen", "kitchen_light", "light",
                                      "state", "on", "off", "2025-01-01 08:00:00")
        base = f"{home_topic}/kitchen/artifacts/kitchen_light"
        assert payload["artifactUri"] == f"{base}#artifact"
        assert payload["propertyUri"] == f"{base}/properties/onOff"
        assert payload["value"] is True and payload["previous"] is False
        assert payload["property"] == "onOff" and payload["attributes"] == {"onOff": True}
        brightness = hbtd.change_payload(home_topic, "kitchen", "kitchen_light", "light",
                                         "brightness", 30, 10, None)
        assert brightness["propertyUri"].endswith("/properties/brightness")
        assert brightness["value"] == 30 and brightness["timestamp"] == ""
        assert hbtd.change_payload(home_topic, "kitchen", "kitchen_light", "light",
                                   "nonsense", 1, 0, None) is None

    def test_the_watcher_reports_what_the_engine_changes(self, machine):
        """The diff the payloads are made from: only the attribute that moved."""
        from ami_agents.environment.integration.SimuHome.notifications import StateWatcher
        watcher = StateWatcher(machine.home_state)
        assert watcher.poll() == []                        # primes the baseline
        machine.apply("kitchen_light", "set_brightness", 99)
        changes = watcher.poll()
        assert [(c["device_id"], c["path"], c["value"], c["previous"]) for c in changes] == [
            ("kitchen_light", "brightness", 99, 10)]


# -- verify_phase2 helpers --------------------------------------------------------------------------------------


class TestVerifyHelpers:
    def test_sample_values_come_from_the_input_schema(self):
        from types import SimpleNamespace
        values = lambda schema: verify_phase2.sample_values(  # noqa: E731
            SimpleNamespace(input_schema=schema))
        assert values(None) == [None]
        assert values({"type": JS + "BooleanSchema"}) == [True, False]
        assert values({"type": JS + "StringSchema", "enum": ["a", "b"]}) == ["a", "b"]
        assert values({"type": JS + "IntegerSchema", "minimum": 16.0, "maximum": 30.0}) == [
            16, 23, 30]
        assert values({"type": JS + "ArraySchema"}) == [[10, 20, 30]]

    def test_the_gold_round_trip_catches_a_broken_table(self, monkeypatch):
        rows = [("''' kitchen.light.set_brightness(20)'''", False),
                ("'''error_input,living_room.curtain.close(),'''", True),
                ("'''vacuum_robot.start()'''", False)]
        report = verify_phase2.gold_round_trip(rows)
        assert report["calls"] == 3 and report["unreachable"] == 1
        assert report["failures"] == []
        # Break the table: the template writes the wrong op.
        m = load_mappings()
        broken = copy.deepcopy(m._commands[("light", "brightness")])
        broken["gold"] = "set_brightnezz({value})"
        monkeypatch.setitem(m._commands, ("light", "brightness"), broken)
        assert verify_phase2.gold_round_trip(rows)["failures"]

    @needs_data
    def test_every_gold_call_of_the_dataset_round_trips(self):
        rows = [(r["output"], r["type"].startswith("multi"))
                for r in convert.read_jsonl(HOMEBENCH / convert.TEST_DATA)]
        report = verify_phase2.gold_round_trip(rows)
        assert report["failures"] == []
        assert report["calls"] == 18832 and report["unreachable"] == 70
        # A sample of every op the gold uses, not only the common ones.
        assert len(report["ops"]) >= 50
