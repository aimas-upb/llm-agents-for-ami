#!/usr/bin/env python3
"""Render a HomeBench home as HMAS/TD Turtle: ``HomeBenchTD(SimuHomeTD)``.

Everything above the device is SimuHome's and is inherited: platform, home and room
workspaces, URL minting, places, the WebSub affordances, serialisation. What this
subclass replaces is the Matter-specific part: ``artifact()`` and the attribute
handling. A HomeBench device is ``{device_id, device_type, attributes, constraints}``
with a flat attribute map, so each attribute becomes one property affordance and each
settable one also one action affordance, as SHTD does for Matter attributes:

* a ``td:PropertyAffordance`` named after the attribute (``brightness``), typed with a
  homeont property class, with a GET form and an output schema;
* for a settable attribute, a ``td:ActionAffordance`` of the same ``td:name`` typed
  with a homeont / SAREF command class (``saref:SetAbsoluteLevelCommand``), with a POST
  form and an input schema, ``saref:actsUpon`` its property.

The state of a device is one affordance of its kind: ``onOff`` (a boolean, shown as
``true``/``false`` for the home's ``on``/``off``), ``openClose``, ``playback`` or
``pack``. Planners resolve goals by the command *class*, never by the action name, so
one ``onOff`` action stands for ``turn_on`` and ``turn_off``; the gold call is a pure
function of ``(device, action, value)`` through the ``command_map`` table.

The environment model is SimuHome's and HASP's, declared and not simulated. With
``environment=True`` (the default) a room whose devices can affect an ambient variable
(``EFFECT_FAMILIES``: an air conditioner, a light, ...) serves one ``#environment``
owned by the room's place (``homeont:hasEnvironment``) carrying those variables
(``ssn:hasProperty``); every action of such a device carries a directionless TD-SOSA
effect on its room's variable, and the device is a ``saref:Actuator`` with the
environment as its ``sosa:hasFeatureOfInterest``. HomeBench has no room values and no
physics, so each variable is ``td:isObservable false`` with no read form, no ambient
sensor exists, and the engine never simulates an effect. ``environment=False`` serves
the flat graph (no environment, no effects, no ``saref:Actuator``) that matches the
ECAI evaluation. See ``README.md``.

Worked example. Home ``home75`` has ``bathroom_light`` (``state: on``, ``brightness:
40``): ``artifact("bathroom", "bathroom_light")`` is a ``homeont:DimmableLight`` with
properties ``onOff`` (reads ``true``) and ``brightness`` (reads ``40``, schema integer
0-100) and actions ``onOff`` (boolean, ``homeont:SetOnOffCommand``) and ``brightness``
(integer, ``saref:SetAbsoluteLevelCommand``). With the environment on, ``bathroom``
also has an environment with ``illuminance`` (not observable), the light is a
``saref:Actuator`` and both of its actions have a ``tdsosa:hasEffectActuation`` that
``sosa:actsOnProperty`` ``bathroom#illuminance``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from rdflib import BNode, Graph, Literal, RDF, URIRef
from rdflib.namespace import RDFS, XSD

from ami_agents.environment.integration.HomeBench.mappings import (
    Mappings, load_mappings,
)
from ami_agents.environment.integration.SimuHome.td_builder import (
    HCTL, HMAS, HOME, HTV, JS, SAREF, SOSA, TD, TDSOSA, SimuHomeTD, _add_order, _add_unit,
    _literal, _new_graph, _no_security, curie_to_uri,
)

LOGGER = logging.getLogger(__name__)

# -- constants ------------------------------------------------------------------------

PLATFORM_TITLE = "HomeBench Platform"

#: The values of the ``--environment`` switch of HBTD, the worker and the bench, and the
#: default: ``on`` serves the environment model, ``off`` the flat graph.
ENVIRONMENT_MODES = ("on", "off")
DEFAULT_ENVIRONMENT = "on"


class HomeBenchTD(SimuHomeTD):
    """Render one loaded HomeBench home as HMAS/TD Turtle.

    ``state`` is what the engine's ``GET /api/home/state`` returns; ``home`` is the
    home id used in every URL (``home<id>``). Built per request, like ``SimuHomeTD``.
    """

    def __init__(self, base: str, home: str, state: Dict[str, Any],
                 environment: bool = True):
        """Set the fields the inherited builders read; ``self.m`` is the HomeBench tables.

        ``environment`` switches the environment model on (the default) or serves the
        flat graph: no environment resource, no effects, no ``saref:Actuator``.

        ``SimuHomeTD.__init__`` is not called: all it does is store these four fields
        and load SimuHome's Matter tables (13k lines of YAML), which a HomeBench
        process has no use for.
        """
        self.base = base.rstrip("/") + "/"
        self.home = home
        self.environment = environment
        self.state = state or {}
        self.rooms: Dict[str, Any] = self.state.get("rooms") or {}
        self.m: Mappings = load_mappings()  # type: ignore[assignment]

    # -- documents inherited, with HomeBench wording or without an environment ----------

    def platform(self) -> Graph:
        """The platform document, titled for HomeBench instead of SimuHome."""
        g = super().platform()
        platform = self.platform_uri()
        g.remove((platform, TD.title, None))
        g.add((platform, TD.title, Literal(PLATFORM_TITLE)))
        return g

    def home_workspace(self) -> Graph:
        """The home workspace, its comment naming the HomeBench home."""
        g = super().home_workspace()
        ws = URIRef(f"{self.home_path()}#workspace")
        g.remove((ws, RDFS.comment, None))
        g.add((ws, RDFS.comment, Literal(
            f"HomeBench home '{self.home}': {len(self.rooms)} rooms, "
            f"{self.device_count()} devices.")))
        return g

    def room_workspace(self, room_id: str) -> Graph:
        """The room workspace with the variables its devices can affect, or without
        any environment resource.

        The inherited document mints a ``homeont:Environment`` for every room but fills
        it from the room's ``state`` (empty in HomeBench). With the environment on:

        * a room with at least one variable keeps its environment, and the variables
          are added declared-only (``readable=False``: not observable, no form);
        * a room with none (its devices affect nothing, 56 of 1,256 rooms) would have
          an empty promise, so the resource is removed with every triple pointing at it.

        With the environment off the resource is removed from every room.
        """
        g = super().room_workspace(room_id)
        foi = self.foi_uri(room_id)
        # 1. Which variables this room has (none with the environment off).
        tokens = (self.m.room_variables(self.rooms.get(room_id) or {})
                  if self.environment else [])
        # 2. No variable: no environment resource at all.
        if not tokens:
            g.remove((foi, None, None))
            g.remove((None, None, foi))
            return g
        # 3. Declare the variables: only the keys of ``state`` are read, so the values
        #    are None (HomeBench has none).
        self._room_state_properties(g, room_id, {t: None for t in tokens}, foi,
                                    readable=False)
        return g

    # -- the device -------------------------------------------------------------------

    def artifact(self, room_id: str, device_id: str) -> Optional[Graph]:
        """One device's Thing Description, or ``None`` for an unknown device."""
        device = self.find_device(room_id, device_id)
        if device is None:
            return None
        # 1. The Thing, its containment chain and its security scheme.
        g = _new_graph()
        path = self.artifact_path(room_id, device_id)
        art = URIRef(f"{path}#artifact")
        attributes = device.get("attributes") or {}
        device_type = str(device.get("device_type") or "")
        g.add((art, RDF.type, TD.Thing))
        g.add((art, RDF.type, HMAS.Artifact))
        g.add((art, TD.title, Literal(device_id)))
        self._containment(g, art, room_id)
        _no_security(g, art)
        # 2. Its homeont class (a light by what it can do) and where it is.
        family = self.m.device_family(device_type, attributes) or {}
        for curie in family.get("td_types") or []:
            uri = curie_to_uri(curie)
            if uri is not None:
                g.add((art, RDF.type, uri))
        # A device that can affect a room variable is an actuator on that room's
        # environment (an environment exists for every such room).
        if self.environment and self.m.affects_environment(device_type):
            g.add((art, RDF.type, SAREF.Actuator))
            g.add((art, SOSA.hasFeatureOfInterest, self.foi_uri(room_id)))
        if family.get("title"):
            g.add((art, RDFS.label, Literal(family["title"])))
        place = self.place_uri(room_id)
        g.add((art, HOME.isLocatedIn, place))
        g.add((place, HOME.containsArtifact, art))
        room_class = self._room_class(room_id)
        if room_class is not None:
            g.add((place, RDF.type, room_class))
        g.add((place, RDF.type, HOME.BuildingSpace))
        # 3. One property per attribute, one action per settable attribute.
        for row in self.m.affordances(device_type, attributes):
            prop = self._property(g, art, path, row, device)
            if row["actuatable"]:
                self._action(g, art, path, row, device, acts_upon=prop,
                             room_id=room_id)
        # 4. Subscription affordances and the resource profile.
        self._subscribe_actions(g, art, f"{path}#artifact", artifact=True)
        profile = URIRef(path)
        g.add((profile, RDF.type, HMAS.ResourceProfile))
        g.add((profile, HMAS.isProfileOf, art))
        return g

    def _containment(self, g: Graph, art: URIRef, room_id: str) -> None:
        """Type and name the room and home inline, both directions, so an agent
        holding only this TD can walk back up to the room and the home."""
        room_ws = URIRef(f"{self.room_path(room_id)}#workspace")
        g.add((art, HMAS.isContainedIn, room_ws))
        g.add((room_ws, RDF.type, HMAS.Workspace))
        g.add((room_ws, TD.title, Literal(self._room_title(room_id))))
        g.add((room_ws, HMAS.contains, art))
        home_ws = URIRef(f"{self.home_path()}#workspace")
        g.add((room_ws, HMAS.isContainedIn, home_ws))
        g.add((home_ws, RDF.type, HMAS.Workspace))
        g.add((home_ws, TD.title, Literal(self.home)))
        g.add((home_ws, HMAS.contains, room_ws))

    # -- affordances --------------------------------------------------------------------

    def _property(self, g: Graph, art: URIRef, path: str, row: Dict[str, Any],
                  device: Dict[str, Any]) -> BNode:
        """The property affordance of one attribute: class, output schema, GET form."""
        name = str(row["affordance_name"])
        prop = BNode()
        g.add((art, TD.hasPropertyAffordance, prop))
        g.add((prop, RDF.type, TD.PropertyAffordance))
        g.add((prop, TD.name, Literal(name)))
        g.add((prop, TD.title, Literal(name)))
        g.add((prop, TD.isObservable, Literal(True, datatype=XSD.boolean)))
        # What the property IS: the TD-SOSA marker by kind, and the homeont class.
        if self.m.property_kind(str(row["property_type"])) == "actuatable":
            g.add((prop, RDF.type, TDSOSA.ActuatablePropertyAffordance))
        else:
            g.add((prop, RDF.type, TDSOSA.ObservablePropertyAffordance))
        uri = curie_to_uri(str(row["property_type"]))
        if uri is not None:
            g.add((prop, RDF.type, uri))
        g.add((prop, TD.hasOutputSchema, self._schema(g, row, device, action=False)))
        form = BNode()
        g.add((prop, TD.hasForm, form))
        g.add((form, HTV.methodName, Literal("GET")))
        g.add((form, HCTL.hasTarget, URIRef(f"{path}/properties/{name}")))
        g.add((form, HCTL.forContentType, Literal("application/json")))
        g.add((form, HCTL.hasOperationType, TD.readProperty))
        return prop

    def _action(self, g: Graph, art: URIRef, path: str, row: Dict[str, Any],
                device: Dict[str, Any], acts_upon: BNode, room_id: str) -> None:
        """The action affordance of a settable attribute: command class, input schema
        (none for ``pack``), POST form, ``saref:actsUpon`` its property.

        With the environment on, also the TD-SOSA effect the ``actuation_effects``
        table claims for this ``(device type, attribute)``: directionless, declared
        domain semantics that the engine does not simulate. Nothing else is written
        for an action with no row.
        """
        name = str(row["affordance_name"])
        action = BNode()
        g.add((art, TD.hasActionAffordance, action))
        g.add((action, RDF.type, TD.ActionAffordance))
        g.add((action, TD.name, Literal(name)))
        g.add((action, TD.title, Literal(name)))
        g.add((action, TD.isIdempotent, Literal(True, datatype=XSD.boolean)))
        # Invoking it changes the world, so it is not safe in the WoT sense.
        g.add((action, TD.isSafe, Literal(False, datatype=XSD.boolean)))
        uri = curie_to_uri(str(row["action_class"]))
        if uri is not None:
            g.add((action, RDF.type, uri))
        g.add((action, SAREF.actsUpon, acts_upon))
        if row["value_kind"] != "none":
            g.add((action, TD.hasInputSchema,
                   self._schema(g, row, device, action=True)))
        form = BNode()
        g.add((action, TD.hasForm, form))
        g.add((form, HTV.methodName, Literal("POST")))
        g.add((form, HCTL.hasTarget, URIRef(f"{path}/actions/{name}")))
        g.add((form, HCTL.forContentType, Literal("application/json")))
        g.add((form, HCTL.hasOperationType, TD.invokeAction))
        # The effect on the room's variable, from the table (none when the flat graph
        # is served or the attribute affects nothing).
        if self.environment:
            claims = self.m.effects(str(device.get("device_type") or ""),
                                    str(row["attribute"]))
            if claims:
                self._write_effects(g, action, room_id, claims)

    # -- value schemas ------------------------------------------------------------------

    @staticmethod
    def schema_kind(row: Dict[str, Any], action: bool) -> str:
        """The shape of the value an affordance carries: ``boolean``, ``enum``,
        ``integer``, ``rgb`` or ``none``.

        An action takes ``value_kind``. A read differs in two cases: a row with a
        ``read_map`` reads a boolean (the home's ``on`` shows as ``true``), and the
        ``pack`` state (an action without input) reads the enum ``read_values``.
        """
        kind = str(row["value_kind"])
        if action:
            return kind
        if row.get("read_map") is not None:
            return "boolean"
        if kind == "none":
            return "enum"
        return kind

    @staticmethod
    def options(row: Dict[str, Any], device: Dict[str, Any], action: bool) -> List[Any]:
        """The values an enum affordance allows: the row's, else this device's own
        ``constraints[attribute].options`` (a per-instance fact, like a range)."""
        if not action and row.get("read_values"):
            return list(row["read_values"])
        if row.get("values"):
            return list(row["values"])
        rule = (device.get("constraints") or {}).get(str(row["attribute"])) or {}
        return list(rule.get("options") or [])

    def _schema(self, g: Graph, row: Dict[str, Any], device: Dict[str, Any],
                action: bool) -> BNode:
        """The SHAPE of the value (never a value): type, unit, range or enum.

        Bounds are this device's own, from its ``constraints`` (heating temperature
        already corrected to 16-30 in the home file). Enum values are listed both as
        ``js:enum`` and in ``js:description``, which is what the agents read, followed by
        the row's own ``description`` where it has one; an enum
        whose row has a ``value_order`` also states that scale, restricted to this
        device's options (``homeont:orderedValues``).
        """
        schema = BNode()
        kind = self.schema_kind(row, action)
        # 1. Type, and the restriction the kind carries.
        if kind == "boolean":
            g.add((schema, RDF.type, JS.BooleanSchema))
        elif kind == "integer":
            g.add((schema, RDF.type, JS.IntegerSchema))
            rule = (device.get("constraints") or {}).get(str(row["attribute"])) or {}
            if "lowest" in rule and "highest" in rule:
                g.add((schema, JS.minimum, _literal(rule["lowest"])))
                g.add((schema, JS.maximum, _literal(rule["highest"])))
        elif kind == "rgb":
            self._rgb_schema(g, schema, device, str(row["attribute"]))
        else:
            g.add((schema, RDF.type, JS.StringSchema))
            options = self.options(row, device, action)
            for option in options:
                g.add((schema, JS.enum, Literal(str(option))))
            if options:
                g.add((schema, JS.description, Literal(
                    "Values on this device: " + ", ".join(str(o) for o in options) + ".")))
            # The scale among them, lowest first, where the attribute has one.
            _add_order(g, schema, [v for v in row.get("value_order") or []
                                   if v in options])
        # 2. The unit, as a QUDT IRI (only where the table states one).
        if row.get("unit"):
            _add_unit(g, schema, row["unit"])
        # 3. What the value means, where the table says (a curtain's "degree" is a
        #    percentage), after any description of its values: one js:description.
        if row.get("description"):
            listed = g.value(schema, JS.description)
            g.remove((schema, JS.description, None))
            g.add((schema, JS.description, Literal(
                " ".join(str(s) for s in (listed, row["description"]) if s))))
        return schema

    @staticmethod
    def _rgb_schema(g: Graph, schema: BNode, device: Dict[str, Any],
                    attribute: str) -> None:
        """A colour: an array of exactly three integers within the device's channel range."""
        rule = (device.get("constraints") or {}).get(attribute) or {}
        low, high = rule.get("lowest", 0), rule.get("highest", 255)
        g.add((schema, RDF.type, JS.ArraySchema))
        item = BNode()
        g.add((schema, JS.items, item))
        g.add((item, RDF.type, JS.IntegerSchema))
        g.add((item, JS.minimum, _literal(low)))
        g.add((item, JS.maximum, _literal(high)))
        g.add((schema, JS.minItems, _literal(3)))
        g.add((schema, JS.maxItems, _literal(3)))
        g.add((schema, JS.description, Literal(
            f"[red, green, blue], each an integer from {low} to {high}.")))
