"""Home Assistant -> home ontology: which homeont / SAREF class a thing is,
and which room variable it observes or changes.

HASP's TDs carry the same semantic layer SimuHome's do -- a device class on the
Thing, a property class on each property affordance, a command class on each
action, a room class on the workspace's place, and the room's environment with
its variables -- so the agents' class-based resolvers work on a Home Assistant
home unchanged.

The classes come from `semantic_mappings.yaml` (defaults, by HA domain /
service / attribute / device_class / unit), overridden per device by the
`semantic:` block of the home's own config (the file `SEMANTIC_CONFIG` points
at). Pure functions and data: no HA connection, no FastAPI, testable on their
own.

Environment variables are not listed anywhere: homeont says which of its
classes are a property of a space's environment (`ssn:isPropertyOf
homeont:Environment`), and a device property typed with one of them is an
observation of that variable. Its name is the class's, in snake_case, as
SimuHome names its room variables. Example, the lab308e temperature sensor:

    HA entity     sensor.temperature_sensing_308e, device_class temperature
    property      state -> homeont:AirTemperature      (`properties` rules)
    variable      homeont:AirTemperature is an environment class -> name
                  `air_temperature`, IRI <base>workspaces/lab308e#air_temperature,
                  qudt quantity kind and unit from homeont
    emitted       the sensor `sosa:observes` it; the room environment
                  `ssn:hasProperty` it (`add_variable`, workspace document)

A home's `semantic:` block adds only what HA cannot know: a `zone` (a second
variable of the same class, `desk_illuminance`), another `environment` (the
outdoor light), an action's `effects`, and comfort `ranges`.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

logger = logging.getLogger("hasp.semantics")

HERE = Path(__file__).resolve().parent
MAPPINGS_PATH = HERE / "semantic_mappings.yaml"
# .../ami_agents/environment/integration/HomeAssistant -> .../ami_agents
HOMEONT_PATH = Path(os.getenv(
    "HOMEONT_PATH", str(HERE.parents[2] / "shared" / "ontologies" / "homeont.ttl")))

# The adapter's one namespace module; re-exported, since hasp.py and
# hasp_cache.py read `hasp_semantics.SAREF` / `.HOMEONT`.
try:
    from .hasp_namespaces import (HCTL, HOMEONT, HTV, JS, QUANTITYKIND, QUDT, SAREF,
                                  SOSA, SSN, TD, TDSOSA, UNIT)
except ImportError:  # pragma: no cover - run from this directory
    from hasp_namespaces import (HCTL, HOMEONT, HTV, JS, QUANTITYKIND, QUDT, SAREF,
                                 SOSA, SSN, TD, TDSOSA, UNIT)

_PREFIXES = {
    "homeont": str(HOMEONT),
    "saref": str(SAREF),
    "sosa": str(SOSA),
    "unit": str(UNIT),
    "quantitykind": str(QUANTITYKIND),
}

# The environment of the room a workspace is; any other environment a home
# declares (`semantic: environments:`) is named by its own key.
ROOM_ENVIRONMENT = "room"

# How an action's service name says which way it moves a variable, when no
# rule states it. Setters move it either way -- the value decides -- so they
# get an effect without a direction.
_INCREASE_MARKERS = ("turn_on", "open", "raise", "increase", "up", "unlock", "start")
_DECREASE_MARKERS = ("turn_off", "close", "lower", "decrease", "down", "lock", "stop")
SETPOINT_SERVICES = {"set_temperature", "set_hvac_mode", "set_percentage", "set_speed",
                     "set_humidity", "set_cover_position", "set_position", "set_value"}
DIRECTIONS = ("increase", "decrease")

# Colour modes that mean a light can be dimmed / can change colour.
_DIM_MODES = {"brightness", "color_temp", "hs", "xy", "rgb", "rgbw", "rgbww", "white"}
_COLOR_MODES = {"hs", "xy", "rgb", "rgbw", "rgbww"}


def expand(curie: Optional[str]) -> Optional[URIRef]:
    """`homeont:Kitchen` -> its IRI; None for None or an unknown prefix."""
    if not curie:
        return None
    prefix, _, local = str(curie).partition(":")
    namespace = _PREFIXES.get(prefix)
    return URIRef(namespace + local) if namespace and local else None


def _as_set(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return {str(value)}


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def snake_case(local_name: str) -> str:
    """`AirTemperature` -> `air_temperature`, `Pm10MassConcentration` ->
    `pm10_mass_concentration`: SimuHome's room-variable names, which are its
    homeont classes' local names in snake_case."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", local_name).lower()


def service_direction(service: str) -> Optional[str]:
    """`increase` / `decrease` from a service's name (`turn_on`,
    `close_cover`), or None when the name says neither (setters, toggles)."""
    name = service.lower()
    if any(marker in name for marker in _INCREASE_MARKERS):
        return "increase"
    if any(marker in name for marker in _DECREASE_MARKERS):
        return "decrease"
    return None


@dataclass(frozen=True)
class EnvironmentClass:
    """A homeont class that is a property of a space's environment, with
    what homeont states about it."""

    curie: str                               # homeont:AirTemperature
    name: str                                # air_temperature
    label: Optional[str] = None
    quantity_kind: Optional[URIRef] = None   # quantitykind:ThermodynamicTemperature
    unit: Optional[URIRef] = None            # unit:DEG_C


@dataclass(frozen=True)
class Variable:
    """One environment variable of one workspace: its class, its name there
    (`desk_illuminance` for a zone), and the environment it belongs to."""

    cls: EnvironmentClass
    name: str
    environment: str = ROOM_ENVIRONMENT


def light_supports(attributes: Dict[str, Any]) -> Set[str]:
    """What a light can do beyond on/off: `brightness`, `color`.

    From `supported_color_modes`, which HA reports whether or not the light is
    on -- unlike `brightness`, which disappears while it is off.
    """
    modes = _as_set(attributes.get("supported_color_modes"))
    supports: Set[str] = set()
    if modes & _DIM_MODES:
        supports.add("brightness")
    if modes & _COLOR_MODES:
        supports.add("color")
    return supports


@dataclass
class Entity:
    """The facts a rule may test about one HA entity."""

    domain: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def device_class(self) -> Optional[str]:
        return self.attributes.get("device_class")

    @property
    def unit(self) -> Optional[str]:
        return self.attributes.get("unit_of_measurement")


def _matches(rule: Dict[str, Any], entity: Entity, *,
             device: Optional[str] = None, signal: Optional[str] = None,
             service: Optional[str] = None) -> bool:
    if "domain" in rule and entity.domain not in _as_set(rule["domain"]):
        return False
    if "device_class" in rule and entity.device_class not in _as_set(rule["device_class"]):
        return False
    if "unit" in rule and entity.unit not in _as_set(rule["unit"]):
        return False
    if "supports" in rule and not (_as_set(rule["supports"])
                                   <= light_supports(entity.attributes)):
        return False
    if "hvac_modes_include" in rule and not (
            _as_set(rule["hvac_modes_include"])
            <= _as_set(entity.attributes.get("hvac_modes"))):
        return False
    if "device" in rule and device not in _as_set(rule["device"]):
        return False
    if "signal" in rule and signal not in _as_set(rule["signal"]):
        return False
    if "service" in rule and service not in _as_set(rule["service"]):
        return False
    return True


class Semantics:
    """The mapping defaults plus one home's overrides."""

    def __init__(self, mappings: Dict[str, Any],
                 home: Optional[Dict[str, Any]] = None,
                 room_labels: Optional[Dict[str, str]] = None,
                 class_labels: Optional[Dict[str, str]] = None,
                 environment_classes: Optional[Dict[str, EnvironmentClass]] = None) -> None:
        self.mappings = mappings or {}
        self.home = home or {}
        self.room_labels = room_labels or {}
        self.class_labels = class_labels or {}
        # homeont's environment classes, by CURIE (`_environment_classes`).
        self.environment_classes = environment_classes or {}
        self._reported: Set[str] = set()

    def label(self, curie: Optional[str]) -> Optional[str]:
        """A class's `rdfs:label` in homeont ("Color Light"), or None."""
        return self.class_labels.get(curie or "")

    # -- overrides -------------------------------------------------------

    def _device_override(self, device_name: Optional[str]) -> Dict[str, Any]:
        devices = self.home.get("devices") or {}
        return devices.get(device_name or "") or {}

    def _unmapped(self, key: str) -> None:
        """Log a gap once, so missing mappings are visible but not noisy."""
        if key not in self._reported:
            self._reported.add(key)
            logger.info("No homeont mapping for %s", key)

    # -- classes ---------------------------------------------------------

    def location_class(self, area: Dict[str, Any]) -> Optional[str]:
        """The room class of an HA area.

        The home's `location_class` wins (one home is one area today);
        otherwise the area's name or id is matched against the room labels in
        homeont ("Living Room" -> homeont:LivingRoom).
        """
        override = self.home.get("location_class")
        if override:
            return str(override)
        for text in (area.get("name"), area.get("area_id")):
            cls = self.room_labels.get(_normalise(text))
            if cls:
                return cls
        self._unmapped(f"area {area.get('area_id')!r}")
        return None

    def device_class(self, device_name: Optional[str],
                     entities: Iterable[Entity]) -> Optional[str]:
        """What a device is, from its primary entity."""
        override = self._device_override(device_name).get("device_class")
        if override:
            return str(override)
        priority = self.mappings.get("domain_priority") or []
        ordered = sorted(entities, key=lambda e: (
            priority.index(e.domain) if e.domain in priority else len(priority)))
        for entity in ordered:
            for rule in self.mappings.get("device_classes") or []:
                if _matches(rule, entity):
                    return str(rule["class"])
        self._unmapped(f"device {device_name!r}")
        return None

    def property_rule(self, device_name: Optional[str], device_class: Optional[str],
                      entity: Entity, signal: str) -> Optional[Dict[str, Any]]:
        """The rule typing one property, or None. An override is a bare class.

        Order: the home's override, then the first matching `properties` rule,
        then -- for a sensor's own reading only -- its unit
        (`environment_class_for_unit`), so a sensor HA gives no device_class
        is still typed when its unit names one environment variable.
        """
        # 1. The home says what this property is.
        override = (self._device_override(device_name).get("properties") or {}).get(signal)
        if override:
            return {"class": str(override)}
        # 2. The defaults, first match wins.
        for rule in self.mappings.get("properties") or []:
            if _matches(rule, entity, device=device_class, signal=signal):
                return rule
        # 3. A sensor's reading, typed by its unit alone.
        if entity.domain == "sensor" and signal == "state":
            by_unit = self.environment_class_for_unit(entity.unit)
            if by_unit is not None:
                return {"class": by_unit.curie}
        return None

    def property_class(self, device_name: Optional[str], device_class: Optional[str],
                       entity: Entity, signal: str) -> Optional[str]:
        rule = self.property_rule(device_name, device_class, entity, signal)
        return str(rule["class"]) if rule else None

    def declared_properties(self, device_name: Optional[str],
                            device_class: Optional[str],
                            entity: Entity) -> List[Dict[str, Any]]:
        """Properties a device has by what it supports, whatever HA reports now."""
        return [
            rule for rule in self.mappings.get("properties") or []
            if rule.get("declared") and rule.get("signal") != "state"
            and _matches(rule, entity, device=device_class,
                         signal=str(rule.get("signal")))
        ]

    def command_class(self, device_name: Optional[str], domain: str,
                      service: str) -> Optional[str]:
        override = (self._device_override(device_name).get("commands") or {}).get(service)
        if override:
            return str(override)
        entity = Entity(domain=domain)
        for rule in self.mappings.get("commands") or []:
            if _matches(rule, entity, service=service):
                return str(rule["class"])
        self._unmapped(f"service {domain}.{service}")
        return None

    def acts_upon(self, domain: str, service: str,
                  fields: Iterable[str] = ()) -> Set[str]:
        """The signals (`state` or attribute names) a service changes.

        Its field names, plus every `acts_upon` rule matching the service.
        """
        fields = {str(f) for f in fields}
        signals = set(fields)
        entity = Entity(domain=domain)
        for rule in self.mappings.get("acts_upon") or []:
            if not _matches(rule, entity, service=service):
                continue
            if "fields_any" in rule and not (_as_set(rule["fields_any"]) & fields):
                continue
            signals |= _as_set(rule.get("signals"))
        return signals

    # -- environment variables --------------------------------------------

    def environment_class(self, curie: Optional[str]) -> Optional[EnvironmentClass]:
        """The environment class a property class is, or None when homeont
        does not make it a property of a space's environment."""
        return self.environment_classes.get(curie or "")

    def environment_class_for_unit(self, unit: Optional[str]) -> Optional[EnvironmentClass]:
        """The one environment class whose homeont unit is this HA unit
        (`lx` -> `unit:LUX` -> homeont:Illuminance), via the `units` table.

        None when the unit is unknown or ambiguous: `%` is both relative
        humidity and glare, and guessing between them would be worse than
        leaving the reading untyped.
        """
        qudt_unit = expand((self.mappings.get("units") or {}).get(unit or ""))
        if qudt_unit is None:
            return None
        matches = [c for c in self.environment_classes.values() if c.unit == qudt_unit]
        return matches[0] if len(matches) == 1 else None

    def variable(self, device_name: Optional[str],
                 curie: Optional[str]) -> Optional[Variable]:
        """The variable a device's property of class `curie` observes (or its
        action moves), or None when `curie` is no environment class.

        The device's `zone` makes it a second variable of the same class in
        the same environment (`desk` -> `desk_illuminance`); its `environment`
        moves it to another environment the home declares (`outdoor` ->
        `outdoor_illuminance`, of the outdoor environment).
        """
        # 1. Only homeont's environment classes are variables.
        cls = self.environment_class(curie)
        if cls is None:
            return None
        # 2. Which environment, and which name in it: the class's, or zoned.
        override = self._device_override(device_name)
        environment = str(override.get("environment") or ROOM_ENVIRONMENT)
        prefix = override.get("zone") or (
            environment if environment != ROOM_ENVIRONMENT else None)
        name = f"{prefix}_{cls.name}" if prefix else cls.name
        return Variable(cls=cls, name=name, environment=environment)

    def effects(self, device_name: Optional[str], domain: str,
                service: str) -> List[Tuple[Variable, Optional[str]]]:
        """The variables a service of a device moves, each with its direction
        (`increase` / `decrease`, or None when the value decides).

        The device's own `effects` replace the defaults (`effects` rules in
        semantic_mappings.yaml). In a rule, `services` restricts which
        services it covers, `<service>: increase|decrease` states a direction,
        and `direction: none` makes every covered service directionless.
        Otherwise the direction is read from the service name; a service with
        none and no setter (a toggle, a stop) has no stated effect.
        """
        # 1. The rules that apply: the device's own, else the defaults for its domain.
        override = self._device_override(device_name)
        if "effects" in override:
            rules = list(override.get("effects") or [])
        else:
            rules = [r for r in self.mappings.get("effects") or []
                     if domain in _as_set(r.get("domain"))]

        # 2. Each rule covering this service, with its direction.
        out: List[Tuple[Variable, Optional[str]]] = []
        for rule in rules:
            if "services" in rule and service not in _as_set(rule["services"]):
                continue
            variable = self.variable(device_name, rule.get("class"))
            if variable is None:
                self._unmapped(f"effect class {rule.get('class')!r}")
                continue
            stated = rule.get(service)
            if stated in DIRECTIONS:
                out.append((variable, stated))
            elif str(rule.get("direction")) == "none" or service in SETPOINT_SERVICES:
                out.append((variable, None))
            elif service_direction(service):
                out.append((variable, service_direction(service)))
        return out

    def target_range(self, variable_name: str) -> Dict[str, Any]:
        """The home's comfort range for a variable (`min`, `max`, `unit`), or {}."""
        return dict((self.home.get("ranges") or {}).get(variable_name) or {})

    def environments(self) -> Dict[str, Dict[str, Any]]:
        """The environments the home declares beside the room's own, by key."""
        return {str(k): dict(v or {}) for k, v in (self.home.get("environments") or {}).items()}

    def variable_name(self, text: str) -> str:
        """A variable's name from a name or an older key (`luminosity` ->
        `illuminance`), via the `variable_aliases` table."""
        return str((self.mappings.get("variable_aliases") or {}).get(text, text))


def _homeont_tables(path: Path):
    """From homeont: (room label lookup, class label table, environment classes).

    The first maps a normalised label / local name to a room class CURIE, for
    matching HA area names; the second maps every homeont class CURIE to its
    `rdfs:label`, for naming what a device is; the third is every class
    homeont makes a property of a space's environment (`_environment_classes`).
    """
    # 1. The ontology; without it, every table is empty and nothing is typed.
    if not path.is_file():
        return {}, {}, {}
    graph = Graph()
    try:
        graph.parse(str(path), format="turtle")
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}, {}, {}

    # 2. Every class's English label.
    class_labels: Dict[str, str] = {}
    for subject, label in graph.subject_objects(RDFS.label):
        text = str(subject)
        if text.startswith(str(HOMEONT)) and getattr(label, "language", None) in (None, "en"):
            class_labels.setdefault("homeont:" + text[len(str(HOMEONT)):], str(label))
    return _room_table(graph), class_labels, _environment_classes(graph, class_labels)


def _environment_classes(graph: Graph,
                         class_labels: Dict[str, str]) -> Dict[str, EnvironmentClass]:
    """CURIE -> EnvironmentClass for every homeont class stated to be a
    property of `homeont:Environment`, with its quantity kind and unit.

    Read from the ontology, not listed here, so adding a variable to homeont
    is enough for HASP to emit it.
    """
    out: Dict[str, EnvironmentClass] = {}
    # Each homeont class with `ssn:isPropertyOf homeont:Environment`, with the
    # name SimuHome would give it and what QUDT says it measures.
    for cls in graph.subjects(SSN.isPropertyOf, HOMEONT.Environment):
        text = str(cls)
        if not text.startswith(str(HOMEONT)):
            continue
        local = text[len(str(HOMEONT)):]
        curie = f"homeont:{local}"
        out[curie] = EnvironmentClass(
            curie=curie, name=snake_case(local), label=class_labels.get(curie),
            quantity_kind=graph.value(cls, QUDT.hasQuantityKind),
            unit=graph.value(cls, QUDT.unit))
    return out


def _room_table(graph: Graph) -> Dict[str, str]:
    """Normalised label / local name -> CURIE, for every homeont room class."""
    root = HOMEONT.BuildingSpace
    rooms = set(graph.transitive_subjects(RDFS.subClassOf, root)) - {root}
    table: Dict[str, str] = {}
    for room in rooms:
        curie = "homeont:" + str(room)[len(str(HOMEONT)):]
        table[_normalise(curie.split(":", 1)[1])] = curie
        for label in graph.objects(room, RDFS.label):
            table[_normalise(str(label))] = curie
    return table


def _home_config(path: Optional[str]) -> Dict[str, Any]:
    """The `semantic:` block of a home config. A relative path that does not
    exist is looked up beside this module, where the home configs live."""
    if not path:
        return {}
    candidate = Path(path)
    if not candidate.is_absolute() and not candidate.exists():
        candidate = HERE / candidate.name
    if not candidate.is_file():
        logger.warning("SEMANTIC_CONFIG %s not found", path)
        return {}
    data = yaml.safe_load(candidate.read_text()) or {}
    return data.get("semantic") or {}


def load_semantics(home_config: Optional[str] = None) -> Semantics:
    """Defaults from `semantic_mappings.yaml`, overrides from the home config."""
    mappings = yaml.safe_load(MAPPINGS_PATH.read_text()) or {}
    home = _home_config(home_config if home_config is not None
                        else os.getenv("SEMANTIC_CONFIG"))
    room_labels, class_labels, environment_classes = _homeont_tables(HOMEONT_PATH)
    return Semantics(mappings, home, room_labels, class_labels, environment_classes)


# -- the semantics in use --------------------------------------------------

_current: Optional[Semantics] = None


def current() -> Semantics:
    """The semantics HASP types its TDs with, loaded on first use."""
    global _current
    if _current is None:
        _current = load_semantics()
    return _current


def reload(home_config: Optional[str] = None) -> Semantics:
    """Re-read the mappings and the home config (after SEMANTIC_CONFIG changes)."""
    global _current
    _current = load_semantics(home_config)
    return _current


# -- IRIs -----------------------------------------------------------------
#
# SimuHome's shapes: <room>#place, <room>#environment, <room>#<variable>. Every
# document names these nodes by the same IRIs, so the workspace document and
# each artifact's describe one graph.

def place_uri(base: str, workspace_id: str) -> URIRef:
    """The physical room an HA area is, distinct from its workspace."""
    return URIRef(f"{base}workspaces/{workspace_id}#place")


def environment_uri(base: str, workspace_id: str,
                    environment: str = ROOM_ENVIRONMENT) -> URIRef:
    """An environment of a workspace: the room's own (`#environment`), or
    one the home declares (`outdoor` -> `#outdoor_environment`)."""
    suffix = "environment" if environment == ROOM_ENVIRONMENT else f"{environment}_environment"
    return URIRef(f"{base}workspaces/{workspace_id}#{suffix}")


def variable_uri(base: str, workspace_id: str, name: str) -> URIRef:
    """An environment variable of a workspace (`#illuminance`)."""
    return URIRef(f"{base}workspaces/{workspace_id}#{name}")


def variable_target(base: str, workspace_id: str, name: str) -> URIRef:
    """Where a variable's current value is read (HASP's room-property route)."""
    return URIRef(f"{base}workspaces/{workspace_id}/properties/{name}")


# -- emitting it ----------------------------------------------------------

def add_room(graph: Graph, base: str, area: Dict[str, Any], workspace: URIRef) -> None:
    """The place and environment of one area, as SimuHome's room_workspace has them.

    The resolvers scope by `?place a <room> ; homeont:isSpaceOfWorkspace ?ws`
    and name the room by the place's `rdfs:label`; they reach the room's
    variables by `?place homeont:hasEnvironment ?env . ?env ssn:hasProperty ?var`.
    """
    # 1. The place and its workspace.
    workspace_id = area["area_id"]
    title = area.get("name") or workspace_id
    place = _add_place(graph, base, area)
    graph.add((place, RDFS.label, Literal(title)))
    graph.add((place, HOMEONT.isSpaceOfWorkspace, workspace))
    graph.add((workspace, HOMEONT.hasSpace, place))

    # 2. Its environment, labelled, and offered on the workspace as SimuHome does.
    environment = environment_uri(base, workspace_id)
    graph.add((environment, RDFS.label, Literal(f"{title} environment")))
    graph.add((workspace, TD.hasPropertyAffordance, environment))
    graph.bind("homeont", HOMEONT)


def _add_place(graph: Graph, base: str, area: Dict[str, Any]) -> URIRef:
    """The place of an area typed with its room class, and its environment
    linked both ways; the place IRI."""
    workspace_id = area["area_id"]
    place = place_uri(base, workspace_id)
    room_class = expand(current().location_class(area))
    if room_class is not None:
        graph.add((place, RDF.type, room_class))
    graph.add((place, RDF.type, HOMEONT.BuildingSpace))
    environment = environment_uri(base, workspace_id)
    graph.add((environment, RDF.type, HOMEONT.Environment))
    graph.add((environment, RDF.type, SOSA.FeatureOfInterest))
    graph.add((environment, HOMEONT.isEnvironmentOf, place))
    graph.add((place, HOMEONT.hasEnvironment, environment))
    return place


def add_place_context(graph: Graph, base: str, area: Dict[str, Any], artifact: URIRef) -> None:
    """Where an artifact is, in its own document: located in the room's
    place, which has the room's environment -- so an agent holding only this
    TD can walk from the device to the room and its variables."""
    place = _add_place(graph, base, area)
    graph.add((artifact, HOMEONT.isLocatedIn, place))
    graph.add((place, HOMEONT.containsArtifact, artifact))
    graph.bind("homeont", HOMEONT)


def add_environment(graph: Graph, base: str, workspace_id: str, key: str,
                    label: Optional[str] = None) -> URIRef:
    """An environment the home declares beside the room's (`outdoor`): a
    feature of interest of its own, which no place `hasEnvironment` -- the
    room's queries must not take the outdoor light for the room's."""
    environment = environment_uri(base, workspace_id, key)
    graph.add((environment, RDF.type, HOMEONT.Environment))
    graph.add((environment, RDF.type, SOSA.FeatureOfInterest))
    graph.add((environment, RDFS.label, Literal(label or f"{key.capitalize()} environment")))
    return environment


def link_variable(graph: Graph, base: str, workspace_id: str, variable: Variable) -> URIRef:
    """The variable as an artifact's document mentions it: its class and the
    environment it belongs to. Its full description is the workspace's
    (`add_variable`); the IRI is the same in both."""
    node = variable_uri(base, workspace_id, variable.name)
    add_type(graph, node, variable.cls.curie)
    graph.add((node, SSN.isPropertyOf, environment_uri(base, workspace_id, variable.environment)))
    return node


def add_variable(graph: Graph, base: str, workspace_id: str, variable: Variable,
                 readable: bool, target: Optional[Dict[str, Any]] = None) -> URIRef:
    """A room variable in full, as SimuHome's room_workspace describes it.

    Typed with its homeont class and as a TD property affordance of the
    environment; named; with its quantity kind and unit; readable through
    HASP's room-property route when some device observes it (`readable`),
    otherwise marked not observable and given no form; with the home's
    comfort range (`target`: min / max / unit) when it states one.
    """
    # 1. What it is, and whose property.
    node = link_variable(graph, base, workspace_id, variable)
    environment = environment_uri(base, workspace_id, variable.environment)
    graph.add((environment, SSN.hasProperty, node))
    for cls in (TD.PropertyAffordance, TDSOSA.ObservablePropertyAffordance,
                SOSA.ObservableProperty):
        graph.add((node, RDF.type, cls))
    graph.add((node, TD.name, Literal(variable.name)))
    graph.add((node, TD.title, Literal(variable.name)))
    graph.add((node, TD.isObservable, Literal(readable)))

    # 2. What it measures, and the shape of its value.
    if variable.cls.quantity_kind is not None:
        graph.add((node, QUDT.hasQuantityKind, variable.cls.quantity_kind))
    if variable.cls.unit is not None:
        graph.add((node, QUDT.unit, variable.cls.unit))
        schema = BNode()
        graph.add((schema, RDF.type, JS.NumberSchema))
        graph.add((schema, QUDT.unit, variable.cls.unit))
        graph.add((node, TD.hasOutputSchema, schema))

    # 3. How to read it -- only when a device observes it.
    if readable:
        form = BNode()
        graph.add((node, TD.hasForm, form))
        graph.add((form, HTV.methodName, Literal("GET")))
        graph.add((form, HCTL.hasTarget, variable_target(base, workspace_id, variable.name)))
        graph.add((form, HCTL.forContentType, Literal("application/json")))
        graph.add((form, HCTL.hasOperationType, TD.readProperty))

    # 4. The comfort range the home wants it kept in.
    target = target or {}
    if "min" in target:
        graph.add((node, TDSOSA.targetMinValue, Literal(target["min"])))
    if "max" in target:
        graph.add((node, TDSOSA.targetMaxValue, Literal(target["max"])))
    if "unit" in target:
        graph.add((node, TDSOSA.targetUnit, Literal(str(target["unit"]))))
    for prefix, namespace in (("ssn", SSN), ("qudt", QUDT), ("unit", UNIT),
                              ("quantitykind", QUANTITYKIND), ("tdsosa", TDSOSA)):
        graph.bind(prefix, namespace)
    return node


def add_label(graph: Graph, artifact: Any, device_class: Optional[str]) -> None:
    """`rdfs:label` the kind of device, as SimuHome's TDs carry it.

    Three names, three facts (see homeont.ttl, "Matter BasicInformation"):
    `td:title` is the instance, `rdfs:label` the kind of device ("Color
    Light"), `schema:model` the product. No class, no label.
    """
    label = current().label(device_class)
    if label:
        graph.add((artifact, RDFS.label, Literal(label)))


def add_type(graph: Graph, subject: Any, curie: Optional[str]) -> None:
    """`subject a <curie>`, when the mapping produced one."""
    uri = expand(curie)
    if uri is not None:
        graph.add((subject, RDF.type, uri))
