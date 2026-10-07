"""Home Assistant -> home ontology: which homeont / SAREF class a thing is.

HASP's TDs carry the same semantic layer SimuHome's do -- a device class on the
Thing, a property class on each property affordance, a command class on each
action, a room class on the workspace's space -- so the agents' class-based
resolvers work on a Home Assistant home unchanged.

The classes come from `semantic_mappings.yaml` (defaults, by HA domain /
service / attribute / device_class), overridden per device by the `semantic:`
block of the home's own config (the file `SEMANTIC_CONFIG` points at). Pure
functions and data: no HA connection, no FastAPI, testable on their own.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

logger = logging.getLogger("hasp.semantics")

HERE = Path(__file__).resolve().parent
MAPPINGS_PATH = HERE / "semantic_mappings.yaml"
# .../ami_agents/environment/integration/HomeAssistant -> .../ami_agents
HOMEONT_PATH = Path(os.getenv(
    "HOMEONT_PATH", str(HERE.parents[2] / "shared" / "ontologies" / "homeont.ttl")))

HOMEONT = Namespace("http://example.org/homeont/")
SAREF = Namespace("https://saref.etsi.org/core/")
SOSA = Namespace("http://www.w3.org/ns/sosa/")

_PREFIXES = {
    "homeont": str(HOMEONT),
    "saref": str(SAREF),
    "sosa": str(SOSA),
}

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
                 class_labels: Optional[Dict[str, str]] = None) -> None:
        self.mappings = mappings or {}
        self.home = home or {}
        self.room_labels = room_labels or {}
        self.class_labels = class_labels or {}
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
        """The rule typing one property, or None. An override is a bare class."""
        override = (self._device_override(device_name).get("properties") or {}).get(signal)
        if override:
            return {"class": str(override)}
        for rule in self.mappings.get("properties") or []:
            if _matches(rule, entity, device=device_class, signal=signal):
                return rule
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

    def environment_class(self, env_key: Optional[str]) -> Optional[str]:
        """The homeont room variable a TD-SOSA effect key stands for."""
        if not env_key:
            return None
        return (self.mappings.get("environment_variables") or {}).get(env_key)


def _homeont_labels(path: Path):
    """From homeont: (room label lookup, class label table).

    The first maps a normalised label / local name to a room class CURIE, for
    matching HA area names; the second maps every homeont class CURIE to its
    `rdfs:label`, for naming what a device is.
    """
    if not path.is_file():
        return {}, {}
    graph = Graph()
    try:
        graph.parse(str(path), format="turtle")
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}, {}
    class_labels: Dict[str, str] = {}
    for subject, label in graph.subject_objects(RDFS.label):
        text = str(subject)
        if text.startswith(str(HOMEONT)) and getattr(label, "language", None) in (None, "en"):
            class_labels.setdefault("homeont:" + text[len(str(HOMEONT)):], str(label))
    return _room_table(graph), class_labels


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
    """The `semantic:` block of a home config, resolved like ygg_ha_adapter."""
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
    room_labels, class_labels = _homeont_labels(HOMEONT_PATH)
    return Semantics(mappings, home, room_labels, class_labels)


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


# -- emitting it ----------------------------------------------------------

def room_place_uri(base: str, workspace_id: str) -> URIRef:
    """The physical room an HA area is, distinct from its workspace."""
    return URIRef(f"{base}workspaces/{workspace_id}#space")


def room_environment_uri(base: str, workspace_id: str) -> URIRef:
    """The room's environment: what its sensors observe and actuators change."""
    return URIRef(f"{base}workspaces/{workspace_id}#environment")


def add_room(graph: Graph, base: str, area: Dict[str, Any], workspace: URIRef) -> None:
    """The place and environment of one area, as SimuHome's room_workspace has them.

    The resolvers scope by `?space a <room> ; homeont:isSpaceOfWorkspace ?ws`
    and name the room by the space's `rdfs:label`.
    """
    workspace_id = area["area_id"]
    title = area.get("name") or workspace_id
    place = room_place_uri(base, workspace_id)
    room_class = expand(current().location_class(area))
    if room_class is not None:
        graph.add((place, RDF.type, room_class))
    graph.add((place, RDF.type, HOMEONT.BuildingSpace))
    graph.add((place, RDFS.label, Literal(title)))
    graph.add((place, HOMEONT.isSpaceOfWorkspace, workspace))
    graph.add((workspace, HOMEONT.hasSpace, place))

    environment = room_environment_uri(base, workspace_id)
    graph.add((environment, RDF.type, HOMEONT.Environment))
    graph.add((environment, RDF.type, SOSA.FeatureOfInterest))
    graph.add((environment, RDFS.label, Literal(f"{title} environment")))
    graph.add((environment, HOMEONT.isEnvironmentOf, place))
    graph.add((place, HOMEONT.hasEnvironment, environment))
    graph.bind("homeont", HOMEONT)


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
