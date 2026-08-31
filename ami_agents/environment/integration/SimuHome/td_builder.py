#!/usr/bin/env python3
"""
Build the HMAS/TD Turtle that SHTD serves: platform -> home -> rooms -> devices.

Phase 2 is read-only: every device attribute becomes a `td:PropertyAffordance`
with a GET form. Actuatable attributes are *marked* here (`tdsosa:` typing and
`writable` output-schema hints) but their invocation forms arrive in phase 3.

Two contracts from the agent crawler drive the shapes, and both were read out of
`integration_engine.py` rather than assumed:

  * `_process_workspace_recursive` (line 1075) only recurses into a contained
    resource when `(sub, rdf:type, hmas:Workspace)` is present **in the parent's
    own graph**. So the home document must type each room inline.
  * `_map_artifacts` (line 1138) likewise only accepts a contained resource as an
    artifact when `(art, rdf:type, hmas:Artifact)` is in the *workspace's* graph.

Both are satisfied by emitting the type triple in the containing document as well
as in the contained resource's own.

Subjects follow HASP's convention: the RDF subject is `<path>#workspace` /
`<path>#artifact` / `<path>#platform`, and the bare path is its
`hmas:ResourceProfile`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rdflib import BNode, Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import RDFS, XSD

try:
    from .classify import classify_device
    from .mappings import load_mappings
except ImportError:  # pragma: no cover - direct script use
    from classify import classify_device  # type: ignore
    from mappings import load_mappings  # type: ignore

# `classify` (imported above) puts this package's directory on sys.path, so the
# vendored registry -- a sibling package -- is importable by the time we get
# here. Every Matter fact comes from it, never from model memory.
from matter_model.registry import load_registry  # noqa: E402

HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
JS = Namespace("https://www.w3.org/2019/wot/json-schema#")
HMAS = Namespace("https://purl.org/hmas/")
# All SHTD vocabulary -- spaces, device families, environmental properties
# and Matter provenance -- lives in one ontology:
# ami_agents/shared/ontologies/homeont.ttl
HOME = Namespace("http://example.org/homeont/")
WOTSEC = Namespace("https://www.w3.org/2019/wot/security#")
HTV = Namespace("http://www.w3.org/2011/http#")
JACAMO = Namespace("https://purl.org/hmas/jacamo/")
WEBSUB = Namespace("https://purl.org/hmas/websub/")
TD = Namespace("https://www.w3.org/2019/wot/td#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
QK = Namespace("http://qudt.org/vocab/quantitykind/")
TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")
SAREF = Namespace("https://saref.etsi.org/core/")

_PREFIXES = {
    "hctl": HCTL, "js": JS, "hmas": HMAS, "wotsec": WOTSEC,
    "htv": HTV, "jacamo": JACAMO, "websub": WEBSUB, "td": TD, "sosa": SOSA,
    "ssn": SSN, "qudt": QUDT, "unit": UNIT, "quantitykind": QK,
    "tdsosa": TDSOSA, "saref": SAREF, "rdfs": RDFS, "homeont": HOME,
}

# CURIEs in the mapping tables resolve against these.
#
# There is no `ex:` alias. `homeont:` is the single prefix for this vocabulary,
# in the tables and in every served document alike -- two spellings for one
# namespace is how a normative term ends up reading as an example placeholder.
# An unknown prefix returns None from `curie_to_uri`, so a stray `ex:` now
# surfaces as a missing type rather than resolving silently.
_NS_BY_PREFIX = {
    "homeont": HOME, "saref": SAREF, "sosa": SOSA, "ssn": SSN,
    "quantitykind": QK, "unit": UNIT, "tdsosa": TDSOSA, "td": TD,
}

# Building-space classes homeont defines. Read from the ontology itself rather
# than hardcoded, so adding a space type there is enough to make it usable here.
def _load_home_space_classes() -> set:
    path = (Path(__file__).resolve().parents[3]
            / "shared" / "ontologies" / "homeont.ttl")
    try:
        onto = Graph()
        onto.parse(str(path), format="turtle")
    except Exception:  # the ontology is advisory here, never fatal
        return set()
    return {
        str(subject).rsplit("/", 1)[-1]
        for subject in onto.subjects(RDF.type,
                                     URIRef("http://www.w3.org/2002/07/owl#Class"))
        if str(subject).startswith(str(HOME))
    }


HOME_SPACE_CLASSES = _load_home_space_classes()

# Matter carries these in centi-units; the TD reports human values, so the
# scale is declared per property rather than left for the agent to guess.
CENTI_TOKENS = {"temperature", "humidity"}


def curie_to_uri(curie: Optional[str]) -> Optional[URIRef]:
    """`"saref:HVAC"` -> URIRef. Returns None for unknown or absent prefixes."""
    if not curie or ":" not in str(curie):
        return None
    prefix, local = str(curie).split(":", 1)
    ns = _NS_BY_PREFIX.get(prefix)
    return URIRef(f"{ns}{local}") if ns is not None else None


def _new_graph() -> Graph:
    g = Graph()
    for prefix, ns in _PREFIXES.items():
        g.bind(prefix, ns)
    return g


def _literal(value: Any) -> Literal:
    """Type a Python value for RDF without losing bool/int distinction."""
    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)
    if isinstance(value, int):
        return Literal(value, datatype=XSD.integer)
    if isinstance(value, float):
        return Literal(value, datatype=XSD.double)
    return Literal(str(value))


# Matter's integer-valued primitives, enumerated from the vendored registry
# rather than pattern-matched. The physical-quantity types carry their unit in
# the name and their scale with it -- `power-mW` is milliwatts, `voltage-mV`
# millivolts -- and every one of them is an integer on the wire.
_INTEGER_TYPE_PREFIXES = (
    "uint", "int", "enum", "map",
    "percent", "temperature",
    "power-", "voltage-", "amperage-", "energy-",
    "elapsed-", "epoch-", "posix-", "systime-",
)
_INTEGER_TYPES = {
    "fabric-idx", "node-id", "vendor-id", "cluster-id", "endpoint-no",
    "SignedTemperature", "UnsignedTemperature",
}


def _is_integer_type(matter_type: str) -> bool:
    return (matter_type in _INTEGER_TYPES
            or matter_type.startswith(_INTEGER_TYPE_PREFIXES))


# Unit symbols the mapping tables use, mapped to their QUDT IRI. Every one of
# these resolves (verified against qudt.org), and each carries `qudt:symbol`
# plus a language-less `rdfs:label`, which is where a consumer gets readable
# text -- so nothing here needs to restate the symbol.
_QUDT_UNITS = {
    "\u00b0C": "DEG_C", "%": "PERCENT", "lx": "LUX", "ug/m3": "MicroGM-PER-M3",
    "s": "SEC", "mA": "MilliA", "mW": "MilliW", "mVA": "MilliV-A", "mV": "MilliV",
}


def _add_unit(g: Graph, schema: BNode, unit: Any) -> None:
    """Attach the unit as a QUDT IRI.

    The readable symbol is NOT emitted: `qudt:symbol` on the unit IRI already
    carries it ("°C", "mW"), so a local copy would be redundant.

    TODO (planner-context serialization): when building the helper functions
    that render a TD for an agent's context, resolve each `qudt:unit` IRI to
    its `qudt:symbol` and its language-less `rdfs:label` ("Degree Celsius").
    QUDT publishes rdfs:label in many languages; take the one with no language
    tag. Verified present for every unit this corpus uses.
    """
    if not unit:
        return
    qudt_local = _QUDT_UNITS.get(str(unit))
    if qudt_local:
        g.add((schema, QUDT.unit, UNIT[qudt_local]))


# Matter types stored scaled: the wire value is the human value x100.
# `temperature` covers Thermostat.LocalTemperature and the setpoints;
# `percent100ths` covers window-covering positions.
SCALED_TYPES = {"temperature": 100.0, "percent100ths": 100.0}


def report_value(matter_type: Optional[str], value: Any) -> Any:
    """The value as the Thing Description reports it.

    Matter carries temperature in hundredths of a degree, so a device reading
    2340 is 23.4 C. The room path already divided; device attributes did not,
    so the same quantity was served two ways while both schemas declared
    unit:DEG_C. One function now decides, used by the TD schema, the property
    read route and the change notifier alike.
    """
    scale = SCALED_TYPES.get(str(matter_type or ""))
    if scale and isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(value / scale, 2)
    return value


def _no_security(g: Graph, subject: URIRef) -> None:
    sec = BNode()
    g.add((subject, TD.hasSecurityConfiguration, sec))
    g.add((sec, RDF.type, WOTSEC.NoSecurityScheme))


class SimuHomeTD:
    """Render one loaded SimuHome home as HMAS/TD Turtle.

    `state` is exactly what `GET /api/home/state` returns; `home` is the episode
    id, which is load-bearing: each seed is a different house (217 distinct
    device layouts across the 600 benchmark episodes).
    """

    def __init__(self, base: str, home: str, state: Dict[str, Any]):
        self.base = base.rstrip("/") + "/"
        self.home = home
        self.state = state or {}
        self.rooms: Dict[str, Any] = self.state.get("rooms") or {}
        self.m = load_mappings()

    # -- URIs --------------------------------------------------------------

    def platform_uri(self) -> URIRef:
        return URIRef(f"{self.base}#platform")

    def home_path(self) -> str:
        return f"{self.base}workspaces/{self.home}"

    def room_path(self, room: str) -> str:
        return f"{self.home_path()}/{room}"

    def artifact_path(self, room: str, device_id: str) -> str:
        return f"{self.room_path(room)}/artifacts/{device_id}"

    def place_uri(self, room: str) -> URIRef:
        """The room as a PLACE -- a physical building space where devices sit.

        Distinct from its environment: a freezer is *located in* the kitchen but
        does not observe or affect the kitchen's air.
        """
        return URIRef(f"{self.room_path(room)}#place")

    def room_property_uri(self, room: str, sosa_property: str) -> URIRef:
        """A room's observable property, addressable from any document."""
        return URIRef(f"{self.room_path(room)}#{sosa_property}")

    def appliance_foi_uri(self, room: str, device_id: str) -> URIRef:
        """The interior of an appliance -- what a freezer's thermometer reads.

        A real feature of interest, distinct from the room: the freezer reads
        -15 C inside a 23.6 C kitchen.
        """
        return URIRef(f"{self.artifact_path(room, device_id)}#interior")

    def foi_uri(self, room: str) -> URIRef:
        """The room's ENVIRONMENT as `sosa:FeatureOfInterest` -- the thing whose
        air temperature, humidity, illuminance and PM10 are being observed.

        Only devices that actually perceive or affect one of those variables
        link to it. Containment is a separate relation and lives on the place.
        """
        return URIRef(f"{self.room_path(room)}#environment")

    # -- documents ---------------------------------------------------------

    def platform(self) -> Graph:
        g = _new_graph()
        platform = self.platform_uri()
        g.add((platform, RDF.type, HMAS.HypermediaMASPlatform))
        g.add((platform, RDF.type, TD.Thing))
        g.add((platform, TD.title, Literal("SimuHome Platform")))
        _no_security(g, platform)

        ws = URIRef(f"{self.home_path()}#workspace")
        g.add((platform, HMAS.hosts, ws))
        # Typed inline so a crawler reading only this document knows what it is.
        g.add((ws, RDF.type, HMAS.Workspace))
        g.add((ws, HMAS.isHostedOn, platform))
        g.add((ws, TD.title, Literal(self.home)))

        profile = URIRef(self.base)
        g.add((profile, RDF.type, HMAS.ResourceProfile))
        g.add((profile, HMAS.isProfileOf, platform))
        return g

    def workspaces(self) -> Graph:
        """The directory of hosted workspaces: just the home."""
        return self.platform()

    def home_workspace(self) -> Graph:
        """The home: contains one workspace per room.

        Rooms are typed `hmas:Workspace` here because the crawler will not
        recurse into them otherwise.
        """
        g = _new_graph()
        ws = URIRef(f"{self.home_path()}#workspace")
        g.add((ws, RDF.type, TD.Thing))
        g.add((ws, RDF.type, HMAS.Workspace))
        g.add((ws, TD.title, Literal(self.home)))
        g.add((ws, HMAS.isHostedOn, self.platform_uri()))
        g.add((self.platform_uri(), RDF.type, HMAS.HypermediaMASPlatform))
        g.add((self.platform_uri(), HMAS.hosts, ws))
        g.add((ws, RDFS.comment, Literal(
            f"SimuHome benchmark episode '{self.home}': "
            f"{len(self.rooms)} rooms, {self.device_count()} devices."
        )))
        _no_security(g, ws)

        for room_id in sorted(self.rooms):
            room_ws = URIRef(f"{self.room_path(room_id)}#workspace")
            g.add((ws, HMAS.contains, room_ws))
            g.add((room_ws, HMAS.isContainedIn, ws))
            g.add((room_ws, RDF.type, HMAS.Workspace))
            g.add((room_ws, TD.title, Literal(self._room_title(room_id))))
            room_class = curie_to_uri((self.m.room(room_id) or {}).get("homeont_class"))
            if room_class is not None:
                g.add((room_ws, RDF.type, room_class))

        self._subscribe_actions(g, ws, self.home_path(), artifact=False)
        profile = URIRef(self.home_path())
        g.add((profile, RDF.type, HMAS.ResourceProfile))
        g.add((profile, HMAS.isProfileOf, ws))
        return g

    def room_workspace(self, room_id: str) -> Graph:
        """One room: its devices as contained artifacts, plus the room's own
        environmental state as a `sosa:FeatureOfInterest` with properties."""
        room = self.rooms.get(room_id) or {}
        g = _new_graph()
        ws = URIRef(f"{self.room_path(room_id)}#workspace")
        home_ws = URIRef(f"{self.home_path()}#workspace")
        g.add((ws, RDF.type, TD.Thing))
        g.add((ws, RDF.type, HMAS.Workspace))
        g.add((ws, TD.title, Literal(self._room_title(room_id))))
        # Both directions, so the parent is reachable from here without having
        # to remember how we arrived. `hmas:contains` and `hmas:isContainedIn`
        # are declared `owl:inverseOf` in the HMAS ontology, so asserting both
        # is redundant to a reasoner but load-bearing for a plain crawler.
        g.add((ws, HMAS.isContainedIn, home_ws))
        g.add((home_ws, RDF.type, HMAS.Workspace))
        g.add((home_ws, HMAS.contains, ws))
        _no_security(g, ws)

        # The PLACE: a physical building space. This is what a device is
        # located in, and it carries the room's homeont: type (homeont:Kitchen).
        place = self.place_uri(room_id)
        room_class = self._room_class(room_id)
        if room_class is not None:
            g.add((place, RDF.type, room_class))
        g.add((place, RDF.type, HOME.BuildingSpace))
        g.add((place, RDFS.label, Literal(self._room_title(room_id))))
        # The workspace is a logical container; the space is the physical room.
        g.add((place, HOME.isSpaceOfWorkspace, ws))
        g.add((ws, HOME.hasSpace, place))

        # The ENVIRONMENT: a distinct resource, and the only thing here that is
        # a sosa:FeatureOfInterest. Its air temperature, humidity, illuminance
        # and PM10 are what sensors observe and actuators change. Conflating it
        # with the place would make every device in the room an observer of the
        # room's air -- which a freezer, reading -15 C inside a 23.6 C kitchen,
        # plainly is not.
        foi = self.foi_uri(room_id)
        g.add((foi, RDF.type, HOME.Environment))
        g.add((foi, RDF.type, SOSA.FeatureOfInterest))
        g.add((foi, RDFS.label, Literal(f"{self._room_title(room_id)} environment")))
        g.add((foi, HOME.isEnvironmentOf, place))
        g.add((place, HOME.hasEnvironment, foi))
        g.add((ws, TD.hasPropertyAffordance, foi))
        self._room_state_properties(g, room_id, room.get("state") or {}, foi)

        for device in sorted(room.get("devices") or [],
                             key=lambda d: str(d.get("device_id") or "")):
            device_id = str(device.get("device_id") or "")
            if not device_id:
                continue
            art = URIRef(f"{self.artifact_path(room_id, device_id)}#artifact")
            g.add((ws, HMAS.contains, art))
            g.add((art, HMAS.isContainedIn, ws))
            g.add((art, RDF.type, HMAS.Artifact))
            g.add((art, TD.title, Literal(device_id)))

        self._subscribe_actions(g, ws, self.room_path(room_id), artifact=False)
        profile = URIRef(self.room_path(room_id))
        g.add((profile, RDF.type, HMAS.ResourceProfile))
        g.add((profile, HMAS.isProfileOf, ws))
        return g

    def artifacts_directory(self, room_id: str) -> Graph:
        """`GET .../artifacts` -- the contained artifacts, nothing else."""
        room = self.rooms.get(room_id) or {}
        g = _new_graph()
        ws = URIRef(f"{self.room_path(room_id)}#workspace")
        g.add((ws, RDF.type, HMAS.Workspace))
        g.add((ws, TD.title, Literal(self._room_title(room_id))))
        home_ws = URIRef(f"{self.home_path()}#workspace")
        g.add((ws, HMAS.isContainedIn, home_ws))
        g.add((home_ws, RDF.type, HMAS.Workspace))
        g.add((home_ws, HMAS.contains, ws))
        for device in sorted(room.get("devices") or [],
                             key=lambda d: str(d.get("device_id") or "")):
            device_id = str(device.get("device_id") or "")
            if not device_id:
                continue
            art = URIRef(f"{self.artifact_path(room_id, device_id)}#artifact")
            g.add((ws, HMAS.contains, art))
            g.add((art, HMAS.isContainedIn, ws))
            g.add((art, RDF.type, HMAS.Artifact))
            g.add((art, RDF.type, TD.Thing))
            g.add((art, TD.title, Literal(device_id)))
        return g

    def artifact(self, room_id: str, device_id: str) -> Optional[Graph]:
        """One device's Thing Description."""
        device = self.find_device(room_id, device_id)
        if device is None:
            return None

        g = _new_graph()
        path = self.artifact_path(room_id, device_id)
        art = URIRef(f"{path}#artifact")
        device_type = str(device.get("device_type") or "")

        g.add((art, RDF.type, TD.Thing))
        g.add((art, RDF.type, HMAS.Artifact))
        g.add((art, TD.title, Literal(device_id)))
        # Reverse link, with the room typed and named inline so an agent holding
        # only this TD can walk back up to the room and the home.
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
        _no_security(g, art)

        family = self.m.device_family(device_type) or {}
        for curie in family.get("td_types") or []:
            uri = curie_to_uri(curie)
            if uri is not None:
                g.add((art, RDF.type, uri))
        if family.get("title"):
            g.add((art, RDFS.label, Literal(family["title"])))

        # WHERE the device is: always true, and purely spatial.
        place = self.place_uri(room_id)
        g.add((art, HOME.isLocatedIn, place))
        g.add((place, HOME.containsArtifact, art))
        room_class = self._room_class(room_id)
        if room_class is not None:
            g.add((place, RDF.type, room_class))
        g.add((place, RDF.type, HOME.BuildingSpace))

        # WHAT it acts on: only families that actually move one of the room's
        # environmental variables, per the approved actuation_effects table.
        if self._touches_room_environment(device_type):
            g.add((art, SOSA.hasFeatureOfInterest, self.foi_uri(room_id)))

        # WHAT it senses. Six families carry a measurement cluster, and they
        # split absolutely: four read the room's own air, two read their own
        # compartment. Both are real observations of different features of
        # interest -- saying a freezer reports the kitchen temperature would be
        # a lie, and saying a heat pump reports nothing loses a sensor a planner
        # could have used.
        sensing = self.m.senses(device_type)
        if sensing:
            g.add((art, RDF.type, SOSA.Sensor))
            observed = (
                self.room_property_uri(room_id, str(sensing["sosa_property"]))
                if sensing.get("sensesRoomProperty")
                else self.appliance_foi_uri(room_id, device_id))
            if sensing.get("sensesRoomProperty"):
                g.add((art, SOSA.observes, observed))
                # The inverse is the direction a planner queries: "who can tell
                # me this room's temperature?"
                g.add((observed, SOSA.isObservedBy, art))
            else:
                # The appliance interior is its own feature of interest, so the
                # reading is anchored to something real rather than dangling.
                interior = observed
                g.add((interior, RDF.type, SOSA.FeatureOfInterest))
                g.add((interior, RDFS.label,
                       Literal(f"{device_id} interior")))
                g.add((art, SOSA.hasFeatureOfInterest, interior))
                interior_property = URIRef(
                    f"{self.artifact_path(room_id, device_id)}"
                    f"#{sensing['sosa_property']}")
                g.add((interior_property, RDF.type, SOSA.ObservableProperty))
                prop_class = curie_to_uri(
                    (self.m.observable_properties.get(
                        f"room_state.{'temperature' if sensing['sosa_property'] == 'air_temperature' else 'humidity'}")
                     or {}).get("homeont_class"))
                if prop_class is not None:
                    g.add((interior_property, RDF.type, prop_class))
                g.add((interior, SSN.hasProperty, interior_property))
                g.add((interior_property, SSN.isPropertyOf, interior))
                g.add((art, SOSA.observes, interior_property))
                g.add((interior_property, SOSA.isObservedBy, art))

        classified = classify_device(device.get("attributes") or {})

        # Limit attributes (MinTemperature/MaxTemperature and friends) bound a
        # sibling on the same endpoint and cluster. Index them so a property can
        # state its own device's permitted range.
        siblings: Dict[str, Dict[str, Any]] = {}
        for record in classified.values():
            key = f"{record.get('endpoint')}.{record.get('cluster')}"
            siblings.setdefault(key, {})[str(record.get("attribute"))] = record.get("value")

        for attr_path, record in sorted(classified.items()):
            if record["role"] in ("drop", "bound"):
                # `bound` records were consumed into the sibling map above and
                # surface as js:minimum/js:maximum on the attribute they bound,
                # never as readable properties of their own.
                continue
            if record["role"] == "thing_metadata":
                # BasicInformation contributes nothing to the served TD. The
                # device's human-readable type is its rdfs:label (from the
                # curated family table, identical to ProductName on every
                # device in the corpus) and its name is its td:title, the
                # SimuHome device id. Vendor and product registration details
                # are not something a planner acts on.
                continue
            key = f"{record.get('endpoint')}.{record.get('cluster')}"
            self._attribute_property(g, art, path, attr_path, record,
                                     siblings.get(key), device_type, room_id)

        self._subscribe_actions(g, art, f"{path}#artifact", artifact=True)

        profile = URIRef(path)
        g.add((profile, RDF.type, HMAS.ResourceProfile))
        g.add((profile, HMAS.isProfileOf, art))
        return g

    # -- property emission -------------------------------------------------

    def _attribute_property(self, g: Graph, art: URIRef, path: str,
                            attr_path: str, record: Dict[str, Any],
                            siblings: Optional[Dict[str, Any]] = None,
                            device_type: str = "", room_id: str = "") -> None:
        cluster = record.get("cluster")
        attribute = record.get("attribute")
        name = self.m.affordance_name(str(cluster), str(attribute))
        row = self.m.attribute(str(cluster), str(attribute)) or {}

        prop = BNode()
        g.add((art, TD.hasPropertyAffordance, prop))
        g.add((prop, RDF.type, TD.PropertyAffordance))
        g.add((prop, TD.name, Literal(name)))
        g.add((prop, TD.title, Literal(name)))
        g.add((prop, TD.isObservable, Literal(True, datatype=XSD.boolean)))

        # Actuatable attributes are typed as such now; their invocation forms
        # come in phase 3. `readOnly` stays true until then, so nothing claims
        # a write path that does not yet exist.
        #
        # The dispatch MECHANISM is deliberately not stated here. Whether SHTD
        # reaches the simulator by cluster command or attribute write is this
        # server's routing concern, not a fact about the device: a Home
        # Assistant or native-Matter deployment serving the same vocabulary has
        # no value to put in such a field. `classify.py` still returns it and
        # `shtd.py` still routes on it -- it simply stops reaching the graph.
        actuatable = record.get("affordance") == "actuatable"
        if actuatable:
            g.add((prop, RDF.type, TDSOSA.ActuatablePropertyAffordance))
        else:
            g.add((prop, RDF.type, TDSOSA.ObservablePropertyAffordance))

        # WHAT THE PROPERTY IS, as a class rather than a string. This is the
        # channel the agent layer already reads: `integration_engine.py` treats
        # every rdf:type but the td:PropertyAffordance marker as a semantic
        # type, and the planner is instructed to match on those.
        self._property_class(g, prop, record, row, device_type)

        # No Matter cluster/attribute triples: `td:name` is unique within a
        # Thing (enforced in attribute_map.yaml, which qualifies the three names
        # that would otherwise collide on one device), and SHTD resolves it back
        # to cluster/attribute/endpoint from the registry when it dispatches.
        # Restating them here would spend planner context on protocol plumbing.
        # `homeont:matterType` stays on the schema -- it distinguishes
        # `power-mW` from `uint32`, which nothing else in the TD records.
        schema = self._value_schema(g, record, row, siblings)
        g.add((prop, TD.hasOutputSchema, schema))

        form = BNode()
        g.add((prop, TD.hasForm, form))
        g.add((form, HTV.methodName, Literal("GET")))
        g.add((form, HCTL.hasTarget, URIRef(f"{path}/properties/{name}")))
        g.add((form, HCTL.forContentType, Literal("application/json")))
        g.add((form, HCTL.hasOperationType, TD.readProperty))

        if actuatable:
            self._action_affordance(g, art, path, name, record, row, siblings,
                                    device_type, room_id)

    def _property_class(self, g: Graph, prop: BNode, record: Dict[str, Any],
                        row: Dict[str, Any], device_type: str) -> None:
        """Type the affordance with what the property IS.

        Leaf first (`homeont:AirConditionerOnOff`), else the generic class
        (`homeont:OnOff`), else the observable class. Each closes on a standard
        root -- sosa:ObservableProperty, sosa:ActuatableProperty, saref:State or
        ssn-system:SystemCapability -- so a consumer that has never heard of
        SimuHome can still tell a reading from a capability declaration.
        """
        curie = self.m.property_class(
            device_type, str(record.get("cluster")), str(record.get("attribute")))
        uri = curie_to_uri(curie)
        if uri is not None:
            g.add((prop, RDF.type, uri))

    def _action_class(self, g: Graph, action: BNode, record: Dict[str, Any],
                      row: Dict[str, Any], device_type: str) -> None:
        """Type the action with the operation it performs, as a saref:Command.

        Keyed by the attribute the action drives, matching how the affordance
        is keyed: `onOff(true|false)` stands for On/Off/Toggle alike.
        """
        curie = self.m.action_class(
            device_type, str(record.get("cluster")), str(record.get("attribute")))
        uri = curie_to_uri(curie)
        if uri is not None:
            g.add((action, RDF.type, uri))

    def _action_affordance(self, g: Graph, art: URIRef, path: str, name: str,
                           record: Dict[str, Any], row: Dict[str, Any],
                           siblings: Optional[Dict[str, Any]],
                           device_type: str = "", room_id: str = "") -> None:
        """The invocation half of an actuatable property.

        One flat action name -- the same `td:name` the property uses -- so a
        planner that has to construct a URL needs only
        `…/artifacts/{device}/actions/{name}`. SHTD resolves the name back to
        cluster / attribute / endpoint and picks the SimuHome endpoint from the
        mechanism, so the planner never learns that OnOff is driven by a command
        while PercentSetting is a plain write.
        """
        action = BNode()
        g.add((art, TD.hasActionAffordance, action))
        g.add((action, RDF.type, TD.ActionAffordance))
        g.add((action, TD.name, Literal(name)))
        g.add((action, TD.title, Literal(name)))
        g.add((action, TD.isIdempotent, Literal(True, datatype=XSD.boolean)))
        # Not safe in the WoT sense: invoking it changes the world.
        g.add((action, TD.isSafe, Literal(False, datatype=XSD.boolean)))

        # The operation this action performs, as a saref:Command subclass. The
        # dispatch mechanism is NOT stated -- see `_attribute_property`. The
        # docstring above already says the command/attribute-write split "is
        # invisible to the planner by design"; publishing it as a triple
        # contradicted that.
        self._action_class(g, action, record, row, device_type)

        # The input schema is the value the action accepts, so it carries the
        # same permitted values and bounds the read schema does -- that is what
        # lets a planner choose a legal value instead of guessing.
        schema = self._value_schema(g, record, row, siblings)
        g.add((action, TD.hasInputSchema, schema))

        form = BNode()
        g.add((action, TD.hasForm, form))
        g.add((form, HTV.methodName, Literal("POST")))
        g.add((form, HCTL.hasTarget, URIRef(f"{path}/actions/{name}")))
        g.add((form, HCTL.forContentType, Literal("application/json")))
        g.add((form, HCTL.hasOperationType, TD.invokeAction))

        self._action_effects(g, action, record, device_type, room_id)

    def _action_effects(self, g: Graph, action: BNode, record: Dict[str, Any],
                        device_type: str, room_id: str) -> None:
        """Which environmental variable this action can move, from the approved
        `actuation_effects` table.

        A HINT, not a recipe: it narrows a planner's search to the commands that
        can influence a variable. Working out that cooling needs SystemMode=3
        *plus* a setpoint below current, in the right order, is the planner's
        job and is what the benchmark measures.

        Direction follows the rule already settled: On/Off commands carry one;
        percentage- and level-wise setters are directionless, because the
        outcome depends on the value carried rather than on invoking the action.
        """
        cluster = str(record.get("cluster") or "")
        attribute = str(record.get("attribute") or "")
        mechanism = str(record.get("mechanism") or "")

        # The table mixes two row kinds, and they match an affordance
        # differently:
        #
        #   command rows        -- keyed by a cluster COMMAND. The affordance is
        #                          driven by that cluster's commands, so any
        #                          command row of the cluster applies.
        #   attribute_write rows -- keyed by the ATTRIBUTE written. Only the row
        #                          naming THIS attribute applies; matching on
        #                          cluster alone would give FanControl's
        #                          SpeedSetting the effect that belongs to
        #                          FanMode and PercentSetting.
        rows = []
        for row in self.m.effects_for(device_type, cluster):
            command = row.get("command") or {}
            if command.get("mechanism") == "attribute_write":
                if command.get("name") == attribute:
                    rows.append(row)
            elif mechanism == "command":
                rows.append(row)
        if not rows:
            return
        # The table is keyed by COMMAND, and per command the direction is right:
        # `On` increases illuminance, `Off` decreases it. But an affordance here
        # is keyed by the ATTRIBUTE and carries a VALUE -- `onOff(true|false)`
        # drives all six On/Off commands -- so its direction depends on the
        # value supplied, not on invoking it.
        #
        # That is the rule already settled for setpoints, applied one level up:
        # a value-carrying setter is directionless. Emitting increase AND
        # decrease on one action would have it claim both at once.
        #
        # A direction survives only where every command that drives this
        # attribute agrees on it.
        claims: Dict[str, Set[Optional[str]]] = {}
        for row in rows:
            claims.setdefault(
                str(row["affectsObservableProperty"]), set()).add(row.get("direction"))

        for sosa_property in sorted(claims):
            directions = claims[sosa_property]
            direction = directions.pop() if len(directions) == 1 else None
            observed = self.room_property_uri(room_id, sosa_property)
            actuation = BNode()
            g.add((action, TDSOSA.hasEffectActuation, actuation))
            g.add((actuation, RDF.type, SOSA.Actuation))
            g.add((actuation, SOSA.actsOnProperty, observed))
            if direction == "increase":
                g.add((actuation, RDF.type, TDSOSA.IncreasingActuation))
                g.add((actuation, TDSOSA.increasesObservableProperty, observed))
            elif direction == "decrease":
                g.add((actuation, RDF.type, TDSOSA.DecreasingActuation))
                g.add((actuation, TDSOSA.decreasesObservableProperty, observed))
            else:
                g.add((actuation, TDSOSA.affectsObservableProperty, observed))

    # Attributes whose min/max bound another attribute of the same cluster.
    # Verified present per-device in the benchmark corpus; the simulator carries
    # them in the same units as the value they bound.
    _LIMIT_PAIRS = {
        ("TemperatureControl", "TemperatureSetpoint"):
            ("MinTemperature", "MaxTemperature"),
        ("LevelControl", "CurrentLevel"): ("MinLevel", "MaxLevel"),
        ("TemperatureMeasurement", "MeasuredValue"):
            ("MinMeasuredValue", "MaxMeasuredValue"),
        ("RelativeHumidityMeasurement", "MeasuredValue"):
            ("MinMeasuredValue", "MaxMeasuredValue"),
    }

    def _enum_description(self, cluster: str, attribute: str) -> Optional[str]:
        """Spell out what each permitted integer means, in one sentence.

        `SystemMode = 3` is opaque; "3 = Cool (cooling demand only)" is not. The
        wire value is always the integer -- the name is only how a planner
        recognises which integer it wants.

        The gloss is included where the spec has one that says something the
        name does not. Items whose summary is a bare cross-reference into the
        PDF spec were already dropped by the normalizer.
        """
        items = load_registry().enum_items(cluster, attribute)
        if not items:
            return None
        parts = []
        for item in items:
            summary = (item.get("summary") or "").strip().rstrip(".")
            label = f"{item['value']} = {item['name']}"
            if summary and summary.lower() != str(item["name"]).lower():
                label += f" ({summary[0].lower() + summary[1:]})"
            parts.append(label)
        return ". ".join(parts) + "."

    def _add_limits(self, g: Graph, schema: BNode, cluster: str, attribute: str,
                    siblings: Dict[str, Any]) -> None:
        """Attach this device's permitted range as a schema RESTRICTION.

        A bound is a constraint on the value, so it belongs in the schema as
        `js:minimum`/`js:maximum` -- machine-checkable, and the standard WoT
        JSON Schema vocabulary. Stating it as prose, or exposing MinTemperature
        as a readable property of its own, models a constraint as if it were
        state and leaves a validator with nothing to act on.

        The bounds are per-instance: one washer's TemperatureControl range is
        not another's. Home Assistant could not carry them at all, which is part
        of why the TD path exists.
        """
        pair = self._LIMIT_PAIRS.get((cluster, attribute))
        if pair is None:
            return
        low, high = (siblings.get(pair[0]), siblings.get(pair[1]))
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            return
        if isinstance(low, bool) or isinstance(high, bool):
            return
        matter_type = load_registry().attribute_type(cluster, attribute) or ""
        if str(matter_type).startswith("temperature"):
            # Reported in human units, so the bounds must be too.
            low, high = round(low / 100.0, 2), round(high / 100.0, 2)
        g.add((schema, JS.minimum, _literal(low)))
        g.add((schema, JS.maximum, _literal(high)))

    def _value_schema(self, g: Graph, record: Dict[str, Any],
                      row: Dict[str, Any],
                      siblings: Optional[Dict[str, Any]] = None) -> BNode:
        """The SHAPE of the value -- never the value itself.

        A schema is a static description of what a reading looks like: its type,
        its unit, its scale. The reading is obtained by dereferencing the
        property's `readProperty` form. Embedding a value here would freeze an
        instant of a continuously ticking simulation into a cached document, and
        an agent trusting it would plan against a world that no longer exists.

        The type is therefore derived from the Matter type in the registry, not
        from whatever the value happened to be when this graph was built.
        """
        schema = BNode()
        matter_type = str(record.get("type") or "")
        value = record.get("value")

        cluster_name = str(record.get("cluster") or "")
        attribute_name = str(record.get("attribute") or "")
        registry = load_registry()

        # Matter names its composite types rather than declaring a primitive:
        # "SystemModeEnum", "ChannelInfoStruct", "AlarmBitmap". None of those
        # look like a primitive, so each must be resolved against the registry
        # or it falls through to StringSchema -- which would be a lie about a
        # value that is actually an integer or an object.
        enum_items = registry.enum_items(cluster_name, attribute_name)
        struct_fields = registry.struct_fields(cluster_name, matter_type)
        entry_type = registry.entry_type(cluster_name, attribute_name)
        is_bitmap = matter_type.endswith(("Bitmap", "bitmap")) or matter_type.startswith("map")

        if enum_items or is_bitmap:
            # Both are integers on the wire; a bitmap is flags OR'd together.
            g.add((schema, RDF.type, JS.IntegerSchema))
        elif struct_fields:
            g.add((schema, RDF.type, JS.ObjectSchema))
        elif matter_type in ("list", "array") or entry_type:
            g.add((schema, RDF.type, JS.ArraySchema))
        elif matter_type == "bool":
            g.add((schema, RDF.type, JS.BooleanSchema))
        elif matter_type.startswith(("single", "double", "float")):
            g.add((schema, RDF.type, JS.NumberSchema))
        elif _is_integer_type(matter_type):
            g.add((schema, RDF.type, JS.IntegerSchema))
        elif matter_type:
            g.add((schema, RDF.type, JS.StringSchema))
        else:
            # No registry type: fall back to the observed value's Python type.
            # This is a shape hint of last resort, still not the value.
            if isinstance(value, bool):
                g.add((schema, RDF.type, JS.BooleanSchema))
            elif isinstance(value, float):
                g.add((schema, RDF.type, JS.NumberSchema))
            elif isinstance(value, int):
                g.add((schema, RDF.type, JS.IntegerSchema))
            elif isinstance(value, (list, tuple)):
                g.add((schema, RDF.type, JS.ArraySchema))
            else:
                g.add((schema, RDF.type, JS.StringSchema))

        if matter_type:
            g.add((schema, HOME.matterType, Literal(matter_type)))
        # No scale declaration. The value a read returns is already converted
        # to the human unit named by `qudt:unit`, so the centi-storage is
        # SimuHome's wire format and stops at the mapper -- a Home Assistant or
        # native-Matter deployment serving this vocabulary has nothing to put
        # in such a field.
        unit = row.get("unit")
        if unit:
            _add_unit(g, schema, unit)

        # What the value MEANS, for an LLM planner that has to choose one.
        # Two sources, both machine-derived: the spec's enum items, and the
        # device's own declared limits.
        # A struct's keys, so "object" is not all the planner is told. Field
        # types come from the registry; nested structs/enums are named by their
        # Matter type rather than expanded, which keeps the graph finite.
        for field in struct_fields:
            field_node = BNode()
            g.add((schema, JS.properties, field_node))
            g.add((field_node, JS.propertyName, Literal(str(field["name"]))))
            field_type = str(field.get("type") or "")
            if field_type == "bool":
                g.add((field_node, RDF.type, JS.BooleanSchema))
            elif field_type.startswith(("uint", "int")) or field_type.endswith("Enum"):
                g.add((field_node, RDF.type, JS.IntegerSchema))
            elif field_type.startswith(("single", "double", "float")):
                g.add((field_node, RDF.type, JS.NumberSchema))
            elif field_type.endswith("Struct"):
                g.add((field_node, RDF.type, JS.ObjectSchema))
            else:
                g.add((field_node, RDF.type, JS.StringSchema))
            if field_type:
                g.add((field_node, HOME.matterType, Literal(field_type)))
            if field.get("mandatory"):
                g.add((schema, JS.required, Literal(str(field["name"]))))

        if entry_type:
            g.add((schema, HOME.matterEntryType, Literal(str(entry_type))))

        sentences = []
        if enum_items:
            for item in enum_items:
                g.add((schema, JS.enum, _literal(item["value"])))
            sentences.append(self._enum_description(cluster_name, attribute_name) or "")

        self._add_limits(g, schema, cluster_name, attribute_name, siblings or {})

        description = " ".join(s for s in sentences if s).strip()
        if description:
            g.add((schema, JS.description, Literal(description)))
        return schema

    def _room_state_properties(self, g: Graph, room_id: str,
                               state: Dict[str, Any], foi: URIRef) -> None:
        """The four environmental variables, as properties of the room.

        Only their shape is described here; the values are read through each
        property's form, since they change on every simulator tick.
        """
        for token in sorted(state):
            row = self.m.room_state_property(token) or {}
            name = str(row.get("sosa_property") or token)
            prop_class = curie_to_uri(row.get("homeont_class"))

            # A stable IRI, not a blank node: a sensor in another document has
            # to name the very property it observes, and an actuation the one it
            # moves. A blank node cannot be referenced across documents.
            prop = self.room_property_uri(room_id, name)
            g.add((foi, SSN.hasProperty, prop))
            g.add((prop, RDF.type, TD.PropertyAffordance))
            g.add((prop, RDF.type, TDSOSA.ObservablePropertyAffordance))
            g.add((prop, RDF.type, SOSA.ObservableProperty))
            if prop_class is not None:
                g.add((prop, RDF.type, prop_class))
            g.add((prop, TD.name, Literal(name)))
            g.add((prop, TD.title, Literal(name)))
            g.add((prop, TD.isObservable, Literal(True, datatype=XSD.boolean)))
            # `ssn:isPropertyOf` is the correct link: this property belongs to
            # the environment. (`sosa:isObservedBy` would name the SENSOR that
            # observes it, not the feature of interest -- pointing it back at
            # the environment made it its own observer.)
            g.add((prop, SSN.isPropertyOf, foi))

            qk = curie_to_uri(row.get("quantityKind"))
            if qk is not None:
                g.add((prop, QUDT.hasQuantityKind, qk))
            qunit = curie_to_uri(row.get("qudt_unit"))
            if qunit is not None:
                g.add((prop, QUDT.unit, qunit))

            schema = BNode()
            g.add((schema, RDF.type, JS.NumberSchema))
            if row.get("unit"):
                _add_unit(g, schema, row["unit"])
            # No scale declaration here either: `scale_room_state` has already
            # divided, and `qudt:unit` names the unit the value is in.
            g.add((prop, TD.hasOutputSchema, schema))

            form = BNode()
            g.add((prop, TD.hasForm, form))
            g.add((form, HTV.methodName, Literal("GET")))
            g.add((form, HCTL.hasTarget,
                   URIRef(f"{self.room_path(room_id)}/properties/{name}")))
            g.add((form, HCTL.forContentType, Literal("application/json")))
            g.add((form, HCTL.hasOperationType, TD.readProperty))

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def scale_room_state(token: str, value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and token in CENTI_TOKENS:
            return round(value / 100.0, 2)
        if isinstance(value, float):
            return round(value, 2)
        return value

    def _room_class(self, room_id: str) -> Optional[URIRef]:
        """The homeont class for a SimuHome room id.

        The approved `room_map` names each room as a `homeont:` class, which is
        the spatial vocabulary this server serves. Mapping by local name here
        keeps the reviewed table as the source of WHICH room a token is, while
        homeont supplies the class.
        """
        curie = (self.m.room(room_id) or {}).get("homeont_class")
        if not curie or ":" not in str(curie):
            return None
        local = str(curie).split(":", 1)[1]
        return HOME[local] if local in HOME_SPACE_CLASSES else None

    def _touches_room_environment(self, device_type: str) -> bool:
        """Does this family perceive or affect one of the room's variables?

        Decided by the approved `actuation_effects` table, not by whether the
        device happens to carry a temperature attribute -- freezers and
        refrigerators do, but of their own compartment.
        """
        family = self.m.device_family(device_type) or {}
        key = str(family.get("key") or "")
        return key in self.m.families_affecting_environment

    def _room_title(self, room_id: str) -> str:
        row = self.m.room(room_id) or {}
        return str(row.get("title") or room_id.replace("_", " ").title())

    def _subscribe_actions(self, g: Graph, subject: URIRef, topic: str,
                           *, artifact: bool) -> None:
        """Subscribe/unsubscribe affordances for a workspace or an artifact.

        `focus` was a JaCaMo notion. On the web a TD says plainly what it
        offers: subscribing to a WORKSPACE yields notifications from every
        device in it (and its environment); subscribing to an ARTIFACT yields
        only that artifact's own state changes. Either way a notification
        carries any internal state change.

        Both forms target `/hub/` and name the resource in `hub.topic`, which is
        plain WebSub and matches how the agent layer already reads hasp.py. The
        input schema pins the topic to THIS resource, so an agent copies the
        affordance rather than constructing a subscription by hand.
        """
        pairs = (
            ("subscribeToArtifact", WEBSUB.subscribeToArtifact, "subscribe"),
            ("unsubscribeFromArtifact", WEBSUB.unsubscribeFromArtifact, "unsubscribe"),
        ) if artifact else (
            ("subscribeToWorkspace", WEBSUB.subscribeToWorkspace, "subscribe"),
            ("unsubscribeFromWorkspace", WEBSUB.unsubscribeFromWorkspace, "unsubscribe"),
        )
        for name, type_uri, mode in pairs:
            act = BNode()
            g.add((subject, TD.hasActionAffordance, act))
            g.add((act, RDF.type, TD.ActionAffordance))
            g.add((act, RDF.type, type_uri))
            g.add((act, TD.name, Literal(name)))
            g.add((act, TD.title, Literal(name)))
            g.add((act, TD.hasInputSchema,
                   self._hub_input_schema(g, topic, mode)))
            form = BNode()
            g.add((act, TD.hasForm, form))
            g.add((form, HTV.methodName, Literal("POST")))
            g.add((form, HCTL.hasTarget, URIRef(f"{self.base}hub/")))
            g.add((form, HCTL.forContentType, Literal("application/json")))
            g.add((form, HCTL.hasOperationType, TD.invokeAction))
            g.add((form, HCTL.forSubProtocol, Literal("websub")))

    def _hub_input_schema(self, g: Graph, topic: str, mode: str) -> BNode:
        """The `/hub/` body, with `hub.mode` and `hub.topic` fixed for this
        resource so only the callback is the agent's to supply."""
        schema = BNode()
        g.add((schema, RDF.type, JS.ObjectSchema))
        for prop_name, const, required in (
            ("hub.mode", mode, True),
            ("hub.topic", topic, True),
            ("hub.callback", None, True),
            ("hub.lease_seconds", None, False),
        ):
            prop = BNode()
            g.add((schema, JS.properties, prop))
            g.add((prop, JS.propertyName, Literal(prop_name)))
            if prop_name == "hub.lease_seconds":
                g.add((prop, RDF.type, JS.IntegerSchema))
            else:
                g.add((prop, RDF.type, JS.StringSchema))
            if const is not None:
                g.add((prop, JS.const, Literal(const)))
            if required:
                g.add((schema, JS.required, Literal(prop_name)))
        return schema

    def find_device(self, room_id: str, device_id: str) -> Optional[Dict[str, Any]]:
        for device in (self.rooms.get(room_id) or {}).get("devices") or []:
            if str(device.get("device_id")) == device_id:
                return device
        return None

    def device_count(self) -> int:
        return sum(len(r.get("devices") or []) for r in self.rooms.values())

    def room_ids(self) -> List[str]:
        return sorted(self.rooms)


def serialize(g: Graph) -> str:
    out = g.serialize(format="turtle")
    return out.decode() if isinstance(out, bytes) else out
