"""HASP's cache of one Home Assistant snapshot and the documents built from it.

The registries (areas, devices, entities), states and service definitions are
read once (`refresh`) and kept up to date from HA's state events; every TD HASP
serves is built from them (`_rebuild_documents_unlocked`), lazily and at most
every `min_rebuild_interval_s`.

One artifact is one HA device, holding all of its entities (`ArtifactEntry`);
an entity with no device is an artifact of its own. Building an artifact's TD
also yields its affordance map, name -> `Affordance` (`hasp_utils`), which is
how a request to `.../artifacts/{a}/properties/{name}` or `.../actions/{name}`
finds its entity. Example: the Effekta inverter's TD lists
`charger_source_priority_select_option`; a POST to that action looks it up and
gets `select.select_option` on `select.solarassistant_effekta_charger_source_priority`.
"""
from __future__ import annotations

import asyncio
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from rdflib import Graph, Literal, RDF, URIRef

try:
    from .hasp_utils import Affordance, HomeAssistantRDF, slugify
    from . import hasp_semantics
    from .hasp_namespaces import HCTL, HMAS, SCHEMA, SOSA, SSN, TD, TDSOSA
except ImportError:  # pragma: no cover
    from hasp_utils import Affordance, HomeAssistantRDF, slugify
    import hasp_semantics
    from hasp_namespaces import HCTL, HMAS, SCHEMA, SOSA, SSN, TD, TDSOSA


# Builds one artifact's TD: (workspace id, slug, display name, its entities,
# states by entity id, its device registry entry) -> (Turtle, name -> Affordance).
ArtifactBuilder = Callable[..., Tuple[str, Dict[str, Affordance]]]


@dataclass
class ArtifactEntry:
    """One artifact of a workspace: an HA device and its entities.

    `slug` is the stable instance name (URL segment, `td:title`);
    `display_name` HA's own, mutable name (`schema:name`); `device` the
    registry entry, empty for an entity with no device.
    """
    slug: str
    display_name: str
    device: Dict[str, Any]
    entities: List[Dict[str, Any]] = field(default_factory=list)


class UnknownAffordance(KeyError):
    """An affordance name the artifact does not have; carries the names it has,
    so the 404 can list them."""

    def __init__(self, name: str, available: List[str]) -> None:
        super().__init__(name)
        self.name = name
        self.available = available


def _curie(uri: Any) -> str:
    """A homeont IRI as a CURIE (`HOMEONT.Illuminance` -> `homeont:Illuminance`);
    any other IRI as it is (matches nothing in the homeont tables)."""
    text = str(uri)
    homeont = str(hasp_semantics.HOMEONT)
    return "homeont:" + text[len(homeont):] if text.startswith(homeont) else text


def _canonical_label(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


def _normalize_workspace_id(area_id: Optional[str], allowed_workspaces: set[str]) -> Optional[str]:
    if not area_id:
        return None
    if not allowed_workspaces:
        return area_id
    for ws in allowed_workspaces:
        if area_id == ws or area_id.startswith(ws + "_"):
            return ws
    return area_id


def _area_matches(workspace_id: str, candidate: Optional[str], allowed_workspaces: set[str]) -> bool:
    if not candidate:
        return False
    if candidate == workspace_id or candidate.startswith(workspace_id + "_"):
        return True
    norm = _normalize_workspace_id(candidate, allowed_workspaces)
    return bool(norm and norm == workspace_id)


def _workspace_allowed(area_id: Optional[str], allowed_workspaces: set[str]) -> bool:
    if not allowed_workspaces:
        return True
    normalized = _normalize_workspace_id(area_id, allowed_workspaces)
    return bool(normalized and normalized in allowed_workspaces)


def _entity_matches_workspace(entity_id: Optional[str], workspace_id: str) -> bool:
    if not entity_id or "." not in entity_id:
        return False
    object_id = entity_id.split(".", 1)[1]
    return object_id.startswith(workspace_id + "_") or object_id.startswith(workspace_id)


def _entity_display_name(entity: Optional[Dict[str, Any]], devices_by_id: Dict[str, Dict[str, Any]]) -> str:
    if not entity:
        return ""
    for key in ("name", "original_name"):
        val = entity.get(key)
        if isinstance(val, str) and val.strip():
            name = val.strip()
            # Clean up duplicates like "lights_308.lights_308" -> "lights_308"
            parts = name.split(".")
            if len(parts) == 2 and parts[0] == parts[1]:
                return parts[0]
            return name
    object_id = ""
    ent_id = entity.get("entity_id", "")
    if isinstance(ent_id, str) and "." in ent_id:
        object_id = ent_id.split(".", 1)[1]
    device = devices_by_id.get(entity.get("device_id"))
    device_name = (device or {}).get("name") if isinstance(device, dict) else None
    if device_name and object_id:
        # Don't duplicate if device_name already equals object_id
        if device_name == object_id:
            return device_name
        return f"{device_name}.{object_id}"
    if device_name:
        return device_name
    return object_id or ent_id or "artifact"


def _object_id(entity_id: str) -> str:
    """`light.desk_lamp` -> `desk_lamp`."""
    return entity_id.split(".", 1)[-1]


def _group_artifacts(entities: List[Dict[str, Any]],
                     dev_by_id: Dict[str, Dict[str, Any]]) -> List[ArtifactEntry]:
    """A workspace's entities grouped into artifacts, one per HA device.

    The slug is the device name's slug ("Blue Air Garage" -> `blue_air_garage`);
    an entity with no device is its own artifact, slugged by its object id. Two
    devices with one slug are told apart by a short device-id suffix. Sorted by
    slug, so listings and rebuilds are deterministic.
    """
    # 1. Group by device; an entity without one stands alone.
    groups: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]] = {}
    for ent in entities:
        entity_id = ent.get("entity_id") or ""
        dev_id = ent.get("device_id")
        key = f"device:{dev_id}" if dev_id in dev_by_id else f"entity:{entity_id}"
        device = dev_by_id.get(dev_id, {}) if dev_id in dev_by_id else {}
        groups.setdefault(key, (device, []))[1].append(ent)

    # 2. Name each group: its slug and HA's display name.
    entries: List[ArtifactEntry] = []
    taken: set[str] = set()
    for key in sorted(groups):
        device, members = groups[key]
        members.sort(key=lambda e: e.get("entity_id") or "")
        if device:
            base = slugify(device.get("name")) or slugify(device.get("name_by_user"))
            display = device.get("name_by_user") or device.get("name") or base
        else:
            entity_id = members[0].get("entity_id") or ""
            base = slugify(_object_id(entity_id))
            display = _entity_display_name(members[0], dev_by_id)
        base = base or "artifact"
        slug = base
        if slug in taken:
            slug = f"{base}_{slugify(str(device.get('id') or key))[:6]}"
        taken.add(slug)
        entries.append(ArtifactEntry(slug=slug, display_name=display,
                                     device=device, entities=members))

    # 3. Display names HA gives twice (device-less covers all called "Cover")
    #    get the slug appended, so each still names one artifact.
    counts: Dict[str, int] = {}
    for entry in entries:
        counts[entry.display_name] = counts.get(entry.display_name, 0) + 1
    for entry in entries:
        if counts[entry.display_name] > 1:
            entry.display_name = f"{entry.display_name} ({entry.slug})"
    return sorted(entries, key=lambda e: e.slug)


def _artifact_aliases(entry: ArtifactEntry) -> List[str]:
    """Every name a request may use for an artifact: its slug, the device's
    names, and each entity's id, object id and registry names (so a link to
    the entity an artifact used to be still finds its device)."""
    names = [entry.slug, entry.display_name,
             entry.device.get("name") or "", entry.device.get("name_by_user") or ""]
    for ent in entry.entities:
        entity_id = ent.get("entity_id") or ""
        names += [entity_id, _object_id(entity_id),
                  ent.get("name") or "", ent.get("original_name") or ""]
    return [n for n in names if n]


class HASPGraphCache:
    def __init__(
        self,
        *,
        ws_client: Any,
        rest_client: Any,
        base_uri: str,
        allowed_workspaces: Optional[set[str]] = None,
        artifact_builder: ArtifactBuilder,
    ) -> None:
        self.ws_client = ws_client
        self.rest_client = rest_client
        self.base_uri = base_uri.rstrip("/") + "/"
        self.allowed_workspaces = set(allowed_workspaces or set())
        self.artifact_builder = artifact_builder
        self._lock = asyncio.Lock()

        self.areas_by_id: Dict[str, Dict[str, Any]] = {}
        self.devices_by_id: Dict[str, Dict[str, Any]] = {}
        self.entities_by_id: Dict[str, Dict[str, Any]] = {}
        self.states_by_entity_id: Dict[str, Dict[str, Any]] = {}
        self.services_by_domain: Dict[str, Dict[str, Any]] = {}

        # Per workspace: its artifacts (one per HA device), by slug.
        self.workspace_artifacts: Dict[str, Dict[str, ArtifactEntry]] = {}

        self.platform_ttl: str = ""
        self.workspaces_ttl: str = ""
        self.workspace_ttls: Dict[str, str] = {}
        self.artifacts_ttls: Dict[str, str] = {}
        self.artifact_ttls: Dict[Tuple[str, str], str] = {}
        # (workspace, artifact slug) -> affordance name -> what it routes to;
        # filled with each TD, so the names served are the names routed.
        self.affordances: Dict[Tuple[str, str], Dict[str, Affordance]] = {}
        # (workspace, variable name) -> [(artifact, property)] that read it, a
        # sensor first: where the room-property route takes a variable's value.
        self.variable_readers: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        self.full_graph: Graph = Graph(base=self.base_uri)
        self._documents_dirty = False
        self._last_rebuild_monotonic = 0.0
        # With frequently-updating sensors the cache is dirty almost always;
        # without a floor, every TTL read pays a full rebuild and a workspace
        # crawl slows from seconds to minutes.
        self.min_rebuild_interval_s = 10.0

    async def refresh(self) -> None:
        # Home Assistant websocket registry calls share one connection and are
        # not safe to issue concurrently. Keep them ordered; only REST reads can
        # be fetched in parallel.
        areas = await self._call_or_default(
            self.ws_client, "get_areas", [], swallow_exceptions=False
        )
        devices = await self._call_or_default(
            self.ws_client, "get_devices", [], swallow_exceptions=False
        )
        entities = await self._call_or_default(
            self.ws_client, "get_entities", [], swallow_exceptions=False
        )
        states, services = await asyncio.gather(
            self._call_or_default(self.rest_client, "get_states", []),
            self._call_or_default(self.rest_client, "get_services", []),
        )
        async with self._lock:
            self._load_snapshot_unlocked(areas, devices, entities, states, services)
            self._rebuild_documents_unlocked()

    async def refresh_states(self) -> None:
        states = await self._call_or_default(self.rest_client, "get_states", [])
        async with self._lock:
            self.states_by_entity_id = {
                state.get("entity_id"): dict(state)
                for state in states
                if isinstance(state, dict) and state.get("entity_id")
            }
            self._rebuild_documents_unlocked()

    @staticmethod
    async def _call_or_default(
        target: Any,
        method_name: str,
        default: Any,
        *,
        swallow_exceptions: bool = True,
    ) -> Any:
        method = getattr(target, method_name, None)
        if method is None:
            return default
        try:
            return await method()
        except Exception:
            if not swallow_exceptions:
                raise
            return default

    async def apply_state_change(self, entity_id: str, new_state: Optional[Dict[str, Any]]) -> None:
        async with self._lock:
            if new_state and isinstance(new_state, dict):
                self.states_by_entity_id[entity_id] = dict(new_state)
            else:
                self.states_by_entity_id.pop(entity_id, None)
            # Rebuilding the RDF documents costs one TD build + turtle parse per
            # entity. Doing it here for every state_changed event starves the
            # event loop when sensors update frequently, so defer to readers.
            self._documents_dirty = True

    async def apply_service_result(self, result: Any) -> None:
        updated = False
        async with self._lock:
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and item.get("entity_id"):
                        self.states_by_entity_id[item["entity_id"]] = dict(item)
                        updated = True
            elif isinstance(result, dict) and result.get("entity_id"):
                self.states_by_entity_id[result["entity_id"]] = dict(result)
                updated = True
            if updated:
                self._documents_dirty = True
                return
        await self.refresh_states()

    def _ensure_documents_unlocked(self) -> None:
        if not self._documents_dirty:
            return
        if time.monotonic() - self._last_rebuild_monotonic < self.min_rebuild_interval_s:
            return
        self._rebuild_documents_unlocked()

    async def has_workspace(self, workspace_id: str) -> bool:
        async with self._lock:
            return workspace_id in self.areas_by_id

    async def get_platform_ttl(self) -> str:
        async with self._lock:
            self._ensure_documents_unlocked()
            return self.platform_ttl

    async def serialize_full_graph(self, fmt: str = "turtle") -> str:
        async with self._lock:
            self._ensure_documents_unlocked()
            return self.full_graph.serialize(format=fmt)

    def variable_uri(self, workspace_id: str, observable_property: str) -> str:
        """A variable's IRI from an IRI, a name (`illuminance`) or an older
        key (`luminosity`, mapped by `variable_aliases`)."""
        if observable_property.startswith(("http://", "https://")):
            return observable_property
        name = hasp_semantics.current().variable_name(observable_property)
        return str(hasp_semantics.variable_uri(self.base_uri, workspace_id, name))

    async def query_actions_affecting_observable_property(
        self, workspace_id: str, observable_property: str
    ) -> Dict[str, Any]:
        """The actions of a workspace's artifacts that move one variable, with
        their direction, and the variable's comfort range."""
        # 1. The variable, and the two queries over the workspace's graph.
        property_uri = self.variable_uri(workspace_id, observable_property)
        artifact_prefix = f"{self.base_uri}workspaces/{workspace_id}/artifacts/"
        action_query = f"""
        PREFIX td: <{TD}>
        PREFIX hctl: <{HCTL}>
        PREFIX sosa: <{SOSA}>
        PREFIX tdsosa: <{TDSOSA}>

        SELECT DISTINCT ?artifact ?artifactTitle ?actionName ?actionTarget ?actuation ?dirType
        WHERE {{
          ?artifact td:hasActionAffordance ?action .
          ?action td:name ?actionName ;
                  td:hasForm ?form ;
                  tdsosa:hasEffectActuation ?actuation .
          ?form hctl:hasTarget ?actionTarget .
          ?actuation sosa:actsOnProperty <{property_uri}> .
          OPTIONAL {{ ?artifact td:title ?artifactTitle . }}
          OPTIONAL {{
            ?actuation a ?dirType .
            FILTER(?dirType IN (tdsosa:IncreasingActuation, tdsosa:DecreasingActuation))
          }}
          FILTER(STRSTARTS(STR(?artifact), "{artifact_prefix}"))
        }}
        ORDER BY ?artifact ?actionName
        """
        property_meta_query = f"""
        PREFIX tdsosa: <{TDSOSA}>

        SELECT ?targetMin ?targetMax ?targetUnit
        WHERE {{
          OPTIONAL {{ <{property_uri}> tdsosa:targetMinValue ?targetMin . }}
          OPTIONAL {{ <{property_uri}> tdsosa:targetMaxValue ?targetMax . }}
          OPTIONAL {{ <{property_uri}> tdsosa:targetUnit ?targetUnit . }}
        }}
        LIMIT 1
        """
        async with self._lock:
            self._ensure_documents_unlocked()
            rows = list(self.full_graph.query(action_query))
            meta_rows = list(self.full_graph.query(property_meta_query))

        # 2. One entry per action, with its direction when the actuation states one.
        results: List[Dict[str, Any]] = []
        for row in rows:
            direction = None
            if row.dirType == TDSOSA.IncreasingActuation:
                direction = "increase"
            elif row.dirType == TDSOSA.DecreasingActuation:
                direction = "decrease"
            item: Dict[str, Any] = {
                "artifact_uri": str(row.artifact),
                "artifact_title": str(row.artifactTitle) if row.artifactTitle is not None else "",
                "action_name": str(row.actionName),
                "action_target": str(row.actionTarget),
                "actuation_uri": str(row.actuation),
                "property_uri": property_uri,
                "direction": direction or "",
            }
            results.append(item)

        # 3. The variable's comfort range, when the home states one.
        target_min: Optional[float] = None
        target_max: Optional[float] = None
        target_unit = ""
        if meta_rows:
            meta_row = meta_rows[0]
            if meta_row.targetMin is not None:
                try:
                    target_min = float(meta_row.targetMin)
                except (TypeError, ValueError):
                    target_min = None
            if meta_row.targetMax is not None:
                try:
                    target_max = float(meta_row.targetMax)
                except (TypeError, ValueError):
                    target_max = None
            if meta_row.targetUnit is not None:
                target_unit = str(meta_row.targetUnit)

        return {
            "property_uri": property_uri,
            "target_min": target_min,
            "target_max": target_max,
            "target_unit": target_unit,
            "actions": results,
        }

    async def query_action_effects(
        self, workspace_id: str, action_url: str
    ) -> Dict[str, Any]:
        """Query TD-SOSA observable-property effects for one action target/URI."""
        # 1. The action's effect actuations, matched by its form target or IRI.
        action_url = str(action_url or "").strip()
        artifact_prefix = f"{self.base_uri}workspaces/{workspace_id}/artifacts/"
        action_query = f"""
        PREFIX td: <{TD}>
        PREFIX hctl: <{HCTL}>
        PREFIX sosa: <{SOSA}>
        PREFIX tdsosa: <{TDSOSA}>

        SELECT DISTINCT ?artifact ?artifactTitle ?action ?actionName ?actionTarget ?actuation ?property ?dirType
        WHERE {{
          ?artifact td:hasActionAffordance ?action .
          ?action td:name ?actionName ;
                  td:hasForm ?form ;
                  tdsosa:hasEffectActuation ?actuation .
          ?form hctl:hasTarget ?actionTarget .
          ?actuation sosa:actsOnProperty ?property .
          OPTIONAL {{ ?artifact td:title ?artifactTitle . }}
          OPTIONAL {{
            ?actuation a ?dirType .
            FILTER(?dirType IN (tdsosa:IncreasingActuation, tdsosa:DecreasingActuation))
          }}
          FILTER(STRSTARTS(STR(?artifact), "{artifact_prefix}"))
          FILTER(STR(?actionTarget) = "{action_url}" || STR(?action) = "{action_url}")
        }}
        ORDER BY ?property ?artifact ?actionName
        """
        async with self._lock:
            self._ensure_documents_unlocked()
            rows = list(self.full_graph.query(action_query))

        # 2. One entry per variable the action moves.
        effects: List[Dict[str, Any]] = []
        for row in rows:
            direction = ""
            if row.dirType == TDSOSA.IncreasingActuation:
                direction = "increase"
            elif row.dirType == TDSOSA.DecreasingActuation:
                direction = "decrease"
            item: Dict[str, Any] = {
                "artifact_uri": str(row.artifact),
                "artifact_title": str(row.artifactTitle) if row.artifactTitle is not None else "",
                "action_uri": str(row.action),
                "action_name": str(row.actionName),
                "action_target": str(row.actionTarget),
                "actuation_uri": str(row.actuation),
                "property_uri": str(row.property),
                "property_name": str(row.property).rsplit("#", 1)[-1],
                "direction": direction,
            }
            effects.append(item)

        return {
            "workspace_id": workspace_id,
            "action_url": action_url,
            "effects": effects,
        }

    async def get_workspaces_ttl(self) -> str:
        async with self._lock:
            self._ensure_documents_unlocked()
            return self.workspaces_ttl

    async def get_workspace_ttl(self, workspace_id: str) -> str:
        async with self._lock:
            self._ensure_documents_unlocked()
            return self.workspace_ttls[workspace_id]

    async def get_artifacts_ttl(self, workspace_id: str) -> str:
        async with self._lock:
            self._ensure_documents_unlocked()
            return self.artifacts_ttls[workspace_id]

    async def get_artifact_ttl(self, workspace_id: str, artifact_name: str) -> Tuple[str, str]:
        """The artifact's TD and HA's display name for it; KeyError when the
        workspace has no artifact by that name."""
        async with self._lock:
            self._ensure_documents_unlocked()
            entry = self._resolve_artifact_unlocked(workspace_id, artifact_name)
            return self.artifact_ttls[(workspace_id, entry.slug)], entry.display_name

    async def get_service_definition(self, domain: str, service: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return ((self.services_by_domain.get(domain) or {}).get("services") or {}).get(service)

    # -- affordances ---------------------------------------------------------

    async def resolve_affordance(self, workspace_id: str, artifact_name: str,
                                 affordance_name: str) -> Tuple[str, Affordance]:
        """The artifact's slug and what one of its affordance names routes to.

        KeyError when there is no such artifact; `UnknownAffordance` (a
        KeyError, carrying the names the artifact has) when it has no such
        affordance.
        """
        async with self._lock:
            return self._resolve_affordance_unlocked(workspace_id, artifact_name, affordance_name)

    def _resolve_affordance_unlocked(self, workspace_id: str, artifact_name: str,
                                     affordance_name: str) -> Tuple[str, Affordance]:
        """`resolve_affordance` without the lock."""
        # 1. The artifact; its map comes with its TD, so build that first if
        #    the cache has never built documents.
        entry = self._resolve_artifact_unlocked(workspace_id, artifact_name)
        if (workspace_id, entry.slug) not in self.affordances:
            self._rebuild_documents_unlocked()
        names = self.affordances.get((workspace_id, entry.slug), {})

        # 2. The name, as served in the TD (URL-decoded).
        name = urllib.parse.unquote(affordance_name)
        if name not in names:
            raise UnknownAffordance(name, sorted(names))
        return entry.slug, names[name]

    def entity_state(self, entity_id: str) -> Dict[str, Any]:
        """The cached HA state object of one entity ({} when it has none)."""
        return dict(self.states_by_entity_id.get(entity_id) or {})

    async def read_property(self, workspace_id: str, artifact_name: str, property_name: str) -> Any:
        """An artifact property's current value (`read_property_unlocked`)."""
        async with self._lock:
            return self._read_property_unlocked(workspace_id, artifact_name, property_name)

    async def read_room_property(self, workspace_id: str, variable_name: str) -> Any:
        """A room variable's current value: the reading of the first device
        that observes it (a sensor before, say, a thermostat). KeyError when
        no device observes it -- the workspace TD gives such a variable no form.
        """
        async with self._lock:
            self._ensure_documents_unlocked()
            name = hasp_semantics.current().variable_name(urllib.parse.unquote(variable_name))
            for artifact_name, property_name in self.variable_readers.get((workspace_id, name), []):
                try:
                    return self._read_property_unlocked(workspace_id, artifact_name, property_name)
                except KeyError:
                    continue        # no value right now; the next observer may have one
            raise KeyError(name)

    def _read_property_unlocked(self, workspace_id: str, artifact_name: str,
                                property_name: str) -> Any:
        """The value of one property of an artifact, from the cached HA state.

        The name is looked up in the artifact's affordance map: a `state`
        reads the entity's state (a number for a sensor or anything with a
        unit), an `attribute` that attribute, `metadata` the descriptive
        attributes and timestamps. KeyError when the artifact has no such
        property or the entity has no value for it right now.
        """
        # 1. What the name stands for; an action name is not readable.
        _, affordance = self._resolve_affordance_unlocked(workspace_id, artifact_name, property_name)
        if affordance.kind not in ("state", "attribute", "metadata"):
            raise KeyError(property_name)
        entity_state = self.states_by_entity_id.get(affordance.entity_id) or {}
        entity_attrs = entity_state.get("attributes") or {}

        # 2. Its descriptive attributes, with HA's timestamps.
        if affordance.kind == "metadata":
            metadata_attrs = {
                key: value for key, value in entity_attrs.items()
                if key in {
                    "friendly_name", "icon", "entity_picture", "supported_features",
                    "device_class", "state_class", "unit_of_measurement", "attribution",
                    "restored", "supported_color_modes",
                }
            }
            for ts_field in ("last_changed", "last_reported", "last_updated"):
                if ts_field in entity_state:
                    metadata_attrs[ts_field] = entity_state[ts_field]
            if metadata_attrs:
                return metadata_attrs
            raise KeyError(property_name)

        # 3. Its state: numeric for a measurement, as the TD's schema says.
        if affordance.kind == "state":
            state_value = entity_state.get("state")
            if not state_value or state_value in ("unknown", "unavailable"):
                raise KeyError(property_name)
            unit = entity_attrs.get("unit_of_measurement")
            if (affordance.domain == "sensor" or unit) and isinstance(state_value, str):
                try:
                    return int(state_value) if "." not in state_value else float(state_value)
                except ValueError:
                    return state_value
            return state_value

        # 4. One attribute.
        value = entity_attrs.get(affordance.attribute or "")
        if value is None:
            raise KeyError(property_name)
        return value

    # -- WebSub contexts -----------------------------------------------------

    def _context(self, workspace_id: str, entry: ArtifactEntry) -> Dict[str, str]:
        """The topics and names a notification about one artifact carries."""
        artifact_uri = f"{self.base_uri}workspaces/{workspace_id}/artifacts/{entry.slug}#artifact"
        return {
            "workspace_id": workspace_id,
            "artifact_title": entry.slug,
            "artifact_slug": entry.slug,
            "artifact_uri": artifact_uri,
            "workspace_topic": f"{self.base_uri.rstrip('/')}/workspaces/{workspace_id}",
            "artifact_topic": artifact_uri,
        }

    async def get_artifact_contexts(self, workspace_id: str, artifact_name: str) -> List[Dict[str, str]]:
        """The WebSub topics of one artifact (for `focus`); KeyError when the
        workspace has no artifact by that name."""
        async with self._lock:
            entry = self._resolve_artifact_unlocked(workspace_id, artifact_name)
            return [self._context(workspace_id, entry)]

    async def get_entity_contexts(self, entity_id: str) -> List[Dict[str, str]]:
        """The topics a state change of one entity is published on: those of
        the artifact (device) holding it, in every workspace that has it, with
        the name of the property that reads its state (`affordance`)."""
        async with self._lock:
            contexts: List[Dict[str, str]] = []
            for workspace_id, entries in self.workspace_artifacts.items():
                for entry in entries.values():
                    # 1. Only the artifact (device) that holds the entity.
                    if not any(e.get("entity_id") == entity_id for e in entry.entities):
                        continue
                    # 2. Its topics, and the property reading the entity's state.
                    context = self._context(workspace_id, entry)
                    names = self.affordances.get((workspace_id, entry.slug), {})
                    context["affordance"] = next(
                        (name for name, aff in names.items()
                         if aff.kind == "state" and aff.entity_id == entity_id), "")
                    contexts.append(context)
            return contexts

    def _load_snapshot_unlocked(self, areas, devices, entities, states, services) -> None:
        filtered_areas = [a for a in areas if _workspace_allowed(a.get("area_id"), self.allowed_workspaces)]
        self.areas_by_id = {a["area_id"]: dict(a) for a in filtered_areas}
        self.devices_by_id = {d["id"]: dict(d) for d in devices if d.get("id")}
        self.entities_by_id = {e["entity_id"]: dict(e) for e in entities if e.get("entity_id")}
        self.states_by_entity_id = {s["entity_id"]: dict(s) for s in states if isinstance(s, dict) and s.get("entity_id")}
        self.services_by_domain = {s["domain"]: dict(s) for s in services if s.get("domain")}
        self._rebuild_workspace_indexes_unlocked()

    def _rebuild_workspace_indexes_unlocked(self) -> None:
        """Which entities each workspace holds, grouped into its artifacts.

        An entity belongs to a workspace when it, or its device, is in the
        area (or its object id carries the area id); a device's entity
        assigned to another allowed area stays with that area. The workspace's
        entities are then grouped one artifact per device (`_group_artifacts`).
        """
        self.workspace_artifacts = {}
        devices = list(self.devices_by_id.values())
        entities = [dict(e) for e in self.entities_by_id.values()]
        for workspace_id in self.areas_by_id:
            # 1. Entities in the area, directly or through their device.
            workspace_device_ids = {
                d["id"] for d in devices if _area_matches(workspace_id, d.get("area_id"), self.allowed_workspaces)
            }
            workspace_entities: List[Dict[str, Any]] = []
            external_entities: List[Dict[str, Any]] = []
            for ent in entities:
                dev_id = ent.get("device_id")
                ent_area = ent.get("area_id")
                ent_match = _area_matches(workspace_id, ent_area, self.allowed_workspaces) or _entity_matches_workspace(ent.get("entity_id"), workspace_id)
                if ent_match or (dev_id in workspace_device_ids):
                    workspace_entities.append(dict(ent))
                    if dev_id:
                        workspace_device_ids.add(dev_id)
                elif _area_matches(workspace_id, next((d.get("area_id") for d in devices if d.get("id") == dev_id), None), self.allowed_workspaces):
                    external_entities.append(dict(ent))
            filtered_devices = [dict(d) for d in devices if d["id"] in workspace_device_ids]
            dev_by_id = {d["id"]: d for d in filtered_devices}

            # 2. Keep those whose own area is allowed (an entity may be moved
            #    to another area than its device's).
            kept: List[Dict[str, Any]] = []
            for ent in workspace_entities + [e for e in external_entities
                                             if e.get("device_id") in workspace_device_ids]:
                ent_area = ent.get("area_id") or (dev_by_id.get(ent.get("device_id"), {}) or {}).get("area_id")
                if _workspace_allowed(ent_area, self.allowed_workspaces):
                    kept.append(ent)

            # 3. One artifact per device.
            self.workspace_artifacts[workspace_id] = {
                entry.slug: entry for entry in _group_artifacts(kept, dev_by_id)}

    def _resolve_artifact_unlocked(self, workspace_id: str, artifact_name: str) -> ArtifactEntry:
        """The artifact a request names: by slug, device name, or the id or
        name of one of its entities -- exactly first, then ignoring case and
        punctuation. KeyError when none matches."""
        decoded_name = urllib.parse.unquote(artifact_name)
        entries = self.workspace_artifacts.get(workspace_id, {})
        # 1. The slug itself, the common case.
        if decoded_name in entries:
            return entries[decoded_name]
        # 2. Any other name, exactly; then canonically.
        for matches in (lambda alias: alias == decoded_name,
                        lambda alias: _canonical_label(alias) == _canonical_label(decoded_name)):
            for entry in entries.values():
                if any(matches(alias) for alias in _artifact_aliases(entry)):
                    return entry
        raise KeyError(decoded_name)

    def _rebuild_documents_unlocked(self) -> None:
        """Every TD HASP serves, rebuilt from the cached HA snapshot: the
        platform, the workspace list, and per workspace its artifacts, its own
        document and its artifact directory; `full_graph` is their union, for
        the graph-query endpoints."""
        # 1. Start empty.
        self._documents_dirty = False
        self._last_rebuild_monotonic = time.monotonic()
        self.platform_ttl = ""
        self.workspaces_ttl = ""
        self.workspace_ttls = {}
        self.artifacts_ttls = {}
        self.artifact_ttls = {}
        self.affordances = {}
        self.full_graph = Graph(base=self.base_uri)

        # 2. The platform and the list of its workspaces.
        areas = [self.areas_by_id[ws] for ws in sorted(self.areas_by_id)]
        platform_rdf = HomeAssistantRDF(self.base_uri)
        platform_rdf.platform_to_rdf(areas)
        self.platform_ttl = platform_rdf.serialize()
        self.full_graph.parse(data=self.platform_ttl, format="turtle")

        workspaces_rdf = HomeAssistantRDF(self.base_uri)
        for area in areas:
            workspaces_rdf.workspace_to_rdf(area, [])
        self.workspaces_ttl = workspaces_rdf.serialize()
        self.full_graph.parse(data=self.workspaces_ttl, format="turtle")

        # 3. Per workspace: its artifacts first -- the workspace document describes
        # the room variables they observe and move -- then the workspace and
        # its artifact directory.
        self.variable_readers = {}
        for workspace_id, area in self.areas_by_id.items():
            entries = list(self.workspace_artifacts.get(workspace_id, {}).values())
            artifacts_graph = Graph()
            for entry in entries:
                ttl, names = self.artifact_builder(
                    workspace_id, entry.slug, entry.slug, entry.entities,
                    self.states_by_entity_id, entry.device,
                    display_label=entry.display_name)
                self.artifact_ttls[(workspace_id, entry.slug)] = ttl
                self.affordances[(workspace_id, entry.slug)] = names
                artifacts_graph.parse(data=ttl, format="turtle")
            variables = self._variables_unlocked(workspace_id, artifacts_graph)
            self.workspace_ttls[workspace_id] = self._build_workspace_ttl(
                area, entries, artifacts_graph, variables)
            self.artifacts_ttls[workspace_id] = self._build_artifacts_ttl(workspace_id, entries)
            self.full_graph.parse(data=self.workspace_ttls[workspace_id], format="turtle")
            self.full_graph.parse(data=self.artifacts_ttls[workspace_id], format="turtle")
            self.full_graph += artifacts_graph

    def _variables_unlocked(self, workspace_id: str,
                            artifacts_graph: Graph) -> List[Tuple[Any, bool]]:
        """The workspace's room variables, read back from its artifacts' TDs:
        every variable some artifact observes or moves, with whether one
        observes it (it can be read). Records, per variable, the artifact
        properties that read it (`variable_readers`), sensors first.

        Read from the TDs rather than recomputed, so the workspace document
        can list nothing the artifacts do not mention, and miss nothing they do.
        """
        semantics = hasp_semantics.current()
        prefix = f"{self.base_uri}workspaces/{workspace_id}#"
        out: List[Tuple[Any, bool]] = []
        for node, environment in sorted(set(artifacts_graph.subject_objects(SSN.isPropertyOf))):
            # 1. A variable of this workspace: a `#name` node with an environment class.
            text = str(node)
            if not text.startswith(prefix) or not isinstance(node, URIRef):
                continue
            cls = next((semantics.environment_class(_curie(t))
                        for t in artifacts_graph.objects(node, RDF.type)
                        if semantics.environment_class(_curie(t))), None)
            if cls is None:
                continue
            env_fragment = str(environment).rsplit("#", 1)[-1]
            env_key = (hasp_semantics.ROOM_ENVIRONMENT if env_fragment == "environment"
                       else env_fragment[:-len("_environment")])
            variable = hasp_semantics.Variable(cls=cls, name=text[len(prefix):],
                                               environment=env_key)

            # 2. Who reads it: each observer's property of the variable's class.
            observers = sorted(artifacts_graph.subjects(SOSA.observes, node),
                               key=lambda a: ((a, RDF.type, hasp_semantics.SAREF.Sensor)
                                              not in artifacts_graph, str(a)))
            readers: List[Tuple[str, str]] = []
            class_uri = hasp_semantics.expand(cls.curie)
            for observer in observers:
                artifact_name = str(observer).split("/artifacts/", 1)[-1].split("#", 1)[0]
                for prop in artifacts_graph.objects(observer, TD.hasPropertyAffordance):
                    if (prop, RDF.type, class_uri) in artifacts_graph:
                        readers.append((artifact_name, str(artifacts_graph.value(prop, TD.name))))
            self.variable_readers[(workspace_id, variable.name)] = readers
            out.append((variable, bool(readers)))
        return out

    def _build_workspace_ttl(
        self,
        area: Dict[str, Any],
        entries: Optional[List[ArtifactEntry]] = None,
        artifacts_graph: Optional[Graph] = None,
        variables: Optional[List[Tuple[Any, bool]]] = None,
    ) -> str:
        """The workspace's TD: the room it is, the room's environment and its
        variables (as SimuHome's room workspace), and the artifacts it contains.

        `variables` are `_variables_unlocked`'s (variable, readable) pairs;
        `artifacts_graph`, the artifacts' TDs, gives each artifact's homeont
        class, listed here as SimuHome's room document lists it.
        """
        # 1. The workspace, its place and the room's environment.
        rdf = HomeAssistantRDF(self.base_uri)
        rdf.workspace_to_rdf(area, [])
        aid = area["area_id"]
        ws = URIRef(f"{rdf.base}workspaces/{aid}#workspace")
        hasp_semantics.add_room(rdf.g, rdf.base, area, ws)

        # 2. Every environment variable an artifact observes or moves, in full;
        #    other environments the home declares, beside the room's.
        semantics = hasp_semantics.current()
        for key, declared in semantics.environments().items():
            if any(v.environment == key for v, _ in variables or []):
                hasp_semantics.add_environment(rdf.g, rdf.base, aid, key, declared.get("label"))
        for variable, readable in variables or []:
            hasp_semantics.add_variable(rdf.g, rdf.base, aid, variable, readable,
                                        semantics.target_range(variable.name))

        # 3. The artifacts it contains, each with what kind of device it is.
        art_dir = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/")
        for entry in entries or []:
            art = self._add_contained_artifact(rdf, ws, art_dir, entry)
            for cls in (artifacts_graph.objects(art, RDF.type) if artifacts_graph is not None else []):
                if str(cls).startswith(str(hasp_semantics.HOMEONT)):
                    rdf.g.add((art, RDF.type, cls))
        return rdf.serialize()

    def _build_artifacts_ttl(self, workspace_id: str, entries: List[ArtifactEntry]) -> str:
        """The workspace's artifact directory: one `hmas:contains` per device."""
        rdf = HomeAssistantRDF(self.base_uri)
        ws = URIRef(f"{rdf.base}workspaces/{workspace_id}#workspace")
        art_dir = URIRef(f"{rdf.base}workspaces/{workspace_id}/artifacts/")
        for entry in entries:
            self._add_contained_artifact(rdf, ws, art_dir, entry)
        return rdf.serialize()

    @staticmethod
    def _add_contained_artifact(rdf: HomeAssistantRDF, ws: URIRef, art_dir: URIRef,
                                entry: ArtifactEntry) -> URIRef:
        """One artifact in a workspace listing: contained, titled by its slug,
        named by HA's display name. Returns its IRI."""
        art = URIRef(f"{art_dir}{urllib.parse.quote(entry.slug, safe='')}#artifact")
        rdf.g.add((art, RDF.type, HMAS.Artifact))
        rdf.g.add((ws, HMAS.contains, art))
        rdf.g.add((art, TD.title, Literal(entry.slug)))
        rdf.g.add((art, SCHEMA.name, Literal(entry.display_name)))
        return art
