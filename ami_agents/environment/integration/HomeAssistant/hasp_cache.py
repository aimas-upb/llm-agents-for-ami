from __future__ import annotations

import asyncio
import time
import urllib.parse
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from rdflib import Graph, Literal, Namespace, RDF, URIRef

try:
    from .hasp_utils import HomeAssistantRDF, state_property_names
    from . import hasp_semantics
    from .hasp_namespaces import HCTL, HMAS, SCHEMA, SOSA, SSN, TD, TDSOSA
except ImportError:  # pragma: no cover
    from hasp_utils import HomeAssistantRDF, state_property_names
    import hasp_semantics
    from hasp_namespaces import HCTL, HMAS, SCHEMA, SOSA, SSN, TD, TDSOSA




ArtifactBuilder = Callable[[str, str, str, List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]], str]


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


def _artifact_slug_from_entity(entity: Dict[str, Any], fallback_label: str) -> str:
    entity_id = entity.get("entity_id", "")
    if isinstance(entity_id, str) and "." in entity_id:
        object_id = entity_id.split(".", 1)[1].strip()
        if object_id:
            return object_id
    return fallback_label


def _pick_entity(device_entities: List[Dict[str, Any]], domain: str) -> Optional[str]:
    for entity in device_entities:
        ent_id = entity.get("entity_id", "")
        if ent_id.startswith(domain + "."):
            return ent_id
    return None


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

        self.workspace_devices: Dict[str, List[Dict[str, Any]]] = {}
        self.workspace_entities: Dict[str, List[Dict[str, Any]]] = {}

        self.platform_ttl: str = ""
        self.workspaces_ttl: str = ""
        self.workspace_ttls: Dict[str, str] = {}
        self.artifacts_ttls: Dict[str, str] = {}
        self.artifact_ttls: Dict[Tuple[str, str], str] = {}
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
        async with self._lock:
            self._ensure_documents_unlocked()
            _, device_entities, device, artifact_label, safe_name = self._resolve_artifact_unlocked(workspace_id, artifact_name)
            return self.artifact_ttls[(workspace_id, safe_name)], artifact_label

    async def get_service_definition(self, domain: str, service: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return ((self.services_by_domain.get(domain) or {}).get("services") or {}).get(service)

    async def resolve_artifact_domain_entity(self, workspace_id: str, artifact_name: str, domain: str) -> str:
        async with self._lock:
            _, device_entities, _, _, _ = self._resolve_artifact_unlocked(workspace_id, artifact_name)
            ent = _pick_entity(device_entities, domain)
            if not ent:
                raise KeyError(domain)
            return ent

    async def get_artifact_snapshot(
        self, workspace_id: str, artifact_name: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        async with self._lock:
            _, device_entities, _, _, _ = self._resolve_artifact_unlocked(workspace_id, artifact_name)
            state_map = {
                ent.get("entity_id"): dict(self.states_by_entity_id.get(ent.get("entity_id"), {}))
                for ent in device_entities
                if ent.get("entity_id")
            }
            return [dict(ent) for ent in device_entities], state_map

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

        `<domain>State` reads one entity's state, bare `state` the first
        entity's (deprecated alias), `metadata` the descriptive attributes,
        anything else the attribute of that name. KeyError when there is none.
        """
        _, device_entities, _, _, _ = self._resolve_artifact_unlocked(workspace_id, artifact_name)
        decoded_property_name = urllib.parse.unquote(property_name)
        # `<domain>State` names one entity's state (see
        # `state_property_names`); bare `state` is the deprecated alias
        # that reads the first entity with one.
        state_names = state_property_names(
            [e.get("entity_id") for e in device_entities if e.get("entity_id")])
        named_entity = next((entity_id for entity_id, name in state_names.items()
                             if name == decoded_property_name), None)
        if named_entity is not None:
            device_entities = [e for e in device_entities
                               if e.get("entity_id") == named_entity]
            decoded_property_name = "state"
        for entity in device_entities:
            entity_id = entity.get("entity_id")
            entity_state = self.states_by_entity_id.get(entity_id, {})
            entity_attrs = entity_state.get("attributes", {}) if isinstance(entity_state, dict) else {}

            if decoded_property_name == "metadata":
                metadata_attrs = {}
                for key, value in entity_attrs.items():
                    if key in {
                        "friendly_name", "icon", "entity_picture", "supported_features",
                        "device_class", "state_class", "unit_of_measurement", "attribution",
                        "restored", "supported_color_modes", "entity_id",
                    }:
                        metadata_attrs[key] = value
                for ts_field in ("last_changed", "last_reported", "last_updated"):
                    if ts_field in entity_state:
                        metadata_attrs[ts_field] = entity_state[ts_field]
                if metadata_attrs:
                    return metadata_attrs

            if decoded_property_name == "state":
                state_value = entity_state.get("state")
                if state_value and state_value not in ("unknown", "unavailable"):
                    entity_domain = entity_id.split(".")[0] if isinstance(entity_id, str) and "." in entity_id else ""
                    unit = entity_attrs.get("unit_of_measurement")
                    if (entity_domain == "sensor" or unit) and isinstance(state_value, str):
                        try:
                            return int(state_value) if "." not in state_value else float(state_value)
                        except ValueError:
                            return state_value
                    return state_value

            if decoded_property_name in entity_attrs and entity_attrs[decoded_property_name] is not None:
                return entity_attrs[decoded_property_name]
        raise KeyError(decoded_property_name)

    async def get_artifact_contexts(self, workspace_id: str, artifact_name: str) -> List[Dict[str, str]]:
        async with self._lock:
            _, _, _, label, safe_name = self._resolve_artifact_unlocked(workspace_id, artifact_name)
            artifact_uri = f"{self.base_uri}workspaces/{workspace_id}/artifacts/{safe_name}#artifact"
            return [
                {
                    "workspace_id": workspace_id,
                    "artifact_title": label,
                    "artifact_slug": safe_name,
                    "artifact_uri": artifact_uri,
                    "workspace_topic": f"{self.base_uri.rstrip('/')}/workspaces/{workspace_id}",
                    "artifact_topic": artifact_uri,
                }
            ]

    async def get_entity_contexts(self, entity_id: str) -> List[Dict[str, str]]:
        async with self._lock:
            contexts: List[Dict[str, str]] = []
            for workspace_id, entities in self.workspace_entities.items():
                for ent in entities:
                    if ent.get("entity_id") != entity_id:
                        continue
                    label = ent.get("_artifact_label") or ent.get("_artifact_base_label") or _entity_display_name(ent, self.devices_by_id)
                    safe_name = ent.get("_artifact_slug") or urllib.parse.quote(label, safe="")
                    artifact_uri = f"{self.base_uri}workspaces/{workspace_id}/artifacts/{safe_name}#artifact"
                    contexts.append(
                        {
                            "workspace_id": workspace_id,
                            "artifact_title": label,
                            "artifact_slug": safe_name,
                            "artifact_uri": artifact_uri,
                            "workspace_topic": f"{self.base_uri.rstrip('/')}/workspaces/{workspace_id}",
                            "artifact_topic": artifact_uri,
                        }
                    )
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
        self.workspace_devices = {}
        self.workspace_entities = {}
        devices = list(self.devices_by_id.values())
        entities = [dict(e) for e in self.entities_by_id.values()]
        for workspace_id in self.areas_by_id:
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
            label_counts: Dict[str, int] = {}
            slug_counts: Dict[str, int] = {}

            def _register(ent: Dict[str, Any]) -> None:
                label = _entity_display_name(ent, dev_by_id)
                ent["_artifact_base_label"] = label
                label_counts[label] = label_counts.get(label, 0) + 1

            filtered_entities: List[Dict[str, Any]] = []
            for ent in workspace_entities:
                ent_area = ent.get("area_id") or (dev_by_id.get(ent.get("device_id"), {}) or {}).get("area_id")
                if _workspace_allowed(ent_area, self.allowed_workspaces):
                    filtered_entities.append(ent)
                    _register(ent)
            workspace_entities = filtered_entities
            for ent in external_entities:
                if ent.get("device_id") in workspace_device_ids:
                    ent_area = ent.get("area_id") or (dev_by_id.get(ent.get("device_id"), {}) or {}).get("area_id")
                    if _workspace_allowed(ent_area, self.allowed_workspaces):
                        workspace_entities.append(ent)
                        _register(ent)
            for ent in workspace_entities:
                base_label = ent.get("_artifact_base_label", "artifact")
                label = base_label
                if label_counts.get(base_label, 0) > 1:
                    suffix = ent.get("entity_id", "")
                    object_id = suffix.split(".", 1)[1] if isinstance(suffix, str) and "." in suffix else suffix
                    label = f"{base_label} ({object_id})"
                ent["_artifact_display_label"] = label
                base_slug = _artifact_slug_from_entity(ent, label)
                slug_counts[base_slug] = slug_counts.get(base_slug, 0) + 1
                if slug_counts[base_slug] > 1:
                    entity_id = ent.get("entity_id", "")
                    if isinstance(entity_id, str) and entity_id:
                        base_slug = entity_id.replace(".", "_")
                    else:
                        base_slug = label
                ent["_artifact_slug"] = urllib.parse.quote(base_slug, safe="")
                # One instance name across listings, TDs, and focus contexts.
                # Keep HA's mutable display name as separate metadata.
                ent["_artifact_label"] = base_slug
            self.workspace_devices[workspace_id] = filtered_devices
            self.workspace_entities[workspace_id] = workspace_entities

    def _resolve_artifact_unlocked(self, workspace_id: str, artifact_name: str):
        decoded_name = urllib.parse.unquote(artifact_name)
        decoded_canon = _canonical_label(decoded_name)
        devices = self.workspace_devices.get(workspace_id, [])
        entities = self.workspace_entities.get(workspace_id, [])
        dev_by_id = {d["id"]: d for d in devices}
        for ent in entities:
            label = ent.get("_artifact_label") or _entity_display_name(ent, dev_by_id)
            object_id = ent.get("entity_id", "").split(".", 1)[-1]
            candidates = {
                label, ent.get("_artifact_base_label", ""), ent.get("_artifact_slug", ""),
                ent.get("_artifact_display_label", ""),
                _entity_display_name(ent, dev_by_id), ent.get("entity_id", ""), object_id,
            }
            matched = decoded_name in {c for c in candidates if c} or any(_canonical_label(c) == decoded_canon for c in candidates if c)
            if matched:
                device = dev_by_id.get(ent.get("device_id")) or {}
                return device, [ent], device, label, ent.get("_artifact_slug") or urllib.parse.quote(label, safe="")
        device = next((d for d in devices if d.get("name") == decoded_name), None)
        if not device:
            device = next((d for d in devices if _canonical_label(d.get("name", "")) == decoded_canon), None)
        if not device:
            raise KeyError("artifact")
        device_entities = [ent for ent in entities if ent.get("device_id") == device["id"]]
        if not device_entities:
            raise KeyError("artifact_entities")
        return device, device_entities, device, decoded_name, urllib.parse.quote(decoded_name, safe="")

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
            devices = self.workspace_devices.get(workspace_id, [])
            entities = self.workspace_entities.get(workspace_id, [])
            artifacts_graph = Graph()
            for ent in entities:
                label = ent.get("_artifact_label") or ent.get("_artifact_base_label") or _entity_display_name(ent, self.devices_by_id)
                safe_name = ent.get("_artifact_slug") or urllib.parse.quote(label, safe="")
                device = self.devices_by_id.get(ent.get("device_id"), {})
                ttl = self.artifact_builder(workspace_id, safe_name, label, [ent], self.states_by_entity_id, device)
                self.artifact_ttls[(workspace_id, safe_name)] = ttl
                artifacts_graph.parse(data=ttl, format="turtle")
            variables = self._variables_unlocked(workspace_id, artifacts_graph)
            self.workspace_ttls[workspace_id] = self._build_workspace_ttl(
                area, devices, entities, artifacts_graph, variables)
            self.artifacts_ttls[workspace_id] = self._build_artifacts_ttl(workspace_id, devices, entities)
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
        devices: List[Dict[str, Any]],
        entities: Optional[List[Dict[str, Any]]] = None,
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
        for ent in entities or []:
            label = ent.get("_artifact_label") or ent.get("_artifact_base_label") or _entity_display_name(ent, self.devices_by_id)
            safe_name = ent.get("_artifact_slug") or urllib.parse.quote(label, safe="")
            art = URIRef(f"{art_dir}{safe_name}#artifact")
            rdf.g.add((art, RDF.type, HMAS.Artifact))
            rdf.g.add((ws, HMAS.contains, art))
            rdf.g.add((art, TD.title, Literal(label)))
            rdf.g.add((art, SCHEMA.name, Literal(ent.get("_artifact_display_label") or label)))
            for cls in (artifacts_graph.objects(art, RDF.type) if artifacts_graph is not None else []):
                if str(cls).startswith(str(hasp_semantics.HOMEONT)):
                    rdf.g.add((art, RDF.type, cls))
        return rdf.serialize()

    def _build_artifacts_ttl(self, workspace_id: str, devices: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> str:
        rdf = HomeAssistantRDF(self.base_uri)
        aid = workspace_id
        ws = URIRef(f"{rdf.base}workspaces/{aid}#workspace")
        art_dir = URIRef(f"{rdf.base}workspaces/{aid}/artifacts/")
        dev_by_id = {d["id"]: d for d in devices}
        for ent in entities:
            label = ent.get("_artifact_label") or _entity_display_name(ent, dev_by_id)
            safe_name = ent.get("_artifact_slug") or urllib.parse.quote(label, safe="")
            art = URIRef(f"{art_dir}{safe_name}#artifact")
            rdf.g.add((art, RDF.type, HMAS.Artifact))
            rdf.g.add((ws, HMAS.contains, art))
            rdf.g.add((art, TD.title, Literal(label)))
            rdf.g.add((art, SCHEMA.name, Literal(ent.get("_artifact_display_label") or label)))
        return rdf.serialize()
