from __future__ import annotations

import asyncio
import urllib.parse
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from rdflib import Graph, Literal, Namespace, RDF, URIRef

try:
    from .hasp_utils import HomeAssistantRDF
except ImportError:  # pragma: no cover
    from hasp_utils import HomeAssistantRDF


HMAS = Namespace("https://purl.org/hmas/")
TD = Namespace("https://www.w3.org/2019/wot/td#")
HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
TDSOSA = Namespace("https://example.org/hmas/td-sosa-ext#")


ArtifactBuilder = Callable[[str, str, str, List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]], str]


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
            return val.strip()
    object_id = ""
    ent_id = entity.get("entity_id", "")
    if isinstance(ent_id, str) and "." in ent_id:
        object_id = ent_id.split(".", 1)[1]
    device = devices_by_id.get(entity.get("device_id"))
    device_name = (device or {}).get("name") if isinstance(device, dict) else None
    if device_name and object_id:
        return f"{device_name}.{object_id}"
    if device_name:
        return device_name
    return object_id or ent_id or "artifact"


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
        self.full_graph: Graph = Graph(base=self.base_uri)

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
            self._rebuild_documents_unlocked()

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
                self._rebuild_documents_unlocked()
                return
        await self.refresh_states()

    async def has_workspace(self, workspace_id: str) -> bool:
        async with self._lock:
            return workspace_id in self.areas_by_id

    async def get_platform_ttl(self) -> str:
        async with self._lock:
            return self.platform_ttl

    async def serialize_full_graph(self, fmt: str = "turtle") -> str:
        async with self._lock:
            return self.full_graph.serialize(format=fmt)

    async def query_actions_affecting_observable_property(
        self, workspace_id: str, observable_property: str
    ) -> List[Dict[str, str]]:
        property_uri = (
            observable_property
            if observable_property.startswith(("http://", "https://"))
            else f"{self.base_uri}workspaces/{workspace_id}/environment/{urllib.parse.quote(observable_property, safe='')}"
        )
        artifact_prefix = f"{self.base_uri}workspaces/{workspace_id}/artifacts/"
        query = f"""
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
        async with self._lock:
            rows = list(self.full_graph.query(query))

        results: List[Dict[str, str]] = []
        for row in rows:
            direction = None
            if row.dirType == TDSOSA.IncreasingActuation:
                direction = "increase"
            elif row.dirType == TDSOSA.DecreasingActuation:
                direction = "decrease"
            results.append(
                {
                    "artifact_uri": str(row.artifact),
                    "artifact_title": str(row.artifactTitle) if row.artifactTitle is not None else "",
                    "action_name": str(row.actionName),
                    "action_target": str(row.actionTarget),
                    "actuation_uri": str(row.actuation),
                    "property_uri": property_uri,
                    "direction": direction or "",
                }
            )
        return results

    async def get_workspaces_ttl(self) -> str:
        async with self._lock:
            return self.workspaces_ttl

    async def get_workspace_ttl(self, workspace_id: str) -> str:
        async with self._lock:
            return self.workspace_ttls[workspace_id]

    async def get_artifacts_ttl(self, workspace_id: str) -> str:
        async with self._lock:
            return self.artifacts_ttls[workspace_id]

    async def get_artifact_ttl(self, workspace_id: str, artifact_name: str) -> Tuple[str, str]:
        async with self._lock:
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
        async with self._lock:
            _, device_entities, _, _, _ = self._resolve_artifact_unlocked(workspace_id, artifact_name)
            decoded_property_name = urllib.parse.unquote(property_name)
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
                ent["_artifact_label"] = label
                ent["_artifact_slug"] = urllib.parse.quote(label, safe="")
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
        self.platform_ttl = ""
        self.workspaces_ttl = ""
        self.workspace_ttls = {}
        self.artifacts_ttls = {}
        self.artifact_ttls = {}
        self.full_graph = Graph(base=self.base_uri)

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

        for workspace_id, area in self.areas_by_id.items():
            devices = self.workspace_devices.get(workspace_id, [])
            entities = self.workspace_entities.get(workspace_id, [])
            self.workspace_ttls[workspace_id] = self._build_workspace_ttl(area, devices)
            self.artifacts_ttls[workspace_id] = self._build_artifacts_ttl(workspace_id, devices, entities)
            self.full_graph.parse(data=self.workspace_ttls[workspace_id], format="turtle")
            self.full_graph.parse(data=self.artifacts_ttls[workspace_id], format="turtle")
            for ent in entities:
                label = ent.get("_artifact_label") or ent.get("_artifact_base_label") or _entity_display_name(ent, self.devices_by_id)
                safe_name = ent.get("_artifact_slug") or urllib.parse.quote(label, safe="")
                device = self.devices_by_id.get(ent.get("device_id"), {})
                ttl = self.artifact_builder(workspace_id, safe_name, label, [ent], self.states_by_entity_id, device)
                self.artifact_ttls[(workspace_id, safe_name)] = ttl
                self.full_graph.parse(data=ttl, format="turtle")

    def _build_workspace_ttl(self, area: Dict[str, Any], devices: List[Dict[str, Any]]) -> str:
        rdf = HomeAssistantRDF(self.base_uri)
        rdf.workspace_to_rdf(area, devices)
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
        return rdf.serialize()
