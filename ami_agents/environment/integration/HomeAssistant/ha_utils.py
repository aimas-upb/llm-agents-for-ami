#!/usr/bin/env python3
"""Home Assistant utilities: async WebSocket client and Jacamo/HMAS RDF exporter."""

from __future__ import annotations
import httpx
import json
import os
import urllib.parse
from typing import Any, Dict, List, Optional

import websockets
from rdflib import BNode, Graph, Literal, Namespace, RDF, URIRef

# Namespaces
BASE_FALLBACK = os.getenv("BASE_WS_URI", "http://localhost:8080/").rstrip("/") + "/"
WEBSUB = Namespace("https://purl.org/hmas/websub/")
HCTL   = Namespace("https://www.w3.org/2019/wot/hypermedia#")
JS     = Namespace("https://www.w3.org/2019/wot/json-schema#")
HMAS   = Namespace("https://purl.org/hmas/")
EX     = Namespace("http://example.org/")
WOTSEC = Namespace("https://www.w3.org/2019/wot/security#")
HTV    = Namespace("http://www.w3.org/2011/http#")
JACAMO = Namespace("https://purl.org/hmas/jacamo/")
TD     = Namespace("https://www.w3.org/2019/wot/td#")

# ---------------- WebSocket client -----------------
class HomeAssistantWS:
    """Minimal async wrapper around Home Assistant's WebSocket API."""
    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._next_id = 0

    async def _ensure(self) -> None:
        if self._ws is not None:
            return
        self._ws = await websockets.connect(self.url)
        if json.loads(await self._ws.recv()).get("type") != "auth_required":
            raise RuntimeError("Unexpected handshake")
        await self._ws.send(json.dumps({"type": "auth", "access_token": self.token}))
        if json.loads(await self._ws.recv()).get("type") != "auth_ok":
            raise RuntimeError("Auth failed")

    async def _call(self, payload: Dict[str, Any]) -> Any:
        await self._ensure()
        self._next_id += 1
        ident = self._next_id
        await self._ws.send(json.dumps({**payload, "id": ident}))
        while True:
            msg = json.loads(await self._ws.recv())
            if msg.get("id") == ident and msg.get("type") == "result":
                if msg.get("success"):
                    return msg["result"]
                raise RuntimeError(msg.get("error"))

    async def get_areas(self) -> List[Dict[str, Any]]:
        return await self._call({"type": "config/area_registry/list"})

    async def get_devices(self, area_id: Optional[str] = None) -> List[Dict[str, Any]]:
        devices = await self._call({"type": "config/device_registry/list"})
        return [d for d in devices if area_id is None or d.get("area_id") == area_id]

    async def get_entities(self) -> List[Dict[str, Any]]:
        return await self._call({"type": "config/entity_registry/list"})

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self):
        await self._ensure()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

# ---------------- REST client -----------------
class HomeAssistantREST:
    """Minimal REST client for Home Assistant."""
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.client = httpx.AsyncClient(base_url=self.base_url, headers={
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        })

    async def get_states(self) -> List[Dict[str, Any]]:
        resp = await self.client.get("/api/states")
        resp.raise_for_status()
        return resp.json()

    async def get_services(self) -> List[Dict[str, Any]]:
        resp = await self.client.get("/api/services")
        resp.raise_for_status()
        return resp.json()

    async def call_service(self, domain: str, service: str, data: Dict[str, Any]) -> Dict[str, Any]:
        resp = await self.client.post(f"/api/services/{domain}/{service}", json=data)
        resp.raise_for_status()
        print("Calling ", domain, "/", service, " with ", data)
        # HA returns a list of changed states; normalize to dict
        try:
            return resp.json()
        except Exception:
            return {"ok": True}

    async def close(self):
        await self.client.aclose()

# ---------------- RDF builder -----------------
class HomeAssistantRDF:
    """Generate RDF/Turtle for workspaces and artifacts using HMAS/Jacamo/TD vocabularies."""
    def __init__(self, base: Optional[str] = None) -> None:
        self.base = (base or BASE_FALLBACK).rstrip("/") + "/"
        self.g = Graph(base=self.base)
        for p, ns in {
            "websub": WEBSUB, "hctl": HCTL, "js": JS, "hmas": HMAS,
            "ex": EX, "wotsec": WOTSEC, "htv": HTV, "jacamo": JACAMO, "td": TD
        }.items():
            self.g.bind(p, ns)

    def _add_action(self, subj: URIRef, name: str, type_uri: URIRef,
                    method: str, target: URIRef, ctype: str,
                    subproto: Optional[str] = None,
                    input_schema: Optional[BNode] = None,
                    output_schema: Optional[BNode] = None,
                    description: Optional[str] = None):
        act = BNode()
        self.g.add((subj, TD.hasActionAffordance, act))
        self.g.add((act, RDF.type, TD.ActionAffordance))
        self.g.add((act, RDF.type, type_uri))
        self.g.add((act, TD.name, Literal(name)))
        self.g.add((act, TD.title, Literal(name)))
        if description:
            # Add RDFS namespace if not already bound
            RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
            self.g.bind("rdfs", RDFS)
            self.g.add((act, RDFS.comment, Literal(description)))
        form = BNode()
        self.g.add((act, TD.hasForm, form))
        self.g.add((form, HTV.methodName, Literal(method)))
        self.g.add((form, HCTL.hasTarget, target))
        self.g.add((form, HCTL.forContentType, Literal(ctype)))
        self.g.add((form, HCTL.hasOperationType, TD.invokeAction))
        if subproto:
            self.g.add((form, HCTL.forSubProtocol, Literal(subproto)))
        if input_schema:
            self.g.add((act, TD.hasInputSchema, input_schema))
        if output_schema:
            self.g.add((act, TD.hasOutputSchema, output_schema))

    # ---- Schema helpers to mirror your sample ----
    def _schema_set_color(self) -> BNode:
        obj = BNode()
        self.g.add((obj, RDF.type, JS.ObjectSchema))
        prop = BNode()
        self.g.add((obj, JS.properties, prop))
        self.g.add((prop, RDF.type, JS.StringSchema))
        self.g.add((prop, JS.propertyName, Literal("color")))
        for c in ("red", "green", "blue"):
            self.g.add((prop, JS.enum, Literal(c)))
        self.g.add((obj, JS.required, Literal("color")))
        return obj

    def _schema_set_intensity(self) -> BNode:
        obj = BNode()
        self.g.add((obj, RDF.type, JS.ObjectSchema))
        prop = BNode()
        self.g.add((obj, JS.properties, prop))
        self.g.add((prop, RDF.type, JS.IntegerSchema))
        self.g.add((prop, JS.propertyName, Literal("lightIntensity")))
        for v in ("100", "25", "50", "75"):
            self.g.add((prop, JS.enum, Literal(v)))
        self.g.add((obj, JS.required, Literal("lightIntensity")))
        return obj

    def _schema_status_output(self) -> BNode:
        obj = BNode()
        self.g.add((obj, RDF.type, JS.ObjectSchema))
        # lightIntensity
        prop1 = BNode()
        self.g.add((obj, JS.properties, prop1))
        self.g.add((prop1, RDF.type, JS.IntegerSchema))
        self.g.add((prop1, JS.propertyName, Literal("lightIntensity")))
        for v in ("100", "25", "50", "75"):
            self.g.add((prop1, JS.enum, Literal(v)))
        # color
        prop2 = BNode()
        self.g.add((obj, JS.properties, prop2))
        self.g.add((prop2, RDF.type, JS.StringSchema))
        self.g.add((prop2, JS.propertyName, Literal("color")))
        for c in ("red", "green", "blue"):
            self.g.add((prop2, JS.enum, Literal(c)))
        # state
        prop3 = BNode()
        self.g.add((obj, JS.properties, prop3))
        self.g.add((prop3, RDF.type, JS.StringSchema))
        self.g.add((prop3, JS.propertyName, Literal("state")))
        for s in ("off", "on"):
            self.g.add((prop3, JS.enum, Literal(s)))
        # required
        for r in ("state", "color", "lightIntensity"):
            self.g.add((obj, JS.required, Literal(r)))
        return obj

    # ---- Dynamic schema builders for HA services ----
    def _build_input_schema_from_fields(self, fields: Dict[str, Any]) -> Optional[BNode]:
        """Build JSON Schema from HA service field definitions."""
        if not fields:
            return None

        # Filter out entity_id as it's implicit
        filtered_fields = {k: v for k, v in fields.items() if k != "entity_id"}
        if not filtered_fields:
            return None

        obj = BNode()
        self.g.add((obj, RDF.type, JS.ObjectSchema))

        for field_name, field_def in filtered_fields.items():
            prop = BNode()
            self.g.add((obj, JS.properties, prop))
            self.g.add((prop, JS.propertyName, Literal(field_name)))

            # Map HA field types to JSON Schema types
            selector = field_def.get("selector", {})
            if "number" in selector:
                self.g.add((prop, RDF.type, JS.NumberSchema))
                num_def = selector["number"]
                if "min" in num_def:
                    self.g.add((prop, JS.minimum, Literal(num_def["min"])))
                if "max" in num_def:
                    self.g.add((prop, JS.maximum, Literal(num_def["max"])))
                if "step" in num_def:
                    self.g.add((prop, JS.multipleOf, Literal(num_def["step"])))
            elif "text" in selector:
                self.g.add((prop, RDF.type, JS.StringSchema))
            elif "select" in selector:
                self.g.add((prop, RDF.type, JS.StringSchema))
                options = selector["select"].get("options", [])
                for opt in options:
                    # Handle both string options and dict options
                    opt_value = opt if isinstance(opt, str) else opt.get("value", opt.get("label"))
                    if opt_value:
                        self.g.add((prop, JS.enum, Literal(opt_value)))
            elif "boolean" in selector:
                self.g.add((prop, RDF.type, JS.BooleanSchema))
            elif "color_rgb" in selector:
                self.g.add((prop, RDF.type, JS.ArraySchema))
                # RGB is [r, g, b] where each is 0-255
                item = BNode()
                self.g.add((prop, JS.items, item))
                self.g.add((item, RDF.type, JS.IntegerSchema))
                self.g.add((item, JS.minimum, Literal(0)))
                self.g.add((item, JS.maximum, Literal(255)))
                self.g.add((prop, JS.minItems, Literal(3)))
                self.g.add((prop, JS.maxItems, Literal(3)))
            elif "color_temp" in selector:
                self.g.add((prop, RDF.type, JS.NumberSchema))
                ct_def = selector["color_temp"]
                if "min_mireds" in ct_def:
                    self.g.add((prop, JS.minimum, Literal(ct_def["min_mireds"])))
                if "max_mireds" in ct_def:
                    self.g.add((prop, JS.maximum, Literal(ct_def["max_mireds"])))
            else:
                # Default to string for unknown types
                self.g.add((prop, RDF.type, JS.StringSchema))

            # Add description if available
            if "description" in field_def:
                self.g.add((prop, JS.description, Literal(field_def["description"])))

            # Mark as required if specified
            if field_def.get("required"):
                self.g.add((obj, JS.required, Literal(field_name)))

        return obj

    def _build_sensor_output_schema(self, device_class: Optional[str], unit: Optional[str]) -> BNode:
        """Build output schema for sensor readings."""
        obj = BNode()
        self.g.add((obj, RDF.type, JS.NumberSchema))

        # Add unit as description if available
        if unit:
            self.g.add((obj, JS.description, Literal(f"Sensor value in {unit}")))

        return obj

    def _build_binary_sensor_output_schema(self) -> BNode:
        """Build output schema for binary sensor state."""
        obj = BNode()
        self.g.add((obj, RDF.type, JS.StringSchema))
        self.g.add((obj, JS.enum, Literal("on")))
        self.g.add((obj, JS.enum, Literal("off")))

        return obj

    def _build_climate_output_schema(self) -> BNode:
        """Build output schema for climate/thermostat state."""
        obj = BNode()
        self.g.add((obj, RDF.type, JS.ObjectSchema))

        # state (hvac mode)
        prop_state = BNode()
        self.g.add((obj, JS.properties, prop_state))
        self.g.add((prop_state, JS.propertyName, Literal("state")))
        self.g.add((prop_state, RDF.type, JS.StringSchema))

        # currentTemperature
        prop_current = BNode()
        self.g.add((obj, JS.properties, prop_current))
        self.g.add((prop_current, JS.propertyName, Literal("currentTemperature")))
        self.g.add((prop_current, RDF.type, JS.NumberSchema))

        # targetTemperature
        prop_target = BNode()
        self.g.add((obj, JS.properties, prop_target))
        self.g.add((prop_target, JS.propertyName, Literal("targetTemperature")))
        self.g.add((prop_target, RDF.type, JS.NumberSchema))

        # hvacAction
        prop_action = BNode()
        self.g.add((obj, JS.properties, prop_action))
        self.g.add((prop_action, JS.propertyName, Literal("hvacAction")))
        self.g.add((prop_action, RDF.type, JS.StringSchema))

        return obj

    def _add_property(self, subj: URIRef, name: str, property_uri: URIRef,
                      output_schema: Optional[BNode] = None,
                      description: Optional[str] = None,
                      observable: bool = True):
        """Add a PropertyAffordance to a Thing."""
        prop = BNode()
        self.g.add((subj, TD.hasPropertyAffordance, prop))
        self.g.add((prop, RDF.type, TD.PropertyAffordance))
        self.g.add((prop, TD.name, Literal(name)))
        self.g.add((prop, TD.title, Literal(name)))

        if description:
            RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
            self.g.bind("rdfs", RDFS)
            self.g.add((prop, RDFS.comment, Literal(description)))

        # Add observable flag
        self.g.add((prop, TD.isObservable, Literal(observable)))

        # Add output schema
        if output_schema:
            self.g.add((prop, TD.hasOutputSchema, output_schema))

        # Add a form for reading the property (GET)
        form = BNode()
        self.g.add((prop, TD.hasForm, form))
        self.g.add((form, HTV.methodName, Literal("GET")))
        self.g.add((form, HCTL.hasTarget, property_uri))
        self.g.add((form, HCTL.forContentType, Literal("application/json")))
        self.g.add((form, HCTL.hasOperationType, TD.readProperty))

        return prop

    def _build_metadata_schema(self, metadata_attrs: Dict[str, Any]) -> Optional[BNode]:
        """Build an object schema for metadata attributes."""
        if not metadata_attrs:
            return None

        obj = BNode()
        self.g.add((obj, RDF.type, JS.ObjectSchema))

        for attr_name, attr_value in metadata_attrs.items():
            prop = BNode()
            self.g.add((obj, JS.properties, prop))
            self.g.add((prop, JS.propertyName, Literal(attr_name)))

            # Determine type from value
            if isinstance(attr_value, bool):
                self.g.add((prop, RDF.type, JS.BooleanSchema))
            elif isinstance(attr_value, int):
                self.g.add((prop, RDF.type, JS.IntegerSchema))
            elif isinstance(attr_value, float):
                self.g.add((prop, RDF.type, JS.NumberSchema))
            elif isinstance(attr_value, str):
                self.g.add((prop, RDF.type, JS.StringSchema))
            elif isinstance(attr_value, (list, tuple)):
                self.g.add((prop, RDF.type, JS.ArraySchema))
            else:
                # Default to string for unknown types
                self.g.add((prop, RDF.type, JS.StringSchema))

        return obj

    def _build_property_schema(self, attr_name: str, attr_value: Any,
                                entity_attributes: Dict[str, Any],
                                entity_domain: Optional[str] = None) -> Optional[BNode]:
        """Build a JSON Schema for a property based on its value and context.

        Args:
            attr_name: The attribute name (e.g., "state", "brightness")
            attr_value: The current value of the attribute
            entity_attributes: All entity attributes for context
            entity_domain: Optional entity domain (e.g., "sensor", "light") for domain-specific logic
        """
        schema = BNode()

        # Determine type from value
        if isinstance(attr_value, bool):
            self.g.add((schema, RDF.type, JS.BooleanSchema))
        elif isinstance(attr_value, int):
            self.g.add((schema, RDF.type, JS.IntegerSchema))
            # Add min/max if available for known attributes
            if attr_name == "brightness" and "brightness" in entity_attributes:
                self.g.add((schema, JS.minimum, Literal(0)))
                self.g.add((schema, JS.maximum, Literal(255)))
            elif attr_name == "color_temp":
                min_mireds = entity_attributes.get("min_mireds")
                max_mireds = entity_attributes.get("max_mireds")
                if min_mireds is not None:
                    self.g.add((schema, JS.minimum, Literal(min_mireds)))
                if max_mireds is not None:
                    self.g.add((schema, JS.maximum, Literal(max_mireds)))
            elif attr_name == "temperature":
                min_temp = entity_attributes.get("min_temp")
                max_temp = entity_attributes.get("max_temp")
                if min_temp is not None:
                    self.g.add((schema, JS.minimum, Literal(min_temp)))
                if max_temp is not None:
                    self.g.add((schema, JS.maximum, Literal(max_temp)))
            # Add unit description for numeric state values
            if attr_name == "state":
                unit = entity_attributes.get("unit_of_measurement")
                if unit:
                    self.g.add((schema, JS.description, Literal(f"Sensor value in {unit}")))
        elif isinstance(attr_value, float):
            self.g.add((schema, RDF.type, JS.NumberSchema))
            # Add constraints for temperature-related attributes
            if attr_name in ("temperature", "current_temperature", "target_temp_low", "target_temp_high"):
                min_temp = entity_attributes.get("min_temp")
                max_temp = entity_attributes.get("max_temp")
                if min_temp is not None:
                    self.g.add((schema, JS.minimum, Literal(min_temp)))
                if max_temp is not None:
                    self.g.add((schema, JS.maximum, Literal(max_temp)))
            # Add unit description for numeric state values
            if attr_name == "state":
                unit = entity_attributes.get("unit_of_measurement")
                if unit:
                    self.g.add((schema, JS.description, Literal(f"Sensor value in {unit}")))
        elif isinstance(attr_value, str):
            self.g.add((schema, RDF.type, JS.StringSchema))
            # Add enums for known constrained string attributes
            if attr_name == "hvac_mode":
                modes = entity_attributes.get("hvac_modes", [])
                for mode in modes:
                    self.g.add((schema, JS.enum, Literal(mode)))
            elif attr_name == "preset_mode":
                modes = entity_attributes.get("preset_modes", [])
                for mode in modes:
                    self.g.add((schema, JS.enum, Literal(mode)))
            elif attr_name == "fan_mode":
                modes = entity_attributes.get("fan_modes", [])
                for mode in modes:
                    self.g.add((schema, JS.enum, Literal(mode)))
            elif attr_name == "swing_mode":
                modes = entity_attributes.get("swing_modes", [])
                for mode in modes:
                    self.g.add((schema, JS.enum, Literal(mode)))
            elif attr_name == "color_mode":
                modes = entity_attributes.get("supported_color_modes", [])
                for mode in modes:
                    self.g.add((schema, JS.enum, Literal(mode)))
            elif attr_name == "state":
                # Only add binary enum for states that are actually binary
                # Skip binary enum for sensor domain or entities with unit_of_measurement
                unit = entity_attributes.get("unit_of_measurement")
                if not (entity_domain == "sensor" or unit):
                    # This is likely a binary state (switch, light, etc.)
                    self.g.add((schema, JS.enum, Literal("on")))
                    self.g.add((schema, JS.enum, Literal("off")))
        elif isinstance(attr_value, (list, tuple)):
            self.g.add((schema, RDF.type, JS.ArraySchema))
            # For RGB colors
            if attr_name in ("rgb_color", "rgbw_color", "rgbww_color"):
                item = BNode()
                self.g.add((schema, JS.items, item))
                self.g.add((item, RDF.type, JS.IntegerSchema))
                self.g.add((item, JS.minimum, Literal(0)))
                self.g.add((item, JS.maximum, Literal(255)))
                if attr_name == "rgb_color":
                    self.g.add((schema, JS.minItems, Literal(3)))
                    self.g.add((schema, JS.maxItems, Literal(3)))
                elif attr_name == "rgbw_color":
                    self.g.add((schema, JS.minItems, Literal(4)))
                    self.g.add((schema, JS.maxItems, Literal(4)))
                elif attr_name == "rgbww_color":
                    self.g.add((schema, JS.minItems, Literal(5)))
                    self.g.add((schema, JS.maxItems, Literal(5)))
            elif attr_name in ("hs_color", "xy_color"):
                item = BNode()
                self.g.add((schema, JS.items, item))
                self.g.add((item, RDF.type, JS.NumberSchema))
                self.g.add((schema, JS.minItems, Literal(2)))
                self.g.add((schema, JS.maxItems, Literal(2)))
        else:
            # For other types, default to string
            self.g.add((schema, RDF.type, JS.StringSchema))

        return schema

    def platform_to_rdf(self, areas: List[Dict[str, Any]]) -> None:
        """
        Add HypermediaMASPlatform representation at the base URI.
        The platform hosts all workspaces.

        Args:
            areas: List of Home Assistant areas (which map to workspaces)
        """
        platform_uri = URIRef(f"{self.base}#platform")
        profile_uri = URIRef(self.base)

        # Platform instance
        self.g.add((platform_uri, RDF.type, HMAS.HypermediaMASPlatform))
        self.g.add((platform_uri, RDF.type, TD.Thing))
        self.g.add((platform_uri, TD.title, Literal("Home Assistant Platform")))

        # Security configuration
        sec = BNode()
        self.g.add((platform_uri, TD.hasSecurityConfiguration, sec))
        self.g.add((sec, RDF.type, WOTSEC.NoSecurityScheme))

        # Add hmas:hosts predicates for each workspace
        for area in areas:
            aid = area["area_id"]
            ws_uri = URIRef(f"{self.base}workspaces/{aid}#workspace")
            self.g.add((platform_uri, HMAS.hosts, ws_uri))

        # Platform profile
        self.g.add((profile_uri, RDF.type, HMAS.ResourceProfile))
        self.g.add((profile_uri, HMAS.isProfileOf, platform_uri))

    def workspace_to_rdf(self, area: Dict[str, Any], devices: List[Dict[str, Any]]) -> None:
        aid = area["area_id"]
        ws = URIRef(f"{self.base}workspaces/{aid}#workspace")
        profile = URIRef(f"{self.base}workspaces/{aid}")
        art_dir = URIRef(f"{self.base}workspaces/{aid}/artifacts/")
        platform_uri = URIRef(f"{self.base}#platform")

        self.g.add((ws, RDF.type, TD.Thing))
        self.g.add((ws, RDF.type, HMAS.Workspace))
        self.g.add((ws, TD.title, Literal(area["name"])))

        # Add backwards pointing property to platform
        self.g.add((ws, HMAS.isHostedOn, platform_uri))
        sec = BNode()
        self.g.add((ws, TD.hasSecurityConfiguration, sec))
        self.g.add((sec, RDF.type, WOTSEC.NoSecurityScheme))
        # workspace actions (subset)
        self._add_action(ws, "createArtifact", JACAMO.createArtifact, "POST", art_dir, "text/turtle")
        self._add_action(ws, "joinWorkspace", JACAMO.JoinWorkspace, "POST",
                         URIRef(f"{self.base}workspaces/{aid}/join"), "application/json")
        self._add_action(ws, "quitWorkspace", JACAMO.QuitWorkspace, "POST",
                         URIRef(f"{self.base}workspaces/{aid}/leave"), "application/json")
        self._add_action(ws, "subscribeToWorkspace", WEBSUB.subscribeToWorkspace, "POST",
                         URIRef(f"{self.base}hub/"), "application/json", "websub")
        # contained artifacts from devices
        for d in devices:
            name = d.get("name", d.get("id"))
            safe_name = urllib.parse.quote(name, safe="")
            art = URIRef(f"{art_dir}{safe_name}#artifact")
            self.g.add((art, RDF.type, HMAS.Artifact))
            self.g.add((ws, HMAS.contains, art))
        # profile
        self.g.add((profile, RDF.type, HMAS.ResourceProfile))
        self.g.add((profile, HMAS.isProfileOf, ws))

    def serialize(self) -> str:
        out = self.g.serialize(format="turtle")
        return out.decode() if isinstance(out, bytes) else out

# Common metadata/admin attributes to filter out
METADATA_ATTRIBUTES = {
    "friendly_name",
    "icon",
    "entity_picture",
    "supported_features",
    "device_class",
    "state_class",
    "unit_of_measurement",
    "attribution",
    "restored",
    "supported_color_modes",
    "entity_id",
    "last_changed",
    "last_updated",
    "context",
}

# Known operational attributes by domain
OPERATIONAL_ATTRIBUTES = {
    "light": {
        "brightness",
        "color_temp",
        "rgb_color",
        "rgbw_color",
        "rgbww_color",
        "xy_color",
        "hs_color",
        "color_mode",
        "effect",
        "transition",
    },
    "climate": {
        "temperature",
        "target_temp_high",
        "target_temp_low",
        "current_temperature",
        "hvac_mode",
        "hvac_action",
        "preset_mode",
        "fan_mode",
        "swing_mode",
    },
    "media_player": {
        "volume_level",
        "is_volume_muted",
        "media_content_id",
        "media_content_type",
        "media_duration",
        "media_position",
        "source",
        "sound_mode",
    },
    "cover": {
        "current_position",
        "current_tilt_position",
    },
    "fan": {
        "percentage",
        "preset_mode",
        "oscillating",
        "direction",
    },
}

def get_writable_fields_from_services(domain_services: Dict[str, Any]) -> set:
    """
    Extract all writable field names from a domain's service definitions.

    Args:
        domain_services: The services dict for a domain (e.g., from services API)

    Returns:
        Set of field names that can be written via services
    """
    writable = set()
    for service_name, service_def in domain_services.items():
        fields = service_def.get("fields", {})
        for field_name in fields.keys():
            if field_name != "entity_id":  # entity_id is not an attribute
                writable.add(field_name)
    return writable


def get_metadata_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata attributes from entity attributes.

    Args:
        attributes: The full attributes dict from the entity state

    Returns:
        Dict containing only metadata attributes
    """
    if not attributes:
        return {}

    metadata = {}

    for key, value in attributes.items():
        # Include if it's a known metadata attribute
        if key in METADATA_ATTRIBUTES:
            metadata[key] = value
            continue

        # Include attributes ending in metadata suffixes
        if key.endswith(("_name", "_id", "_list", "_modes", "_features")):
            metadata[key] = value

    return metadata


def get_operational_attributes(domain: str, attributes: Dict[str, Any],
                               service_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Filter entity attributes to return only operational (non-metadata) attributes.

    Args:
        domain: The entity domain (e.g., "light", "climate")
        attributes: The full attributes dict from the entity state
        service_fields: Optional dict of service fields from HA service definitions
                       Can be a single service's fields or use get_writable_fields_from_services()

    Returns:
        Dict containing only operational attributes
    """
    if not attributes:
        return {}

    operational = {}

    # Strategy 1: Use known operational attributes for the domain
    known_ops = OPERATIONAL_ATTRIBUTES.get(domain, set())

    # Strategy 2: Check against service fields if available
    service_field_names = set(service_fields.keys()) if service_fields else set()

    for key, value in attributes.items():
        # Skip known metadata
        if key in METADATA_ATTRIBUTES:
            continue

        # Include if it's a known operational attribute for this domain
        if key in known_ops:
            operational[key] = value
            continue

        # Include if it matches a service field (writable)
        if service_field_names and key in service_field_names:
            operational[key] = value
            continue

        # Additional heuristics:
        # - Skip attributes ending in common metadata suffixes
        if key.endswith(("_name", "_id", "_list", "_modes", "_features")):
            continue

        # - Include attributes with numeric/boolean values (likely operational)
        if isinstance(value, (int, float, bool)):
            operational[key] = value

    return operational


def get_supported_service_fields(domain: str, entity_attributes: Dict[str, Any],
                                 service_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter service fields to only those supported by the specific entity.

    Args:
        domain: Entity domain (e.g., "light", "climate")
        entity_attributes: The entity's current attributes
        service_fields: All available service fields

    Returns:
        Dict of service fields that are actually supported by this entity
    """
    if not entity_attributes or not service_fields:
        return service_fields

    supported = {}

    if domain == "light":
        # Get supported color modes
        supported_color_modes = set(entity_attributes.get("supported_color_modes", []))

        for field_name, field_def in service_fields.items():
            # Always include basic fields
            if field_name in ("transition", "flash", "effect"):
                supported[field_name] = field_def
                continue

            # Brightness fields
            if field_name in ("brightness", "brightness_pct", "brightness_step", "brightness_step_pct"):
                # Include if device has brightness attribute or supports brightness/onoff modes
                if "brightness" in entity_attributes or "brightness" in supported_color_modes or "onoff" in supported_color_modes:
                    supported[field_name] = field_def
                continue

            # Color fields - check against supported_color_modes
            # rgb_color is supported if device supports any of: hs, xy, rgb, rgbw, rgbww
            if field_name in ("rgb_color", "rgbw_color", "rgbww_color"):
                compatible_modes = {"hs", "xy", "rgb", "rgbw", "rgbww"}
                if supported_color_modes & compatible_modes:  # intersection check
                    supported[field_name] = field_def
                continue

            if field_name in ("hs_color", "xy_color"):
                mode = field_name.replace("_color", "")  # "hs" or "xy"
                if mode in supported_color_modes or field_name in supported_color_modes:
                    supported[field_name] = field_def
                continue

            # Color temp fields
            if field_name in ("color_temp", "color_temp_kelvin", "kelvin"):
                if "color_temp" in supported_color_modes:
                    supported[field_name] = field_def
                continue

            # White value
            if field_name == "white":
                if "white" in supported_color_modes:
                    supported[field_name] = field_def
                continue

            # Default: include if not color-related
            supported[field_name] = field_def

    elif domain == "climate":
        # Check supported features
        hvac_modes = entity_attributes.get("hvac_modes", [])
        preset_modes = entity_attributes.get("preset_modes", [])
        fan_modes = entity_attributes.get("fan_modes", [])
        swing_modes = entity_attributes.get("swing_modes", [])

        for field_name, field_def in service_fields.items():
            # Temperature fields - check if entity supports temperature setting
            if field_name in ("temperature", "target_temp_high", "target_temp_low"):
                if "min_temp" in entity_attributes or "max_temp" in entity_attributes:
                    supported[field_name] = field_def
                continue

            # HVAC mode
            if field_name == "hvac_mode":
                if hvac_modes:
                    supported[field_name] = field_def
                continue

            # Preset mode
            if field_name == "preset_mode":
                if preset_modes:
                    supported[field_name] = field_def
                continue

            # Fan mode
            if field_name == "fan_mode":
                if fan_modes:
                    supported[field_name] = field_def
                continue

            # Swing mode
            if field_name == "swing_mode":
                if swing_modes:
                    supported[field_name] = field_def
                continue

            # Include by default
            supported[field_name] = field_def

    elif domain == "cover":
        # Check supported features via supported_features bitmask
        supported_features = entity_attributes.get("supported_features", 0)

        for field_name, field_def in service_fields.items():
            # Position (bit 4 = SUPPORT_SET_POSITION)
            if field_name == "position":
                if supported_features & 4:  # 0b0100
                    supported[field_name] = field_def
                continue

            # Tilt (bit 128 = SUPPORT_SET_TILT_POSITION)
            if field_name == "tilt_position":
                if supported_features & 128:
                    supported[field_name] = field_def
                continue

            # Default: include
            supported[field_name] = field_def

    else:
        # For other domains, return all fields
        return service_fields

    return supported

__all__ = ["HomeAssistantWS", "HomeAssistantRDF", "HomeAssistantREST",
           "get_operational_attributes", "get_metadata_attributes",
           "get_writable_fields_from_services", "get_supported_service_fields"]