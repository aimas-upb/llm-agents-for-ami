#!/usr/bin/env python3
"""Home Assistant utilities: async WebSocket client and Jacamo/HMAS RDF exporter."""

from __future__ import annotations
import httpx
import json
import urllib.parse
from typing import Any, Dict, List, Optional

import websockets
from rdflib import BNode, Graph, Literal, Namespace, RDF, URIRef

# Namespaces
BASE_FALLBACK = "http://localhost:8080/"
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
                    output_schema: Optional[BNode] = None):
        act = BNode()
        self.g.add((subj, TD.hasActionAffordance, act))
        self.g.add((act, RDF.type, TD.ActionAffordance))
        self.g.add((act, RDF.type, type_uri))
        self.g.add((act, TD.name, Literal(name)))
        self.g.add((act, TD.title, Literal(name)))
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

    def workspace_to_rdf(self, area: Dict[str, Any], devices: List[Dict[str, Any]]) -> None:
        aid = area["area_id"]
        ws = URIRef(f"{self.base}workspaces/{aid}#workspace")
        profile = URIRef(f"{self.base}workspaces/{aid}")
        art_dir = URIRef(f"{self.base}workspaces/{aid}/artifacts/")
        self.g.add((ws, RDF.type, TD.Thing))
        self.g.add((ws, RDF.type, HMAS.Workspace))
        self.g.add((ws, TD.title, Literal(area["name"])))
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

__all__ = ["HomeAssistantWS", "HomeAssistantRDF", "HomeAssistantREST"]
