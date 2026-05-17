#!/usr/bin/env python3
"""
Check whether YAML-defined virtual devices are exposed by a running HASP instance.

Usage:
  python check_virtual_devices_exposed.py <yaml_file> [workspace_id]

Arguments:
  yaml_file:
      Path to the source YAML file used by Home Assistant Virtual Devices.
  workspace_id:
      Optional HASP workspace/area id. If omitted, the script requires exactly
      one value in the AREAS environment variable.

Environment:
  - HASP_URL:    Running HASP base URL, e.g. http://localhost:8080/
  - AREAS:       Comma-separated HASP workspace ids. Used when workspace_id is omitted.
  - HA_URL:      Accepted for compatibility with prepare-adapter-env.sh
  - HA_TOKEN:    Accepted for compatibility with prepare-adapter-env.sh

Behavior:
  1. Reads the YAML and collects top-level device keys
  2. Derives candidate artifact slugs from each device key and child entity names
  3. Fetches HASP's artifact directory for the target workspace
  4. Reports which YAML devices have at least one matching HASP artifact

Notes:
  - A YAML device is treated as exposed if any of its candidate artifact slugs
    is present in HASP.
  - This is intentionally tolerant because a single YAML device can create
    multiple Home Assistant entities and HASP exposes artifacts per entity.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import httpx
from rdflib import RDF, Graph, Namespace, URIRef

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to run this script") from exc


HMAS = Namespace("https://purl.org/hmas/")


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _ensure_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        _die(f"Missing required environment variable: {var}")
    return val


def _canonical_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _slugify_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def _resolve_workspace_id(arg_value: Optional[str]) -> str:
    if arg_value:
        return arg_value.strip()
    raw = os.getenv("AREAS", "")
    workspace_ids = [item.strip() for item in raw.split(",") if item.strip()]
    if len(workspace_ids) != 1:
        _die("workspace_id argument is required when AREAS does not contain exactly one value")
    return workspace_ids[0]


def _load_yaml_devices(yaml_file: Path) -> Dict[str, List[Dict[str, Any]]]:
    payload = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _die(f"YAML root must be a mapping: {yaml_file}")
    devices = payload.get("devices")
    if not isinstance(devices, dict):
        _die(f"YAML does not contain a top-level 'devices' mapping: {yaml_file}")

    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for device_key, items in devices.items():
        if not isinstance(device_key, str) or not device_key.strip():
            continue
        if not isinstance(items, list):
            items = [items]
        normalized[device_key] = [item for item in items if isinstance(item, dict)]
    return normalized


def _device_candidate_slugs(device_key: str, items: Iterable[Dict[str, Any]]) -> List[str]:
    candidates: List[str] = []
    seen: Set[str] = set()

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            candidates.append(value)

    add(device_key)
    for item in items:
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            add(_slugify_name(name))
            add(name)
    return candidates


def _artifact_slug_from_subject(subject: URIRef) -> Optional[str]:
    text = str(subject)
    if "/artifacts/" not in text or not text.endswith("#artifact"):
        return None
    encoded = text.split("/artifacts/", 1)[1][:-len("#artifact")]
    return urllib.parse.unquote(encoded)


async def _fetch_hasp_artifact_slugs(base_uri: str, workspace_id: str) -> Set[str]:
    artifacts_url = f"{base_uri.rstrip('/')}/workspaces/{workspace_id}/artifacts"
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(artifacts_url, headers={"Accept": "text/turtle"})
        response.raise_for_status()
    graph = Graph()
    graph.parse(data=response.text, format="turtle")
    slugs: Set[str] = set()
    for subject in graph.subjects(RDF.type, HMAS.Artifact):
        if not isinstance(subject, URIRef):
            continue
        slug = _artifact_slug_from_subject(subject)
        if slug:
            slugs.add(slug)
    return slugs


async def main() -> None:
    if len(sys.argv) not in {2, 3}:
        _die("Usage: python check_virtual_devices_exposed.py <yaml_file> [workspace_id]")

    yaml_file = Path(sys.argv[1]).expanduser().resolve()
    if not yaml_file.is_file():
        _die(f"YAML file not found: {yaml_file}")

    _ensure_env("HA_URL")
    _ensure_env("HA_TOKEN")
    hasp_url = os.getenv("HASP_URL", "http://localhost:8080/").strip() or "http://localhost:8080/"
    workspace_id = _resolve_workspace_id(sys.argv[2] if len(sys.argv) == 3 else None)

    yaml_devices = _load_yaml_devices(yaml_file)
    exposed_slugs = await _fetch_hasp_artifact_slugs(hasp_url, workspace_id)
    canonical_exposed = {_canonical_token(slug): slug for slug in exposed_slugs}

    exposed: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for device_key, items in sorted(yaml_devices.items()):
        candidates = _device_candidate_slugs(device_key, items)
        matched_slug = None
        for candidate in candidates:
            if candidate in exposed_slugs:
                matched_slug = candidate
                break
            canonical = _canonical_token(candidate)
            if canonical in canonical_exposed:
                matched_slug = canonical_exposed[canonical]
                break
        record = {
            "device_key": device_key,
            "candidate_slugs": candidates,
        }
        if matched_slug:
            record["matched_artifact_slug"] = matched_slug
            exposed.append(record)
        else:
            missing.append(record)

    print(
        json.dumps(
            {
                "yaml_file": str(yaml_file),
                "workspace_id": workspace_id,
                "hasp_url": hasp_url,
                "yaml_device_count": len(yaml_devices),
                "hasp_artifact_count": len(exposed_slugs),
                "all_devices_exposed": not missing,
                "exposed_count": len(exposed),
                "missing_count": len(missing),
                "exposed_devices": exposed,
                "missing_devices": missing,
                "hasp_artifact_slugs": sorted(exposed_slugs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
