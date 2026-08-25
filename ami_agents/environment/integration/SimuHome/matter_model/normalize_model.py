#!/usr/bin/env python3
"""
Normalize vendored CSA Matter data-model XML into compact JSON lookup tables.

Usage:
  python normalize_model.py <vendored_dir> [--source URL] [--tag TAG]
                            [--commit SHA] [--spec-version VER]

<vendored_dir> is matter_model/vendor/<VER>/, containing the clusters/ and
device_types/ subdirectories copied out of connectedhomeip's data_model/<VER>/.

Emits, next to those subdirectories:
  clusters.json      decimal clusterId    -> cluster record
  device-types.json  decimal deviceTypeId -> device type record
  manifest.json      provenance + sha256 of every vendored/generated file

Stdlib only (xml.etree.ElementTree, glob, json, hashlib).

XML shapes this parser relies on (verified against the vendored tree, not assumed):
  - <cluster id="0x0202" name="Fan Control Cluster" revision="6">
    Some files omit the root id and instead carry <clusterIds> with several
    concrete <clusterId id="0x0071" name="HEPA Filter Monitoring"/> entries;
    one file can therefore define many clusters. Files with neither a root id
    nor any identified clusterId are abstract bases (AlarmBase, ModeBase,
    Label-Cluster) and are skipped.
  - <attribute id="0x0002" name="PercentSetting" type="percent"> with an
    optional <quality .../> child.
  - <command id="0x00" name="Step" direction="commandToServer">
  - <deviceType id="0x002D" name="Air Purifier" revision="2"> whose <clusters>
    children carry conformance as a child element (<mandatoryConform/> vs
    <optionalConform/>), not as an attribute.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

# Files whose root <cluster> carries no id and whose <clusterIds> entries are
# likewise unidentified: abstract base definitions, not addressable clusters.
_ABSTRACT_SKIPPED: List[str] = []


def _parse_id(raw: Optional[str]) -> Optional[int]:
    """Parse a Matter id, which appears as 0x-hex in the XML but is keyed on
    decimal everywhere it is used at runtime (ServerList, attribute paths)."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(text, 10)
    except ValueError:
        return None


def _hex4(value: int) -> str:
    return f"0x{value:04X}"


def _hex2(value: int) -> str:
    return f"0x{value:02X}"


def _quality(node: ET.Element) -> Dict[str, str]:
    quality = node.find("quality")
    if quality is None:
        return {}
    return {k: v for k, v in quality.attrib.items()}


def _access(node: ET.Element) -> Optional[Dict[str, str]]:
    """Read/write permissions from an attribute's <access> child, if present."""
    access = node.find("access")
    if access is None:
        return None
    return {k: v for k, v in access.attrib.items()} or None


def _writable(access: Optional[Dict[str, str]], *, has_access: bool) -> Optional[bool]:
    """Tri-state writability.

    True  — access declares write="true" (126 attributes) or write="optional" (13).
    False — an <access> element exists but declares no write permission (767).
    None  — no <access> element at all (52 attributes); the spec simply does not
            say, and callers must not read that as read-only.
    """
    if not has_access or access is None:
        return None
    write = access.get("write")
    if write in {"true", "optional"}:
        return True
    return False


def _conformance(node: ET.Element) -> str:
    """Device-type cluster conformance is expressed as a child element."""
    for child in node:
        tag = child.tag
        if tag == "mandatoryConform":
            return "mandatory"
        if tag == "optionalConform":
            return "optional"
        if tag in {"disallowConform", "deprecateConform"}:
            return "disallowed"
        if tag in {"otherwiseConform", "provisionalConform"}:
            return "otherwise"
    return "unspecified"


def _cluster_attributes(root: ET.Element) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    container = root.find("attributes")
    if container is None:
        return out
    for node in container.findall("attribute"):
        attr_id = _parse_id(node.get("id"))
        name = node.get("name")
        if attr_id is None or not name:
            continue
        record: Dict[str, Any] = {
            "id": attr_id,
            "id_hex": _hex4(attr_id),
            "name": name,
            "type": node.get("type"),
        }
        # A list-typed attribute declares its element type in a child <entry>;
        # without it "list" says nothing about what the list holds.
        entry = node.find("entry")
        if entry is not None and entry.get("type"):
            record["entry_type"] = entry.get("type")
        quality = _quality(node)
        if quality:
            record["quality"] = quality
        access = _access(node)
        if access:
            record["access"] = access
        record["writable"] = _writable(access, has_access=access is not None)
        out.append(record)
    return sorted(out, key=lambda item: item["id"])


def _cluster_enums(root: ET.Element) -> Dict[str, List[Dict[str, Any]]]:
    """Every named enum in a cluster's <dataTypes>, as value -> name (+ summary).

    An attribute's `type` names its enum ("SystemModeEnum"), and the definition
    lives in the cluster's <dataTypes> section. Without this table an agent sees
    `SystemMode = 3` and has to know from somewhere that 3 means Cool -- which is
    exactly the kind of Matter fact that must never come from model memory.

    `summary` is kept only where it says something the name does not. Some items
    carry "(see Terms)", a cross-reference into the PDF spec that is useless to
    an agent and misleading when rendered as a description.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    container = root.find("dataTypes")
    if container is None:
        return out
    for enum in container.findall("enum"):
        enum_name = enum.get("name")
        if not enum_name:
            continue
        items: List[Dict[str, Any]] = []
        for item in enum.findall("item"):
            value = _parse_id(item.get("value"))
            name = item.get("name")
            if value is None or not name:
                continue
            record: Dict[str, Any] = {"value": value, "name": name}
            summary = (item.get("summary") or "").strip()
            if summary and not summary.lower().startswith("(see "):
                record["summary"] = summary
            items.append(record)
        if items:
            out[enum_name] = sorted(items, key=lambda i: i["value"])
    return out


def _cluster_structs(root: ET.Element) -> Dict[str, List[Dict[str, Any]]]:
    """Every named struct in a cluster's <dataTypes>, as an ordered field list.

    A struct-typed attribute carries a JSON object, not a scalar:
    Channel.CurrentChannel is a ChannelInfoStruct and reads back as
    {"MajorNumber": 2, "Name": "KBS2", ...}. Without the field list a consumer
    cannot say what the object contains.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    container = root.find("dataTypes")
    if container is None:
        return out
    for struct in container.findall("struct"):
        struct_name = struct.get("name")
        if not struct_name:
            continue
        fields: List[Dict[str, Any]] = []
        for field in struct.findall("field"):
            name = field.get("name")
            if not name:
                continue
            record: Dict[str, Any] = {"name": name, "type": field.get("type")}
            if field.find("mandatoryConform") is not None:
                record["mandatory"] = True
            fields.append(record)
        if fields:
            out[struct_name] = fields
    return out


def _cluster_commands(root: ET.Element) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    container = root.find("commands")
    if container is None:
        return out
    for node in container.findall("command"):
        cmd_id = _parse_id(node.get("id"))
        name = node.get("name")
        if cmd_id is None or not name:
            continue
        out.append(
            {
                "id": cmd_id,
                "id_hex": _hex2(cmd_id),
                "name": name,
                "direction": node.get("direction"),
            }
        )
    return sorted(out, key=lambda item: (item["id"], item["name"]))


def _cluster_events(root: ET.Element) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    container = root.find("events")
    if container is None:
        return out
    for node in container.findall("event"):
        event_id = _parse_id(node.get("id"))
        name = node.get("name")
        if event_id is None or not name:
            continue
        out.append({"id": event_id, "id_hex": _hex4(event_id), "name": name})
    return sorted(out, key=lambda item: item["id"])


def _cluster_identities(root: ET.Element, path: str) -> List[Tuple[int, str]]:
    """Return every (decimal_id, name) this file defines.

    A single file may define multiple concrete clusters: ResourceMonitoring.xml
    carries HEPA Filter Monitoring (0x0071), Activated Carbon Filter Monitoring
    (0x0072) and Water Tank Level Monitoring (0x0079). Keying only on the root
    id would silently drop those.
    """
    identities: List[Tuple[int, str]] = []
    seen: set[int] = set()

    container = root.find("clusterIds")
    if container is not None:
        for node in container.findall("clusterId"):
            cluster_id = _parse_id(node.get("id"))
            name = node.get("name")
            if cluster_id is None or not name or cluster_id in seen:
                continue
            seen.add(cluster_id)
            identities.append((cluster_id, name))

    if not identities:
        root_id = _parse_id(root.get("id"))
        root_name = root.get("name")
        if root_id is not None and root_name:
            identities.append((root_id, root_name))

    if not identities:
        _ABSTRACT_SKIPPED.append(os.path.basename(path))
    return identities


# Global attributes exist on every cluster and are defined once in the spec's
# globals section rather than repeated in each cluster's <attributes>. Real
# device dumps do report them, so every cluster record gets them appended.
GLOBAL_ATTRIBUTES: List[Dict[str, Any]] = [
    {"id": 0xFFF8, "name": "GeneratedCommandList", "type": "list"},
    {"id": 0xFFF9, "name": "AcceptedCommandList", "type": "list"},
    {"id": 0xFFFA, "name": "EventList", "type": "list"},
    {"id": 0xFFFB, "name": "AttributeList", "type": "list"},
    {"id": 0xFFFC, "name": "FeatureMap", "type": "map32"},
    {"id": 0xFFFD, "name": "ClusterRevision", "type": "uint16"},
]


def _global_attributes() -> List[Dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "id_hex": _hex4(item["id"]),
            "name": item["name"],
            "type": item["type"],
            "global": True,
            # Global attributes are reportable metadata, never client-writable.
            "access": {"read": "true"},
            "writable": False,
        }
        for item in GLOBAL_ATTRIBUTES
    ]


def _merge_attributes(
    own: List[Dict[str, Any]], inherited: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Own definitions win over inherited ones, FIELD BY FIELD.

    A derived cluster often restates an inherited attribute to add conformance
    while omitting everything else -- DishwasherMode redeclares `CurrentMode`
    with no type and no <access>, because Mode Base already says it is a uint8.
    Replacing the base record wholesale would erase those, leaving the attribute
    untyped and its writability unknown, so merge per field and keep whichever
    side actually carries a value.
    """
    merged: Dict[int, Dict[str, Any]] = {item["id"]: dict(item) for item in inherited}
    for item in own:
        base = merged.get(item["id"])
        if base is None:
            merged[item["id"]] = dict(item)
            continue
        for key, value in item.items():
            # `writable` is always present (tri-state), so None means "the
            # derived cluster declared no <access>" -- defer to the base.
            if value is None and base.get(key) is not None:
                continue
            base[key] = value
    return sorted(merged.values(), key=lambda item: item["id"])


def parse_clusters(clusters_dir: str) -> Dict[str, Any]:
    """Parse every cluster file, resolving derived clusters against their bases.

    Two passes are required. 17 of the 124 files declare
    <classification hierarchy="derived" baseCluster="Mode Base"/> and omit the
    attributes they inherit; three of the bases (Alarm Base, Mode Base, Label)
    are abstract and carry no cluster id of their own, so they never appear in
    the output table but must still be available as inheritance sources.
    """
    parsed: List[Dict[str, Any]] = []
    bases_by_name: Dict[str, Dict[str, Any]] = {}

    for path in sorted(glob.glob(os.path.join(clusters_dir, "*.xml"))):
        root = ET.parse(path).getroot()
        if root.tag != "cluster":
            continue
        classification = root.find("classification")
        base_cluster = classification.get("baseCluster") if classification is not None else None
        hierarchy = classification.get("hierarchy") if classification is not None else None
        record = {
            "path": path,
            "revision": _parse_id(root.get("revision")),
            "attributes": _cluster_attributes(root),
            "commands": _cluster_commands(root),
            "events": _cluster_events(root),
            "enums": _cluster_enums(root),
            "structs": _cluster_structs(root),
            "identities": _cluster_identities(root, path),
            "hierarchy": hierarchy,
            "base_cluster": base_cluster,
        }
        parsed.append(record)

        # Index by the cluster's own name and by every identity name, so a
        # baseCluster reference resolves whether the base is abstract
        # ("Alarm Base") or concrete ("Operational State").
        names = {root.get("name") or ""}
        names |= {name for _cid, name in record["identities"]}
        for name in names:
            for candidate in {name, re.sub(r"\s+Cluster$", "", name)}:
                if candidate:
                    bases_by_name.setdefault(candidate, record)

    table: Dict[str, Any] = {}
    globals_list = _global_attributes()

    for record in parsed:
        attributes = record["attributes"]
        commands = record["commands"]
        events = record["events"]
        enums = dict(record["enums"])
        structs = dict(record["structs"])
        inherited_from = None

        if record["hierarchy"] == "derived" and record["base_cluster"]:
            base = bases_by_name.get(record["base_cluster"])
            if base is None:
                raise SystemExit(
                    f"{os.path.basename(record['path'])}: unresolved baseCluster "
                    f"{record['base_cluster']!r}"
                )
            attributes = _merge_attributes(attributes, base["attributes"])
            commands = _merge_attributes(commands, base["commands"])
            events = _merge_attributes(events, base["events"])
            # Enums inherit the same way attributes do: DishwasherMode declares
            # no enums of its own but its CurrentMode is typed by Mode Base's.
            # Own definitions win.
            enums = {**base["enums"], **enums}
            structs = {**base["structs"], **structs}
            inherited_from = record["base_cluster"]

        attributes = _merge_attributes(attributes, globals_list)

        for cluster_id, name in record["identities"]:
            key = str(cluster_id)
            if key in table:
                raise SystemExit(
                    f"Duplicate cluster id {cluster_id} ({_hex4(cluster_id)}): "
                    f"{table[key]['name']!r} vs {name!r} in {os.path.basename(record['path'])}"
                )
            entry = {
                "id_hex": _hex4(cluster_id),
                "name": name,
                "revision": record["revision"],
                "source_file": os.path.basename(record["path"]),
                "attributes": attributes,
                "commands": commands,
                "events": events,
            }
            if enums:
                entry["enums"] = dict(sorted(enums.items()))
            if structs:
                entry["structs"] = dict(sorted(structs.items()))
            if inherited_from:
                entry["inherits_from"] = inherited_from
            table[key] = entry

    return dict(sorted(table.items(), key=lambda kv: int(kv[0])))


def parse_device_types(device_types_dir: str) -> Dict[str, Any]:
    table: Dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(device_types_dir, "*.xml"))):
        root = ET.parse(path).getroot()
        if root.tag != "deviceType":
            continue
        device_type_id = _parse_id(root.get("id"))
        name = root.get("name")
        if device_type_id is None or not name:
            continue

        required: List[Dict[str, Any]] = []
        optional: List[Dict[str, Any]] = []
        container = root.find("clusters")
        for node in (container.findall("cluster") if container is not None else []):
            cluster_id = _parse_id(node.get("id"))
            cluster_name = node.get("name")
            if cluster_id is None or not cluster_name:
                continue
            entry = {
                "id": cluster_id,
                "id_hex": _hex4(cluster_id),
                "name": cluster_name,
                "side": node.get("side"),
            }
            conformance = _conformance(node)
            if conformance == "mandatory":
                required.append(entry)
            elif conformance == "optional":
                optional.append(entry)

        key = str(device_type_id)
        if key in table:
            raise SystemExit(
                f"Duplicate device type id {device_type_id} ({_hex4(device_type_id)}): "
                f"{table[key]['name']!r} vs {name!r} in {os.path.basename(path)}"
            )
        table[key] = {
            "id_hex": _hex4(device_type_id),
            "name": name,
            "revision": _parse_id(root.get("revision")),
            "source_file": os.path.basename(path),
            "required_clusters": sorted(required, key=lambda item: item["id"]),
            "optional_clusters": sorted(optional, key=lambda item: item["id"]),
        }
    return dict(sorted(table.items(), key=lambda kv: int(kv[0])))


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_by_file(root_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in sorted(filenames):
            if filename == "manifest.json":
                continue
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, root_dir)
            out[rel.replace(os.sep, "/")] = _sha256(full)
    return dict(sorted(out.items()))


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize vendored Matter data-model XML into JSON lookup tables."
    )
    parser.add_argument("vendored_dir", help="matter_model/vendor/<VER>/")
    parser.add_argument("--source", default="https://github.com/project-chip/connectedhomeip.git")
    parser.add_argument("--tag", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument("--spec-version", default="")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.vendored_dir)
    clusters_dir = os.path.join(root_dir, "clusters")
    device_types_dir = os.path.join(root_dir, "device_types")
    for path in (clusters_dir, device_types_dir):
        if not os.path.isdir(path):
            print(f"Missing required directory: {path}", file=sys.stderr)
            return 2

    clusters = parse_clusters(clusters_dir)
    device_types = parse_device_types(device_types_dir)
    if not clusters:
        print("No clusters parsed", file=sys.stderr)
        return 2
    if not device_types:
        print("No device types parsed", file=sys.stderr)
        return 2

    _write_json(os.path.join(root_dir, "clusters.json"), clusters)
    _write_json(os.path.join(root_dir, "device-types.json"), device_types)

    manifest = {
        "source": args.source,
        "tag": args.tag,
        "commit_sha": args.commit,
        "spec_version": args.spec_version or os.path.basename(root_dir),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cluster_count": len(clusters),
        "device_type_count": len(device_types),
        "abstract_files_skipped": sorted(set(_ABSTRACT_SKIPPED)),
        "sha256_by_file": _sha256_by_file(root_dir),
    }
    _write_json(os.path.join(root_dir, "manifest.json"), manifest)

    print(
        f"clusters={len(clusters)} device_types={len(device_types)} "
        f"skipped_abstract={len(set(_ABSTRACT_SKIPPED))} -> {root_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
