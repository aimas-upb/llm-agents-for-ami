#!/usr/bin/env python3
"""
Offline lookup API over the vendored Matter data model.

Loads matter_model/vendor/<VER>/clusters.json + device-types.json once and
answers the questions the SimuHome converter needs:

    reg = load_registry()
    reg.cluster_by_name("OnOff")                 -> cluster record (id 6)
    reg.attribute("FanControl", "PercentSetting")-> {'id': 2, 'type': 'percent', ...}
    reg.is_global("Identify", "ClusterRevision") -> True
    reg.device_type(45)                          -> {'name': 'Air Purifier', ...}

No network, no hardcoded Matter facts: every device type, cluster, attribute,
command and event is read from the vendored JSON. Stdlib only.

Name matching is canonicalized (case/space/slash/hyphen-insensitive) because the
spec and real device dumps disagree on punctuation: the spec says "On/Off",
dumps say "OnOff"; the spec says "RVC Operational State", dumps say
"RVCOperationalState".
"""

from __future__ import annotations

import functools
import json
import os
import re
from typing import Any, Dict, List, Optional

VENDOR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")

# SimuHome uses abbreviations the CSA spec spells out. Each entry maps a
# benchmark cluster name to its canonical spec name; verified against the
# vendored tables by ami_agents/environment/integration/SimuHome/matter_model/selftest.py.
CLUSTER_NAME_ALIASES: Dict[str, str] = {
    # Refrigerator And Temperature Controlled Cabinet Mode (0x0052)
    "rtccmode": "Refrigerator And Temperature Controlled Cabinet Mode",
}


def _canon(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


class MatterRegistry:
    """Read-only view over one vendored spec version."""

    def __init__(self, clusters: Dict[str, Any], device_types: Dict[str, Any], manifest: Dict[str, Any]):
        self.clusters = clusters
        self.device_types = device_types
        self.manifest = manifest

        self._cluster_by_canon: Dict[str, Dict[str, Any]] = {}
        for cluster_id, record in clusters.items():
            entry = dict(record)
            entry["id"] = int(cluster_id)
            for variant in (record["name"], re.sub(r"\s+Cluster$", "", record["name"])):
                self._cluster_by_canon.setdefault(_canon(variant), entry)

        self._attr_index: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for cluster_id, record in clusters.items():
            self._attr_index[int(cluster_id)] = {
                _canon(attr["name"]): attr for attr in record.get("attributes", [])
            }

    # -- clusters ----------------------------------------------------------

    def cluster(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        record = self.clusters.get(str(cluster_id))
        if record is None:
            return None
        entry = dict(record)
        entry["id"] = cluster_id
        return entry

    def cluster_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        canon = _canon(name)
        alias = CLUSTER_NAME_ALIASES.get(canon)
        if alias:
            canon = _canon(alias)
        return self._cluster_by_canon.get(canon)

    # -- attributes --------------------------------------------------------

    def attribute(self, cluster_name: str, attribute_name: str) -> Optional[Dict[str, Any]]:
        cluster = self.cluster_by_name(cluster_name)
        if cluster is None:
            return None
        return self._attr_index.get(cluster["id"], {}).get(_canon(attribute_name))

    def attribute_type(self, cluster_name: str, attribute_name: str) -> Optional[str]:
        attribute = self.attribute(cluster_name, attribute_name)
        return attribute.get("type") if attribute else None

    def is_global(self, cluster_name: str, attribute_name: str) -> bool:
        attribute = self.attribute(cluster_name, attribute_name)
        return bool(attribute and attribute.get("global"))

    def is_writable(self, cluster_name: str, attribute_name: str) -> Optional[bool]:
        """Tri-state: True writable, False read-only, None unknown/not found.

        None means the spec declares no <access> for the attribute — callers must
        not treat that as read-only. Note that in Matter most actuation happens
        through cluster *commands*, not attribute writes, so a False here does
        NOT mean the attribute cannot be changed: OnOff.OnOff is read-only yet is
        driven by the On/Off/Toggle commands.
        """
        attribute = self.attribute(cluster_name, attribute_name)
        if attribute is None:
            return None
        return attribute.get("writable")

    def enum_items(self, cluster_name: str, attribute_name: str) -> List[Dict[str, Any]]:
        """The permitted values of an enum-typed attribute, with their names.

        `Thermostat.SystemMode` is a `uint8` on the wire whose type is declared
        as `SystemModeEnum`; this returns [{value: 3, name: "Cool", summary: ...}].
        Empty when the attribute is not enum-typed, or when its enum is not
        defined in the same cluster.
        """
        attribute_type = self.attribute_type(cluster_name, attribute_name)
        if not attribute_type:
            return []
        cluster = self.cluster_by_name(cluster_name)
        if cluster is None:
            return []
        return list((cluster.get("enums") or {}).get(attribute_type) or [])

    def struct_fields(self, cluster_name: str, type_name: str) -> List[Dict[str, Any]]:
        """The fields of a named struct, in declaration order.

        `Channel.CurrentChannel` is a `ChannelInfoStruct`, so its value is an
        object; this says which keys that object has and what type each holds.
        """
        if not type_name:
            return []
        cluster = self.cluster_by_name(cluster_name)
        if cluster is None:
            return []
        return list((cluster.get("structs") or {}).get(type_name) or [])

    def entry_type(self, cluster_name: str, attribute_name: str) -> Optional[str]:
        """Element type of a list-typed attribute (`"list"` alone says nothing)."""
        attribute = self.attribute(cluster_name, attribute_name)
        return attribute.get("entry_type") if attribute else None

    def commands(self, cluster_name: str) -> List[Dict[str, Any]]:
        cluster = self.cluster_by_name(cluster_name)
        return list(cluster.get("commands") or []) if cluster else []

    def events(self, cluster_name: str) -> List[Dict[str, Any]]:
        cluster = self.cluster_by_name(cluster_name)
        return list(cluster.get("events") or []) if cluster else []

    # -- device types ------------------------------------------------------

    def device_type(self, device_type_id: int) -> Optional[Dict[str, Any]]:
        return self.device_types.get(str(device_type_id))

    def device_type_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        canon = _canon(name)
        for record in self.device_types.values():
            if _canon(record["name"]) == canon:
                return record
        return None

    @property
    def spec_version(self) -> str:
        return str(self.manifest.get("spec_version", ""))

    @property
    def provenance(self) -> str:
        return (
            f"{self.manifest.get('source')}@{self.manifest.get('tag')} "
            f"({str(self.manifest.get('commit_sha'))[:12]}) spec {self.spec_version}"
        )


def available_versions() -> List[str]:
    if not os.path.isdir(VENDOR_ROOT):
        return []
    return sorted(
        name for name in os.listdir(VENDOR_ROOT)
        if os.path.isdir(os.path.join(VENDOR_ROOT, name))
    )


@functools.lru_cache(maxsize=4)
def load_registry(version: Optional[str] = None) -> MatterRegistry:
    """Load the vendored registry. Defaults to the highest version present."""
    versions = available_versions()
    if not versions:
        raise RuntimeError(
            f"No vendored Matter data model under {VENDOR_ROOT}. "
            "Run: python ami_agents/environment/integration/SimuHome/matter_model/bootstrap.py"
        )
    chosen = version or versions[-1]
    root = os.path.join(VENDOR_ROOT, chosen)

    def _read(filename: str) -> Dict[str, Any]:
        path = os.path.join(root, filename)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Missing {path}. Run: python ami_agents/environment/integration/SimuHome/matter_model/bootstrap.py"
            )
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    return MatterRegistry(
        clusters=_read("clusters.json"),
        device_types=_read("device-types.json"),
        manifest=_read("manifest.json"),
    )
