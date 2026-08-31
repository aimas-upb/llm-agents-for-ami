#!/usr/bin/env python3
"""
Load the approved Phase A mapping tables and expose them as lookups.

The tables under `tests/simuhome/td/mappings/` are the reviewed Matter->semantics
layer: which Matter attribute is which SOSA property, what an affordance is
called, which SAREF/ex classes a device family carries, which room maps to which
`s4bldg:BuildingSpace` subclass. They were curated and approved row by row, so
SHTD reads them rather than re-deriving anything.

Only `ha_binding.yaml` is skipped -- it maps affordances onto Home Assistant
services, which SHTD replaces with direct Matter calls.

A row is honoured only when `status` is `approved` or `auto`; anything still
marked `review`/`todo` is reported by `unsettled()` instead of being silently
used, so an unreviewed decision cannot leak into a served Thing Description.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

MAPPINGS_DIR = (
    Path(__file__).resolve().parents[4] / "tests" / "simuhome" / "td" / "mappings"
)

# Rows in these states are trusted. Anything else still needs a human.
SETTLED = {"approved", "auto"}


def _load_table(name: str) -> Dict[str, Any]:
    path = MAPPINGS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"mapping table not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _rows(name: str, settled_only: bool = True) -> List[Dict[str, Any]]:
    rows = _load_table(name).get("rows") or []
    if not settled_only:
        return rows
    return [r for r in rows if (r.get("status") or "auto") in SETTLED]


class Mappings:
    """Every Phase A table, indexed for lookup by the TD generator."""

    def __init__(self) -> None:
        # key is "<cluster_id>.<attribute_id>", but we index by the readable
        # `yaml_path` ("OnOff.OnOff") because that is what the simulator reports.
        self.attributes: Dict[str, Dict[str, Any]] = {}
        for row in _rows("attribute_map"):
            self.attributes[str(row.get("yaml_path"))] = row

        self.device_types: Dict[str, Dict[str, Any]] = {
            str(r["key"]): r for r in _rows("device_type_map")
        }
        self.rooms: Dict[str, Dict[str, Any]] = {
            str(r["key"]): r for r in _rows("room_map")
        }
        self.observable_properties: Dict[str, Dict[str, Any]] = {
            str(r["key"]): r for r in _rows("observable_properties")
        }
        # Environmental properties, keyed by the room-state token SimuHome uses
        # ("temperature", "humidity", ...) rather than by table key.
        self.room_state_properties: Dict[str, Dict[str, Any]] = {}
        for row in self.observable_properties.values():
            key = str(row.get("key", ""))
            if key.startswith("room_state."):
                self.room_state_properties[key.split(".", 1)[1]] = row

        # Generic, cross-family property classes ("homeont:OnOff"), keyed by
        # the class IRI itself.
        self.property_classes: Dict[str, Dict[str, Any]] = {
            str(r["key"]): r for r in _rows("property_classes")
        }
        # The same table indexed by what the TD generator actually has in hand
        # when it emits a property: the Matter path.
        self.property_class_by_path: Dict[str, Dict[str, Any]] = {
            str(r["yaml_path"]): r
            for r in self.property_classes.values() if r.get("yaml_path")
        }

        # Per-family leaves ("homeont:AirConditionerOnOff"), keyed by
        # (family, yaml_path) -- the leaf is what a device instantiates.
        self.actuatable_leaves: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in _rows("actuatable_properties"):
            self.actuatable_leaves[
                (str(row.get("family")), str(row.get("yaml_path")))] = row

        # Observable classes, keyed by Matter path. Added in phase 6: until
        # then every observable device property reached the TD untyped.
        self.observable_classes: Dict[str, Dict[str, Any]] = {}
        for row in _rows("observable_property_classes"):
            if row.get("yaml_path"):
                self.observable_classes[str(row["yaml_path"])] = row

        # Which attribute each cluster's command writes. The Matter XML states
        # no command->attribute relation, so this curated table is the only
        # source -- `classify.py` reads it from here rather than keeping its
        # own copy, which could drift.
        # Keyed by CLUSTER ID, not by the display name. The table spells a
        # cluster as the Matter registry does ("Refrigerator And Temperature
        # Controlled Cabinet Mode") while the simulator uses a compact token
        # ("RTCCMode"); stripping spaces does not bridge that, and silently
        # missing the row would de-actuate every freezer's mode control.
        # `command_targets_for` resolves a caller's spelling through the
        # registry, which canonicalises both.
        self.command_targets_by_cluster_id: Dict[int, set] = {}
        for row in _rows("command_targets"):
            try:
                cluster_id = int(str(row.get("key", "")).split(".")[0])
            except (TypeError, ValueError):
                continue
            attribute = row.get("writes_attribute")
            if attribute:
                self.command_targets_by_cluster_id.setdefault(
                    cluster_id, set()).add(str(attribute))

        # Which device families act on a ROOM environmental variable, from the
        # approved actuation_effects table. This decides which devices link to
        # the room's sosa:FeatureOfInterest.
        #
        # Sensing is deliberately NOT a second criterion. The only families that
        # measure a room quantity without affecting it are Freezer and
        # Refrigerator, and both measure their own compartment: a freezer reads
        # -15 C inside a 23.6 C kitchen. Treating that as an observation of the
        # room's air would be wrong, not merely redundant.
        # Which family senses which property, and whose property it is. Keyed by
        # family; `featureOfInterest` is "room" or "appliance_interior".
        self.sensing: Dict[str, Dict[str, Any]] = {
            str(r["family"]): r for r in _rows("sensing")
        }

        self.families_affecting_environment: Dict[str, set] = {}
        # (family, cluster) -> the effect rows for commands of that cluster.
        # The table is keyed by COMMAND while an action affordance is keyed by
        # the attribute the command drives, so effects are grouped by cluster
        # and matched to the affordance through command_targets.
        self.effects_by_cluster: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in _rows("actuation_effects"):
            prop = row.get("affectsObservableProperty")
            if not prop:
                continue
            family = str(row["family"])
            self.families_affecting_environment.setdefault(family, set()).add(str(prop))
            cluster = str((row.get("command") or {}).get("cluster_name") or "")
            # The tables spell cluster names as the registry does ("On/Off",
            # "Level Control"); the simulator uses the compact token.
            token = cluster.replace("/", "").replace(" ", "")
            self.effects_by_cluster.setdefault((family, token), []).append(row)

    # -- lookups -----------------------------------------------------------

    def attribute(self, cluster: str, attribute: str) -> Optional[Dict[str, Any]]:
        """The approved row for one Matter attribute, by `Cluster.Attribute`."""
        return self.attributes.get(f"{cluster}.{attribute}")

    def affordance_name(self, cluster: str, attribute: str) -> str:
        """The TD `td:name` for an attribute.

        Falls back to a lowerCamel form of the attribute name so an attribute
        the tables have not seen still gets a stable, usable name instead of
        vanishing from the Thing Description.
        """
        row = self.attribute(cluster, attribute)
        name = (row or {}).get("affordance_name")
        if name:
            return str(name)
        return attribute[:1].lower() + attribute[1:] if attribute else attribute

    def device_family(self, device_type: str) -> Optional[Dict[str, Any]]:
        """Row for a SimuHome `device_type`.

        SimuHome reports snake_case (`air_conditioner`); the tables are keyed by
        the CamelCase family name (`AirConditioner`).
        """
        if device_type in self.device_types:
            return self.device_types[device_type]
        camel = "".join(part.capitalize() for part in str(device_type).split("_"))
        return self.device_types.get(camel)

    def room(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Row for a SimuHome room id (`living_room` -> `LivingRoom`)."""
        if room_id in self.rooms:
            return self.rooms[room_id]
        camel = "".join(part.capitalize() for part in str(room_id).split("_"))
        return self.rooms.get(camel)

    def room_state_property(self, token: str) -> Optional[Dict[str, Any]]:
        """The observable-property row for a room-state key SimuHome reports."""
        row = self.room_state_properties.get(token)
        if row is not None:
            return row
        # `air_quality` is reported by some endpoints where the table calls the
        # measured quantity pm10; match on the declared aliases too.
        for candidate in self.room_state_properties.values():
            if token in (candidate.get("legacy_aliases") or []):
                return candidate
        return None

    # -- review status -----------------------------------------------------

    def senses(self, device_type: str) -> Optional[Dict[str, Any]]:
        """The sensing row for a SimuHome device_type, if the family senses."""
        family = self.device_family(device_type) or {}
        return self.sensing.get(str(family.get("key") or ""))

    def effects_for(self, device_type: str, cluster: str) -> List[Dict[str, Any]]:
        """Approved effect rows for one family's cluster, if any."""
        family = self.device_family(device_type) or {}
        return self.effects_by_cluster.get(
            (str(family.get("key") or ""), cluster), [])

    def command_targets_for(self, cluster: str) -> set:
        """Which attributes this cluster's commands write.

        Accepts either spelling of the cluster name -- the registry canonicalises
        "RTCCMode" and "Refrigerator And Temperature Controlled Cabinet Mode" to
        the same cluster.
        """
        from matter_model.registry import load_registry
        record = load_registry().cluster_by_name(cluster)
        if record is None:
            return set()
        return self.command_targets_by_cluster_id.get(int(record.get("id", -1)), set())

    def property_class(self, device_type: str, cluster: str,
                       attribute: str) -> Optional[str]:
        """The most specific homeont class for one device property.

        Leaf first (`homeont:AirConditionerOnOff`), then the generic
        cross-family class (`homeont:OnOff`), then the observable table. Returns
        None when no approved row covers the attribute, so an unclassed property
        is visible as a gap rather than guessed at.
        """
        path = f"{cluster}.{attribute}"
        family = (self.device_family(device_type) or {}).get("key")
        if family:
            leaf = self.actuatable_leaves.get((str(family), path))
            if leaf and leaf.get("property_type"):
                return str(leaf["property_type"])
        generic = self.property_class_by_path.get(path)
        if generic and generic.get("homeont_class"):
            return str(generic["homeont_class"])
        observable = self.observable_classes.get(path)
        if observable and observable.get("homeont_class"):
            return str(observable["homeont_class"])
        return None

    def action_class(self, device_type: str, cluster: str,
                     attribute: str) -> Optional[str]:
        """The saref:Command subclass for one action affordance.

        Keyed by the ATTRIBUTE the action drives, matching how the affordance
        itself is keyed -- `onOff(true|false)` stands for On/Off/Toggle.
        """
        path = f"{cluster}.{attribute}"
        family = (self.device_family(device_type) or {}).get("key")
        if family:
            leaf = self.actuatable_leaves.get((str(family), path))
            if leaf and leaf.get("action_class"):
                return str(leaf["action_class"])
        generic = self.property_class_by_path.get(path)
        if generic and generic.get("action_class"):
            return str(generic["action_class"])
        return None

    def unsettled(self) -> Dict[str, int]:
        """Count rows per table that are still awaiting review.

        `ha_binding` is excluded: SHTD does not use it.
        """
        out: Dict[str, int] = {}
        for name in (
            "attribute_map", "device_type_map", "room_map",
            "observable_properties", "property_classes",
            "observable_property_classes",
            "actuatable_properties", "internal_properties",
            "command_targets", "actuation_effects", "sensing",
        ):
            try:
                pending = len(_rows(name, settled_only=False)) - len(_rows(name))
            except FileNotFoundError:
                continue
            if pending:
                out[name] = pending
        return out


@lru_cache(maxsize=1)
def load_mappings() -> Mappings:
    return Mappings()


if __name__ == "__main__":
    m = load_mappings()
    print(f"attributes          {len(m.attributes)}")
    print(f"device families     {len(m.device_types)}")
    print(f"rooms               {len(m.rooms)}")
    print(f"observable props    {len(m.observable_properties)}")
    print(f"room-state props    {sorted(m.room_state_properties)}")
    pending = m.unsettled()
    print(f"unsettled           {pending or 'none'}")
