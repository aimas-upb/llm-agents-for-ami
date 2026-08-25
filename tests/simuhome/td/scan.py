#!/usr/bin/env python3
"""
Scan the SimuHome benchmark YAML and build the distinct vocabulary Phase A needs.

The YAML is the primary input: it is the Home Assistant layer that Thing
Descriptions are actually invoked over, and it carries `platform` (the HA
domain), which the source JSON does not. Every Matter fact — cluster id,
attribute id, type, writability, commands — is resolved from the vendored
registry, never from memory.

Name grammar (verified across all 600 episodes):

    <Scope>.Clock                                 2 segments  simulator meta
    <Scope>.Simulator.TickInterval                3 segments  simulator meta
    <Scope>.<Room>.{Temperature|Humidity|         3 segments  room observation
                    Illuminance|Pm10}                         (feature of interest)
    <Scope>.<Room>.<Device>                       3 segments  a Thing
    <Scope>.<Room>.<Device>.<Cluster>.<Attribute> 5 segments  a Thing's attribute

The 3-segment forms are ambiguous by shape, so they are separated by token:
the four environment tokens are always `platform: sensor` and never have
5-segment children, while device Things always do.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from . import registry_access

# Room-state tokens that denote an environment observation rather than a device.
ENVIRONMENT_TOKENS = ("Temperature", "Humidity", "Illuminance", "Pm10")

# Third-segment token for simulator bookkeeping, excluded from rooms and things.
SIMULATOR_ROOM_TOKEN = "Simulator"


def _strip_instance_digits(token: str) -> str:
    """`AirPurifier1` -> `AirPurifier`. Verified 1:1 with SimuHome device_type."""
    return re.sub(r"\d+$", "", token)


@dataclass
class Corpus:
    """Distinct vocabulary aggregated across every episode."""

    episodes: int = 0
    # family -> {platform, instances, episodes, clusters, attributes, rooms}
    families: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # room token -> {instances, episodes, state_tokens}
    rooms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # (cluster_token, attribute_token) -> {occurrences, families, units, samples}
    attributes: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    # cluster_token -> occurrences
    clusters: Counter = field(default_factory=Counter)
    # room state token -> {occurrences, units, device_classes}
    room_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # family -> ProductName, from the source JSON (one-time 16-row lookup)
    titles: Dict[str, str] = field(default_factory=dict)
    unresolved: Counter = field(default_factory=Counter)
    platform_counts: Counter = field(default_factory=Counter)
    sample_entity_names: Dict[str, str] = field(default_factory=dict)


def _family_record(corpus: Corpus, family: str) -> Dict[str, Any]:
    return corpus.families.setdefault(
        family,
        {
            "platforms": Counter(),
            "instances": 0,
            "episodes": set(),
            "clusters": set(),
            "attributes": set(),
            "rooms": set(),
        },
    )


def _attribute_record(corpus: Corpus, key: Tuple[str, str]) -> Dict[str, Any]:
    return corpus.attributes.setdefault(
        key,
        {
            "occurrences": 0,
            "families": set(),
            "units": Counter(),
            "device_classes": Counter(),
            "samples": [],
        },
    )


def _index_episode(corpus: Corpus, path: str, payload: Dict[str, Any]) -> None:
    episode = os.path.basename(path)
    devices = payload.get("devices") or {}

    # First pass: which 3-segment device tokens have 5-segment children.
    has_children: Set[Tuple[str, str]] = set()
    for entries in devices.values():
        for entry in entries or []:
            name = entry.get("name")
            if not isinstance(name, str):
                continue
            segments = name.split(".")
            if len(segments) == 5:
                has_children.add((segments[1], segments[2]))

    for entries in devices.values():
        for entry in entries or []:
            name = entry.get("name")
            platform = entry.get("platform")
            if not isinstance(name, str) or not isinstance(platform, str):
                continue
            corpus.platform_counts[platform] += 1
            segments = name.split(".")

            if len(segments) == 2:
                continue  # <Scope>.Clock — simulator meta

            room_token = segments[1]
            if room_token == SIMULATOR_ROOM_TOKEN:
                continue  # <Scope>.Simulator.TickInterval — simulator meta

            if len(segments) == 3:
                token = segments[2]
                if token in ENVIRONMENT_TOKENS and (room_token, token) not in has_children:
                    state = corpus.room_states.setdefault(
                        token, {"occurrences": 0, "units": Counter(), "device_classes": Counter()}
                    )
                    state["occurrences"] += 1
                    if entry.get("unit_of_measurement"):
                        state["units"][entry["unit_of_measurement"]] += 1
                    if entry.get("class"):
                        state["device_classes"][entry["class"]] += 1

                    room = corpus.rooms.setdefault(
                        room_token, {"instances": 0, "episodes": set(), "state_tokens": set()}
                    )
                    room["episodes"].add(episode)
                    room["state_tokens"].add(token)
                    corpus.sample_entity_names.setdefault(f"room:{token}", name)
                    continue

                # A device Thing.
                family = _strip_instance_digits(token)
                record = _family_record(corpus, family)
                record["platforms"][platform] += 1
                record["instances"] += 1
                record["episodes"].add(episode)
                record["rooms"].add(room_token)
                room = corpus.rooms.setdefault(
                    room_token, {"instances": 0, "episodes": set(), "state_tokens": set()}
                )
                room["instances"] += 1
                room["episodes"].add(episode)
                corpus.sample_entity_names.setdefault(f"thing:{family}", name)
                continue

            if len(segments) == 5:
                family = _strip_instance_digits(segments[2])
                cluster_token, attribute_token = segments[3], segments[4]
                record = _family_record(corpus, family)
                record["clusters"].add(cluster_token)
                record["attributes"].add((cluster_token, attribute_token))

                corpus.clusters[cluster_token] += 1
                attr = _attribute_record(corpus, (cluster_token, attribute_token))
                attr["occurrences"] += 1
                attr["families"].add(family)
                if entry.get("unit_of_measurement"):
                    attr["units"][entry["unit_of_measurement"]] += 1
                if entry.get("class"):
                    attr["device_classes"][entry["class"]] += 1
                if len(attr["samples"]) < 3 and "initial_value" in entry:
                    attr["samples"].append(entry["initial_value"])
                corpus.sample_entity_names.setdefault(
                    f"attr:{cluster_token}.{attribute_token}", name
                )


def _load_titles(input_dir: str) -> Dict[str, str]:
    """device family -> BasicInformation.ProductName, from the source JSON.

    Read once, not per episode: ProductName is 1:1 with the device family across
    all 600 episodes, so this is a 16-row constant. It is the only thing Phase A
    takes from the JSON, and only because the YAML filters BasicInformation.* as
    boilerplate. Lives on endpoint 0 (the Matter root node).
    """
    titles: Dict[str, str] = {}
    conflicts: Dict[str, Set[str]] = defaultdict(set)
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        rooms = ((payload.get("initial_home_config") or {}).get("rooms")) or {}
        for room_id, room_cfg in rooms.items():
            for device in (room_cfg or {}).get("devices") or []:
                device_type = device.get("device_type")
                if not device_type:
                    continue
                for key, value in (device.get("attributes") or {}).items():
                    if key.endswith("BasicInformation.ProductName") and isinstance(value, str):
                        family = "".join(
                            part[:1].upper() + part[1:] for part in device_type.split("_")
                        )
                        conflicts[family].add(value)
                        titles[family] = value
        if len(titles) >= 16 and all(len(v) == 1 for v in conflicts.values()):
            # Vocabulary is closed; stop early rather than parsing all 600.
            break
    return titles


def _flag_unresolved(corpus: Corpus) -> None:
    """Record any cluster.attribute the vendored registry cannot resolve."""
    reg = registry_access.get_registry()
    corpus.unresolved = Counter()
    for cluster_token, attribute_token in corpus.attributes:
        if reg.attribute(cluster_token, attribute_token) is None:
            corpus.unresolved[f"{cluster_token}.{attribute_token}"] += 1


def _cache_signature(paths: List[str]) -> str:
    """Cheap fingerprint of the input set: path, size and mtime of every file."""
    digest = hashlib.sha256()
    for path in paths:
        stat = os.stat(path)
        digest.update(f"{path}:{stat.st_size}:{int(stat.st_mtime)}\n".encode("utf-8"))
    return digest.hexdigest()


def _to_cacheable(corpus: Corpus) -> Dict[str, Any]:
    return {
        "episodes": corpus.episodes,
        "families": {
            name: {
                "platforms": dict(record["platforms"]),
                "instances": record["instances"],
                "episodes": sorted(record["episodes"]),
                "clusters": sorted(record["clusters"]),
                "attributes": sorted(list(pair) for pair in record["attributes"]),
                "rooms": sorted(record["rooms"]),
            }
            for name, record in corpus.families.items()
        },
        "rooms": {
            name: {
                "instances": record["instances"],
                "episodes": sorted(record["episodes"]),
                "state_tokens": sorted(record["state_tokens"]),
            }
            for name, record in corpus.rooms.items()
        },
        "attributes": {
            f"{cluster}\t{attribute}": {
                "occurrences": record["occurrences"],
                "families": sorted(record["families"]),
                "units": dict(record["units"]),
                "device_classes": dict(record["device_classes"]),
                "samples": record["samples"],
            }
            for (cluster, attribute), record in corpus.attributes.items()
        },
        "clusters": dict(corpus.clusters),
        "room_states": {
            name: {
                "occurrences": record["occurrences"],
                "units": dict(record["units"]),
                "device_classes": dict(record["device_classes"]),
            }
            for name, record in corpus.room_states.items()
        },
        "titles": corpus.titles,
        "platform_counts": dict(corpus.platform_counts),
        "sample_entity_names": corpus.sample_entity_names,
    }


def _from_cacheable(data: Dict[str, Any]) -> Corpus:
    corpus = Corpus(episodes=data["episodes"])
    corpus.families = {
        name: {
            "platforms": Counter(record["platforms"]),
            "instances": record["instances"],
            "episodes": set(record["episodes"]),
            "clusters": set(record["clusters"]),
            "attributes": {tuple(pair) for pair in record["attributes"]},
            "rooms": set(record["rooms"]),
        }
        for name, record in data["families"].items()
    }
    corpus.rooms = {
        name: {
            "instances": record["instances"],
            "episodes": set(record["episodes"]),
            "state_tokens": set(record["state_tokens"]),
        }
        for name, record in data["rooms"].items()
    }
    corpus.attributes = {
        tuple(key.split("\t", 1)): {
            "occurrences": record["occurrences"],
            "families": set(record["families"]),
            "units": Counter(record["units"]),
            "device_classes": Counter(record["device_classes"]),
            "samples": record["samples"],
        }
        for key, record in data["attributes"].items()
    }
    corpus.clusters = Counter(data["clusters"])
    corpus.room_states = {
        name: {
            "occurrences": record["occurrences"],
            "units": Counter(record["units"]),
            "device_classes": Counter(record["device_classes"]),
        }
        for name, record in data["room_states"].items()
    }
    corpus.titles = data["titles"]
    corpus.platform_counts = Counter(data["platform_counts"])
    corpus.sample_entity_names = data["sample_entity_names"]
    return corpus


def scan(input_dir: str, *, read_titles: bool = True, cache_path: Optional[str] = None) -> Corpus:
    """Scan the corpus, optionally memoising the result keyed on input mtimes."""
    paths = sorted(glob.glob(os.path.join(input_dir, "*.yaml")))
    signature = _cache_signature(paths) if (paths and cache_path) else None

    corpus: Optional[Corpus] = None
    if signature and cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if cached.get("signature") == signature and cached.get("read_titles") == read_titles:
                corpus = _from_cacheable(cached["corpus"])
        except Exception:
            corpus = None  # A malformed or stale cache is never fatal; rescan.

    if corpus is not None:
        _flag_unresolved(corpus)
        return corpus

    corpus = _scan_uncached(input_dir, paths, read_titles=read_titles)
    _flag_unresolved(corpus)

    if signature and cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = f"{cache_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(
                {"signature": signature, "read_titles": read_titles, "corpus": _to_cacheable(corpus)},
                handle,
            )
        os.replace(tmp, cache_path)

    return corpus


def _scan_uncached(input_dir: str, paths: List[str], *, read_titles: bool) -> Corpus:
    corpus = Corpus()
    if not paths:
        raise FileNotFoundError(
            f"No *.yaml found in {input_dir}. Point --input at the SimuHome benchmark "
            "directory, and run initial_home_config_to_homeassistant_yaml.py first."
        )

    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            continue
        corpus.episodes += 1
        _index_episode(corpus, path, payload)

    if read_titles:
        corpus.titles = _load_titles(input_dir)

    return corpus


def family_platform(corpus: Corpus, family: str) -> Optional[str]:
    """The single HA domain a family maps to, or None if ambiguous."""
    platforms = corpus.families[family]["platforms"]
    return platforms.most_common(1)[0][0] if len(platforms) == 1 else None


def sorted_families(corpus: Corpus) -> List[str]:
    return sorted(corpus.families)


def sorted_rooms(corpus: Corpus) -> List[str]:
    return sorted(corpus.rooms)


def sorted_attributes(corpus: Corpus) -> List[Tuple[str, str]]:
    """Attribute keys ordered by resolved decimal cluster/attribute id."""
    reg = registry_access.get_registry()

    def order(key: Tuple[str, str]) -> Tuple[int, int, str, str]:
        cluster = reg.cluster_by_name(key[0])
        attribute = reg.attribute(key[0], key[1])
        return (
            cluster["id"] if cluster else 1 << 30,
            attribute["id"] if attribute else 1 << 30,
            key[0],
            key[1],
        )

    return sorted(corpus.attributes, key=order)
