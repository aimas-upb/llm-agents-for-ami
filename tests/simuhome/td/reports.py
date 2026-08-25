#!/usr/bin/env python3
"""inventory.md (what exists) and coverage.md (what is decided vs outstanding)."""

from __future__ import annotations

from typing import Any, Dict, List

from . import registry_access, scan, tables
from .scan import Corpus


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def inventory(corpus: Corpus, generated: Dict[str, Any]) -> str:
    reg = registry_access.get_registry()
    lines: List[str] = []
    add = lines.append

    add("# SimuHome Phase A — Inventory")
    add("")
    add(f"- Matter data model: `{reg.provenance}`")
    add(f"- Episodes scanned: **{corpus.episodes}**")
    add(f"- Source: SimuHome benchmark YAML (the Home Assistant layer TDs are invoked over)")
    add("")

    add("## Scale")
    add("")
    add(_table(
        ["what", "distinct", "note"],
        [
            ["device families", len(corpus.families), "3rd name segment, digits stripped"],
            ["rooms", len(corpus.rooms), "`Simulator` excluded as bookkeeping"],
            ["clusters in use", len(corpus.clusters), f"of {len(reg.clusters)} in the spec"],
            ["cluster.attribute pairs", len(corpus.attributes), "the attribute_map row count"],
            ["room state variables", len(corpus.room_states), "the environmental observables"],
            ["entities", sum(corpus.platform_counts.values()), "across all episodes"],
        ],
    ))
    add("")
    add("Home Assistant platforms: " + ", ".join(
        f"`{k}` {v}" for k, v in sorted(corpus.platform_counts.items(), key=lambda x: -x[1])
    ))
    add("")

    add("## Rooms")
    add("")
    add(_table(
        ["room", "instances", "episodes", "state variables"],
        [
            [room, corpus.rooms[room]["instances"], len(corpus.rooms[room]["episodes"]),
             ", ".join(sorted(corpus.rooms[room]["state_tokens"]))]
            for room in scan.sorted_rooms(corpus)
        ],
    ))
    add("")

    add("## Device families")
    add("")
    add(_table(
        ["family", "HA platform", "title", "instances", "episodes", "clusters"],
        [
            [
                family,
                scan.family_platform(corpus, family),
                corpus.titles.get(family, ""),
                corpus.families[family]["instances"],
                len(corpus.families[family]["episodes"]),
                len(corpus.families[family]["clusters"]),
            ]
            for family in scan.sorted_families(corpus)
        ],
    ))
    add("")

    add("## Clusters")
    add("")
    cluster_rows = []
    for token in sorted(corpus.clusters, key=lambda t: (reg.cluster_by_name(t) or {}).get("id", 1 << 30)):
        cluster = reg.cluster_by_name(token)
        used = sum(1 for c, _a in corpus.attributes if c == token)
        cluster_rows.append([
            cluster["id"] if cluster else "?",
            cluster["id_hex"] if cluster else "?",
            cluster["name"] if cluster else f"UNRESOLVED {token}",
            token,
            used,
            len(cluster["attributes"]) if cluster else 0,
            len(cluster["commands"]) if cluster else 0,
            len(cluster["events"]) if cluster else 0,
        ])
    add(_table(
        ["id", "hex", "spec name", "YAML token", "attrs used", "attrs in spec", "commands", "events"],
        cluster_rows,
    ))
    add("")

    add("## Attribute census")
    add("")
    attr_rows = []
    for cluster_token, attribute_token in scan.sorted_attributes(corpus):
        record = corpus.attributes[(cluster_token, attribute_token)]
        attribute = reg.attribute(cluster_token, attribute_token)
        cluster = reg.cluster_by_name(cluster_token)
        writable = reg.is_writable(cluster_token, attribute_token)
        attr_rows.append([
            f"{cluster['id']}.{attribute['id']}" if cluster and attribute else "?",
            f"{cluster_token}.{attribute_token}",
            attribute.get("type") if attribute else "?",
            {True: "yes", False: "no", None: "unknown"}[writable],
            record["occurrences"],
            len(record["families"]),
            ", ".join(sorted(record["units"])) or "",
        ])
    add(_table(
        ["key", "YAML path", "type", "writable", "occurrences", "families", "unit"],
        attr_rows,
    ))
    add("")

    add("## Commands available (the actuation surface)")
    add("")
    command_rows = []
    for token in sorted(corpus.clusters, key=lambda t: (reg.cluster_by_name(t) or {}).get("id", 1 << 30)):
        commands = reg.commands(token)
        if not commands:
            continue
        command_rows.append([
            token,
            len(commands),
            ", ".join(f"{c['name']}({c['id']})" for c in commands),
        ])
    add(_table(["cluster", "n", "commands"], command_rows))
    add("")

    add("## Anomalies")
    add("")
    add("These are properties of the upstream SimuHome data, recorded so downstream")
    add("consumers do not rediscover them:")
    add("")
    add("1. **Matter device-type ids are unusable.** Only 4 of 16 families declare a")
    add("   `Descriptor.DeviceTypeList` at all, and all of them report `45`, which the")
    add("   registry resolves to *Air Purifier* — so SimuHome's freezers, heat pumps and")
    add("   refrigerators claim to be air purifiers. `device_type_map.yaml` is therefore")
    add("   keyed on the device family token, and no Matter device-type id is emitted.")
    add("2. **Read-only attributes are still actuated.** Matter actuates through cluster")
    add("   *commands*, not attribute writes. `OnOff.OnOff` is read-only in the spec yet is")
    add("   the most-actuated property in the benchmark; the same holds for")
    add("   `WindowCovering.TargetPositionLiftPercent100ths`. Affordance classification")
    add("   therefore uses `writable OR command-driven`, never writability alone.")
    add("3. **Protocol plumbing is filtered upstream.** `Descriptor.*`, `Identify.*`,")
    add("   `BasicInformation.*`, `PowerTopology.*` and the global attributes")
    add("   (`ClusterRevision`, `FeatureMap`, …) are absent from the YAML by design. The")
    add("   only casualty of value was `BasicInformation.ProductName`, recovered as a")
    add("   16-row constant because it is 1:1 with the device family.")
    fan_mode = corpus.attributes.get(("FanControl", "FanMode"))
    fan_seq = corpus.attributes.get(("FanControl", "FanModeSequence"))
    if fan_mode and fan_seq:
        add("4. **`FanMode` contradicts `FanModeSequence`.** Every fan-bearing device")
        add("   declares `FanModeSequence: OffLowHigh`, which excludes *Medium* — yet")
        add("   devices across the corpus report `FanMode: Medium`, and benchmark queries")
        add("   ask for a \"medium\" fan in plain language. The capability declaration and")
        add("   the live state disagree, so a schema derived from `FanModeSequence` would")
        add("   reject both the current value and the user's request. `ha_binding.yaml`")
        add("   therefore enumerates the modes the corpus actually uses")
        add("   (`off, low, medium, high`), not the ones the device claims to support.")
        add("")
        add("   `FanModeSequence` is a **per-instance** capability, like `min_temp` /")
        add("   `max_temp`: it constrains the legal values of `FanMode` for one deployed")
        add("   device. Phase B must read it per episode and attach it to that Thing's")
        add("   anonymous property node — it cannot live on the property class.")
        add("")
        add("5. **`FanControl.Step` is unreachable.** Home Assistant's `fan` domain")
        add("   exposes no step service, and no query in the benchmark asks for a")
        add("   relative fan change: of the fan-mentioning queries, the overwhelming")
        add("   majority give an absolute percentage and the rest name a mode. It is")
        add("   omitted from `actuation_effects` rather than asserting an effect that")
        add("   nothing can trigger.")
        next_index = 6
    else:
        next_index = 4

    if corpus.unresolved:
        add(f"{next_index}. **Unresolved against the registry: {len(corpus.unresolved)}** — "
            + ", ".join(sorted(corpus.unresolved)))
    else:
        add(f"{next_index}. **Registry resolution is complete**: every one of the "
            f"{len(corpus.attributes)} distinct `cluster.attribute` pairs resolves.")
    add("")
    return "\n".join(lines) + "\n"


def coverage(
    corpus: Corpus,
    generated: Dict[str, Any],
    merge_stats: Dict[str, Dict[str, int]],
) -> str:
    lines: List[str] = []
    add = lines.append
    reg = registry_access.get_registry()

    add("# SimuHome Phase A — Coverage")
    add("")
    add(f"- Matter data model: `{reg.provenance}`")
    add(f"- Episodes scanned: **{corpus.episodes}**")
    add("")

    add("## Status matrix")
    add("")
    total = {tables.AUTO: 0, tables.REVIEW: 0, tables.TODO: 0, tables.APPROVED: 0}
    matrix_rows = []
    for name in sorted(generated):
        counts = tables.count_statuses(generated[name])
        for key in total:
            total[key] += counts[key]
        n = sum(counts.values())
        matrix_rows.append([
            name, n, counts[tables.AUTO], counts[tables.REVIEW],
            counts[tables.TODO], counts[tables.APPROVED],
            f"{counts[tables.REVIEW] + counts[tables.TODO]} of {n}" if n else "-",
        ])
    grand = sum(total.values())
    outstanding = total[tables.REVIEW] + total[tables.TODO]
    matrix_rows.append([
        "**total**", grand, total[tables.AUTO], total[tables.REVIEW],
        total[tables.TODO], total[tables.APPROVED],
        f"**{outstanding} of {grand}**" if grand else "-",
    ])
    add(_table(
        ["table", "rows", "auto", "review", "todo", "approved", "rows awaiting you"],
        matrix_rows,
    ))
    add("")
    add("`rows awaiting you` counts `review` + `todo` — the rows that still need a human")
    add("decision. `auto` rows are machine-determined and need no attention; `approved` rows")
    add("you have already signed off. A table reading `9 of 9` needs you on every row;")
    add("`0 of 116` needs you on none.")
    add("")

    add("## Outstanding `todo` rows")
    add("")
    todo_by_reason: Dict[str, List[str]] = {}
    for name in sorted(generated):
        items = generated[name].get("rows", []) if isinstance(generated[name], dict) else generated[name]
        for row in items:
            if row.get("status") != tables.TODO:
                continue
            reason = row.get("reason") or row.get("foi_iri_reason") or "unspecified"
            todo_by_reason.setdefault(reason, []).append(f"{name}: {row['key']}")
    if not todo_by_reason:
        add("None.")
    else:
        for reason in sorted(todo_by_reason):
            keys = todo_by_reason[reason]
            add(f"### {reason} ({len(keys)})")
            add("")
            for key in sorted(keys)[:40]:
                add(f"- `{key}`")
            if len(keys) > 40:
                add(f"- … and {len(keys) - 40} more")
            add("")

    add("## Directionality")
    add("")
    effects = generated.get("actuation_effects", {})
    rows = effects.get("rows", []) if isinstance(effects, dict) else effects
    directional = [r for r in rows if r.get("monotonic") is True]
    neutral = [r for r in rows if r.get("monotonic") is False]
    add(f"- **{len(directional)}** commands proposed as directional "
        "(monotonic, with an increase/decrease direction).")
    add(f"- **{len(neutral)}** proposed as directionless *on purpose* — toggles and "
        "level/percent setters have no inherent direction, matching the repo's existing "
        "treatment of setpoint services.")
    add("")
    if neutral:
        add("Directionless set:")
        add("")
        for row in sorted(neutral, key=lambda r: r["key"])[:30]:
            command = row.get("command") or {}
            add(f"- `{row['key']}` — {command.get('cluster_name')}.{command.get('name')}")
        if len(neutral) > 30:
            add(f"- … and {len(neutral) - 30} more")
        add("")

    add("## Merge")
    add("")
    merge_rows = [
        [name, s.get("new", 0), s.get("approved_kept", 0), s.get("preserved_fields", 0),
         s.get("drift", 0), s.get("orphaned", 0)]
        for name, s in sorted(merge_stats.items())
    ]
    add(_table(["table", "new", "approved kept", "fields preserved", "drift", "orphaned"], merge_rows))
    add("")

    add("## Phase B readiness")
    add("")
    if total[tables.TODO] == 0:
        add("**READY** — no `todo` rows remain.")
    else:
        add(f"**NOT READY** — {total[tables.TODO]} `todo` row(s) must be resolved first.")
        add("")
        add("Unmet conditions:")
        add("")
        add(f"1. Resolve every `todo` listed above ({total[tables.TODO]} rows).")
        add(f"2. Confirm the {total[tables.REVIEW]} `review` rows and mark them `approved`.")
    add("")
    return "\n".join(lines) + "\n"
