#!/usr/bin/env python3
"""
Bootstrap: vendor a pinned, offline Matter data-model registry.

Fetches the CSA data model ONCE from connectedhomeip, freezes it into the repo
under ami_agents/environment/integration/SimuHome/matter_model/vendor/<VER>/, and emits compact JSON lookup
tables. After this runs, conversion needs no network and no Matter fact (device
type, cluster, attribute, command, event) has to come from model memory.

Usage:
  python ami_agents/environment/integration/SimuHome/matter_model/bootstrap.py
  python ami_agents/environment/integration/SimuHome/matter_model/bootstrap.py --tag v1.5.1.0 --spec-version 1.5.1
  python ami_agents/environment/integration/SimuHome/matter_model/bootstrap.py --validate-only

Steps:
  1. Resolve release tags from the remote (never a moving branch) and pick the
     latest stable; record the tag and the resolved commit SHA.
  2. Sparse, shallow, blobless clone of data_model/ only.
  3. Vendor data_model/<VER>/ into matter_model/vendor/<VER>/.
  4. Normalize to JSON via matter_model/normalize_model.py.
  5. Validation gate; print a table and stop.

Airgapped hosts: run the clone elsewhere, copy the folder in, then re-run with
--validate-only. The manifest sha256 proves the version.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MATTER_MODEL_DIR = Path(__file__).resolve().parent
NORMALIZER = MATTER_MODEL_DIR / "normalize_model.py"
VENDOR_ROOT = MATTER_MODEL_DIR / "vendor"
SOURCE_REPO = "https://github.com/project-chip/connectedhomeip.git"

# Tags that exist but are not stable releases.
_PRERELEASE = re.compile(r"(alpha|beta|rc|sve|test|harness|dev)", re.IGNORECASE)
_RELEASE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?$")


def _run(cmd: List[str], *, cwd: Optional[Path] = None, timeout: float = 900.0) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def _tag_sort_key(tag: str) -> Tuple[int, ...]:
    match = _RELEASE_TAG.match(tag)
    if not match:
        return (-1,)
    return tuple(int(part) if part else 0 for part in match.groups())


def resolve_latest_tag() -> Tuple[str, str]:
    """List remote tags and return (tag, commit_sha) for the latest stable release.

    Tags are resolved from the remote rather than assumed to exist.
    """
    output = _run(["git", "ls-remote", "--tags", "--refs", SOURCE_REPO, "v*"], timeout=180.0)
    candidates: Dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        sha, ref = parts
        tag = ref.rsplit("/", 1)[-1]
        if _PRERELEASE.search(tag) or not _RELEASE_TAG.match(tag):
            continue
        candidates[tag] = sha
    if not candidates:
        raise RuntimeError("No stable release tags found on the remote")
    tag = max(candidates, key=_tag_sort_key)
    return tag, candidates[tag]


def fetch_and_vendor(tag: str, spec_version: Optional[str], keep_tmp: bool) -> Tuple[str, str, Path]:
    """Sparse/shallow clone, pick the spec dir, copy it into vendor/. Returns
    (commit_sha, spec_version, vendored_dir)."""
    tmp_dir = MATTER_MODEL_DIR / "_chip_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"[bootstrap] Cloning {SOURCE_REPO} @ {tag} (shallow, blobless, sparse)", flush=True)
    _run(
        [
            "git", "clone",
            "--depth", "1",
            "--filter=blob:none",
            "--sparse",
            "--branch", tag,
            SOURCE_REPO,
            str(tmp_dir),
        ]
    )
    try:
        commit_sha = _run(["git", "rev-parse", "HEAD"], cwd=tmp_dir).strip()
        _run(["git", "sparse-checkout", "set", "data_model"], cwd=tmp_dir)

        data_model = tmp_dir / "data_model"
        if not data_model.is_dir():
            raise RuntimeError(f"data_model/ not present at tag {tag}")

        available = sorted(
            (p.name for p in data_model.iterdir() if p.is_dir()),
            key=lambda name: [int(x) if x.isdigit() else -1 for x in name.split(".")],
        )
        if not available:
            raise RuntimeError(f"No spec-version directories under data_model/ at {tag}")
        print(f"[bootstrap] spec-version dirs present: {', '.join(available)}", flush=True)

        version = spec_version or available[-1]
        if version not in available:
            raise RuntimeError(f"Requested spec version {version!r} not in {available}")

        source_dir = data_model / version
        for required in ("clusters", "device_types"):
            if not (source_dir / required).is_dir():
                raise RuntimeError(f"data_model/{version}/{required}/ missing")

        vendored_dir = VENDOR_ROOT / version
        if vendored_dir.exists():
            shutil.rmtree(vendored_dir)
        vendored_dir.mkdir(parents=True, exist_ok=True)
        for required in ("clusters", "device_types"):
            shutil.copytree(source_dir / required, vendored_dir / required)
        # Provenance files the scraper leaves alongside the XML, when present.
        for extra in ("spec_sha", "spec_tag", "scraper_version"):
            candidate = source_dir / extra
            if candidate.is_file():
                shutil.copy2(candidate, vendored_dir / extra)

        print(
            f"[bootstrap] Vendored data_model/{version} -> {vendored_dir} "
            f"(commit {commit_sha[:12]})",
            flush=True,
        )
        return commit_sha, version, vendored_dir
    finally:
        if not keep_tmp and tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def normalize(vendored_dir: Path, tag: str, commit_sha: str, version: str) -> None:
    print("[bootstrap] Normalizing XML -> JSON", flush=True)
    output = _run(
        [
            sys.executable, str(NORMALIZER), str(vendored_dir),
            "--source", SOURCE_REPO,
            "--tag", tag,
            "--commit", commit_sha,
            "--spec-version", version,
        ],
        timeout=600.0,
    )
    print(output.rstrip(), flush=True)


# --- Validation gate -------------------------------------------------------

EXPECTED_DEVICE_TYPES = {45: "Air Purifier"}
EXPECTED_CLUSTERS = {
    29: "Descriptor",
    3: "Identify",
    6: "On/Off",
    514: "Fan Control",
}
EXPECTED_FAN_ATTRS = {"PercentSetting", "PercentCurrent"}


def validate(vendored_dir: Path) -> int:
    clusters_path = vendored_dir / "clusters.json"
    device_types_path = vendored_dir / "device-types.json"
    manifest_path = vendored_dir / "manifest.json"
    for path in (clusters_path, device_types_path, manifest_path):
        if not path.is_file():
            print(f"FAIL: missing generated file {path}", file=sys.stderr)
            return 1

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    device_types = json.loads(device_types_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows: List[Tuple[str, str, str, str]] = []
    failures: List[str] = []

    def check(label: str, expected: str, actual: str) -> None:
        ok = expected == actual
        rows.append(("PASS" if ok else "FAIL", label, expected, actual))
        if not ok:
            failures.append(f"{label}: expected {expected!r}, got {actual!r}")

    for type_id, expected_name in sorted(EXPECTED_DEVICE_TYPES.items()):
        record = device_types.get(str(type_id))
        actual = record["name"] if record else "<missing>"
        check(f"deviceType {type_id} (0x{type_id:04X})", expected_name, actual)

    for cluster_id, expected_name in sorted(EXPECTED_CLUSTERS.items()):
        record = clusters.get(str(cluster_id))
        actual = record["name"] if record else "<missing>"
        # The XML titles clusters "Fan Control Cluster"; accept the spec name
        # with or without the trailing "Cluster" noise word.
        normalized = re.sub(r"\s+Cluster$", "", actual) if record else actual
        check(f"cluster {cluster_id} (0x{cluster_id:04X})", expected_name, normalized)

    fan = clusters.get("514")
    fan_attrs = {a["name"] for a in fan["attributes"]} if fan else set()
    missing_attrs = EXPECTED_FAN_ATTRS - fan_attrs
    check(
        "cluster 514 attributes",
        ", ".join(sorted(EXPECTED_FAN_ATTRS)),
        ", ".join(sorted(EXPECTED_FAN_ATTRS - missing_attrs)) or "<missing>",
    )

    # Every parsed decimal id must round-trip to its recorded hex form.
    roundtrip_bad: List[str] = []
    for table_name, table, width in (("cluster", clusters, 4), ("deviceType", device_types, 4)):
        for key, record in table.items():
            if f"0x{int(key):0{width}X}" != record["id_hex"]:
                roundtrip_bad.append(f"{table_name} {key} -> {record['id_hex']}")
            for field, sub_width in (("attributes", 4), ("commands", 2), ("events", 4)):
                for item in record.get(field, []) or []:
                    if f"0x{item['id']:0{sub_width}X}" != item["id_hex"]:
                        roundtrip_bad.append(f"{table_name} {key}.{field} {item['id']} -> {item['id_hex']}")
    check(
        "decimal/hex round-trip",
        "all ids consistent",
        "all ids consistent" if not roundtrip_bad else f"{len(roundtrip_bad)} mismatched",
    )
    if roundtrip_bad:
        failures.extend(roundtrip_bad[:10])

    width_label = max(len(r[1]) for r in rows)
    width_exp = max(len(r[2]) for r in rows)
    print()
    print(f"{'':4}  {'CHECK'.ljust(width_label)}  {'EXPECTED'.ljust(width_exp)}  ACTUAL")
    print(f"{'-' * 4}  {'-' * width_label}  {'-' * width_exp}  {'-' * 20}")
    for status, label, expected, actual in rows:
        print(f"{status:4}  {label.ljust(width_label)}  {expected.ljust(width_exp)}  {actual}")
    print()
    print(
        f"tag={manifest.get('tag')} commit={str(manifest.get('commit_sha'))[:12]} "
        f"spec_version={manifest.get('spec_version')} "
        f"clusters={manifest.get('cluster_count')} device_types={manifest.get('device_type_count')} "
        f"files_hashed={len(manifest.get('sha256_by_file') or {})}"
    )

    if failures:
        print(f"\nVALIDATION FAILED ({len(failures)}):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\nVALIDATION PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Vendor a pinned, offline Matter data-model registry.")
    parser.add_argument("--tag", help="Release tag to pin; default: latest stable resolved from the remote")
    parser.add_argument("--spec-version", help="data_model/<VER> subdir; default: highest present")
    parser.add_argument("--validate-only", action="store_true", help="Skip network; validate an existing vendored tree")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep the _chip_tmp clone for inspection")
    args = parser.parse_args()

    if args.validate_only:
        if args.spec_version:
            vendored_dir = VENDOR_ROOT / args.spec_version
        else:
            candidates = sorted(p for p in VENDOR_ROOT.glob("*") if p.is_dir()) if VENDOR_ROOT.is_dir() else []
            if not candidates:
                print(f"No vendored tree under {VENDOR_ROOT}", file=sys.stderr)
                return 2
            vendored_dir = candidates[-1]
        return validate(vendored_dir)

    if args.tag:
        tag = args.tag
        listing = _run(["git", "ls-remote", "--tags", "--refs", SOURCE_REPO, tag], timeout=180.0)
        line = next((l for l in listing.splitlines() if l.strip()), "")
        if not line:
            print(f"Tag not found on remote: {tag}", file=sys.stderr)
            return 2
        remote_sha = line.split()[0]
        print(f"[bootstrap] Pinned tag {tag} -> {remote_sha}", flush=True)
    else:
        tag, remote_sha = resolve_latest_tag()
        print(f"[bootstrap] Latest stable tag {tag} -> {remote_sha}", flush=True)

    commit_sha, version, vendored_dir = fetch_and_vendor(tag, args.spec_version, args.keep_tmp)
    if commit_sha != remote_sha:
        print(
            f"[bootstrap] note: clone HEAD {commit_sha[:12]} differs from ls-remote {remote_sha[:12]} "
            "(annotated tag object vs commit); recording clone HEAD",
            flush=True,
        )
    normalize(vendored_dir, tag, commit_sha, version)
    return validate(vendored_dir)


if __name__ == "__main__":
    raise SystemExit(main())
