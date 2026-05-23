#!/usr/bin/env python3
"""Augment a results CSV with structural plan-quality metrics and scores.

Usage:
    python tests/e2e-lab308e/score_plans.py tests/e2e-lab308e/results.csv
    python tests/e2e-lab308e/score_plans.py results.csv --output results.scored.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ARTIFACT_RE = re.compile(r"/artifacts/([^/#?]+)")
CASE_DIR = Path(__file__).with_name("cases")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _artifact_name_from_url(url: str) -> str:
    match = ARTIFACT_RE.search(str(url or ""))
    return match.group(1) if match else ""


def _is_property_url(url: str) -> bool:
    return "/properties/" in str(url or "")


def _is_action_url(url: str) -> bool:
    text = str(url or "")
    return bool(text) and "/properties/" not in text


def _looks_like_sensor_artifact(artifact_name: str) -> bool:
    name = (artifact_name or "").lower()
    keywords = (
        "sensing",
        "sensor",
        "temperature",
        "humidity",
        "co2",
        "glare",
        "light",
        "illumin",
        "presence",
        "person_counter",
    )
    return any(tok in name for tok in keywords)


def _looks_like_continuous_condition(node: Dict[str, Any]) -> bool:
    expected = _safe_float(node.get("expected_value"))
    if expected is None:
        return False
    property_url = str(node.get("property_url") or "")
    artifact_name = _artifact_name_from_url(property_url)
    if property_url.endswith("/properties/current_position"):
        return False
    if property_url.endswith("/properties/temperature"):
        return True
    return _looks_like_sensor_artifact(artifact_name)


def _artifact_properties(artifact_name: str) -> set[str]:
    name = (artifact_name or "").lower()
    props: set[str] = set()
    if any(tok in name for tok in ("glare_sensing", "glare")):
        props.add("glare")
    if any(tok in name for tok in ("co2_sensing", "co2")):
        props.add("air_quality")
    if any(tok in name for tok in ("humidity_sensing", "humidity")):
        props.add("humidity")
    if any(tok in name for tok in ("temperature_sensing", "temperature")):
        props.add("thermal_comfort")
    if any(tok in name for tok in ("internal_light_sensing", "external_light_sensing", "ambient_lights", "task_lights", "blinds", "window_308e_cover")):
        props.add("luminosity")
    if "desk_light_sensing" in name or "desk_lamp" in name:
        props.update({"luminosity", "desk_luminosity"})
    return props


def _load_case_expectations() -> Dict[str, Dict[str, Any]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    for path in sorted(CASE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = str(data.get("name") or "").strip()
        if not name:
            continue
        loaded[name] = dict(data.get("semantic_expectations") or {})
    return loaded


def _flatten_tree(
    node: Dict[str, Any],
    parent_type: Optional[str] = None,
    depth: int = 1,
    selectors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    selectors = selectors or []
    nodes = 1
    actions: List[Dict[str, Any]] = []
    conditions: List[Dict[str, Any]] = []
    node_types = [str(node.get("type") or "").lower()]
    max_depth = depth
    if str(node.get("type") or "").lower() == "action":
        actions.append(node)
    elif str(node.get("type") or "").lower() == "condition":
        conditions.append(node)

    children = node.get("children") or []
    if isinstance(children, list):
        for child in children:
            if not isinstance(child, dict):
                continue
            child_flat = _flatten_tree(
                child,
                parent_type=str(node.get("type") or "").lower(),
                depth=depth + 1,
                selectors=selectors,
            )
            nodes += child_flat["nodes"]
            actions.extend(child_flat["actions"])
            conditions.extend(child_flat["conditions"])
            node_types.extend(child_flat["node_types"])
            max_depth = max(max_depth, child_flat["max_depth"])

    if str(node.get("type") or "").lower() == "selector" and isinstance(children, list):
        selectors.append(node)

    return {
        "nodes": nodes,
        "actions": actions,
        "conditions": conditions,
        "node_types": node_types,
        "max_depth": max_depth,
        "selectors": selectors,
    }


def _guarded_action_keys(tree: Dict[str, Any]) -> set[Tuple[str, str]]:
    guarded: set[Tuple[str, str]] = set()

    def walk(node: Dict[str, Any]) -> None:
        node_type = str(node.get("type") or "").lower()
        children = node.get("children") or []
        if node_type == "selector" and isinstance(children, list):
            conditions = [c for c in children if isinstance(c, dict) and str(c.get("type") or "").lower() == "condition"]
            actions = [c for c in children if isinstance(c, dict) and str(c.get("type") or "").lower() == "action"]
            for cond in conditions:
                cond_artifact = _artifact_name_from_url(str(cond.get("property_url") or ""))
                for action in actions:
                    action_artifact = _artifact_name_from_url(str(action.get("action_url") or ""))
                    if cond_artifact and cond_artifact == action_artifact:
                        guarded.add((action_artifact, str(action.get("name") or "")))
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    walk(child)

    walk(tree)
    return guarded


def _score_complexity(node_count: int, max_depth: int) -> float:
    if node_count <= 0:
        return 0.0
    node_penalty = max(0.0, (node_count - 18) / 18.0)
    depth_penalty = max(0.0, (max_depth - 6) / 6.0)
    return _clamp01(1.0 - 0.5 * node_penalty - 0.5 * depth_penalty)


def _score_plan(plan_text: str, expectations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "plan_parse_ok": 0,
        "plan_has_tree": 0,
        "plan_has_explanation": 0,
        "plan_nodes_total": 0,
        "action_count": 0,
        "condition_count": 0,
        "selector_count": 0,
        "sequence_count": 0,
        "parallel_count": 0,
        "unique_action_artifact_count": 0,
        "max_depth": 0,
        "numeric_condition_count": 0,
        "exact_numeric_condition_count": 0,
        "invalid_condition_url_count": 0,
        "invalid_action_url_count": 0,
        "guarded_action_count": 0,
        "action_guard_coverage": 0.0,
        "score_parse": 0.0,
        "score_explanation": 0.0,
        "score_url_validity": 0.0,
        "score_numeric_operator_quality": 0.0,
        "score_guard_coverage": 0.0,
        "score_complexity": 0.0,
        "structural_quality_avg": 0.0,
        "semantic_case_found": 0,
        "semantic_target_property_count": 0,
        "semantic_matched_property_count": 0,
        "semantic_preferred_artifact_count": 0,
        "semantic_acceptable_artifact_count": 0,
        "semantic_forbidden_artifact_count": 0,
        "semantic_preferred_action_count": 0,
        "semantic_acceptable_action_count": 0,
        "semantic_forbidden_action_count": 0,
        "score_semantic_property_match": 0.0,
        "score_semantic_artifact_relevance": 0.0,
        "score_semantic_preferred_coverage": 0.0,
        "score_semantic_forbidden_safety": 0.0,
        "score_semantic_threshold_discipline": 0.0,
        "semantic_quality_avg": 0.0,
        "plan_quality_avg": 0.0,
    }
    if not plan_text:
        return metrics

    try:
        plan_obj = json.loads(plan_text)
    except Exception:
        return metrics

    metrics["plan_parse_ok"] = 1
    metrics["score_parse"] = 1.0
    metrics["plan_has_explanation"] = 1 if str(plan_obj.get("explanation") or "").strip() else 0
    metrics["score_explanation"] = float(metrics["plan_has_explanation"])

    tree = plan_obj.get("tree")
    if not isinstance(tree, dict):
        score_cols = [
            "score_parse",
            "score_explanation",
            "score_url_validity",
            "score_numeric_operator_quality",
            "score_guard_coverage",
            "score_complexity",
        ]
        metrics["structural_quality_avg"] = round(
            sum(float(metrics[c]) for c in score_cols) / len(score_cols), 4
        )
        metrics["plan_quality_avg"] = metrics["structural_quality_avg"]
        return metrics

    metrics["plan_has_tree"] = 1
    flat = _flatten_tree(tree)
    actions = flat["actions"]
    conditions = flat["conditions"]
    node_types = flat["node_types"]
    metrics["plan_nodes_total"] = flat["nodes"]
    metrics["action_count"] = len(actions)
    metrics["condition_count"] = len(conditions)
    metrics["selector_count"] = sum(1 for t in node_types if t == "selector")
    metrics["sequence_count"] = sum(1 for t in node_types if t == "sequence")
    metrics["parallel_count"] = sum(1 for t in node_types if t == "parallel")
    metrics["max_depth"] = flat["max_depth"]
    metrics["unique_action_artifact_count"] = len(
        {name for name in (_artifact_name_from_url(a.get("action_url", "")) for a in actions) if name}
    )

    invalid_action_urls = 0
    for action in actions:
        if not _is_action_url(str(action.get("action_url") or "")):
            invalid_action_urls += 1
    metrics["invalid_action_url_count"] = invalid_action_urls

    invalid_condition_urls = 0
    numeric_conditions = 0
    exact_numeric_conditions = 0
    for cond in conditions:
        property_url = str(cond.get("property_url") or "")
        if not _is_property_url(property_url):
            invalid_condition_urls += 1
        if _looks_like_continuous_condition(cond):
            numeric_conditions += 1
            if str(cond.get("operator") or "").strip() == "==":
                exact_numeric_conditions += 1

    metrics["invalid_condition_url_count"] = invalid_condition_urls
    metrics["numeric_condition_count"] = numeric_conditions
    metrics["exact_numeric_condition_count"] = exact_numeric_conditions

    guarded_keys = _guarded_action_keys(tree)
    guarded_actions = 0
    for action in actions:
        artifact = _artifact_name_from_url(str(action.get("action_url") or ""))
        name = str(action.get("name") or "")
        if (artifact, name) in guarded_keys:
            guarded_actions += 1
    metrics["guarded_action_count"] = guarded_actions
    metrics["action_guard_coverage"] = round(
        guarded_actions / len(actions), 4
    ) if actions else 0.0

    total_urls = len(actions) + len(conditions)
    invalid_urls = invalid_action_urls + invalid_condition_urls
    metrics["score_url_validity"] = round(
        1.0 if total_urls == 0 else _clamp01(1.0 - (invalid_urls / total_urls)), 4
    )
    metrics["score_numeric_operator_quality"] = round(
        1.0
        if numeric_conditions == 0
        else _clamp01(1.0 - (exact_numeric_conditions / numeric_conditions)),
        4,
    )
    metrics["score_guard_coverage"] = round(
        1.0 if not actions else metrics["action_guard_coverage"], 4
    )
    metrics["score_complexity"] = round(
        _score_complexity(metrics["plan_nodes_total"], metrics["max_depth"]), 4
    )

    structural_score_cols = [
        "score_parse",
        "score_explanation",
        "score_url_validity",
        "score_numeric_operator_quality",
        "score_guard_coverage",
        "score_complexity",
    ]
    metrics["structural_quality_avg"] = round(
        sum(float(metrics[c]) for c in structural_score_cols) / len(structural_score_cols), 4
    )

    expectations = expectations or {}
    if expectations:
        metrics["semantic_case_found"] = 1
        target_properties = {str(p).strip().lower() for p in expectations.get("target_properties") or [] if str(p).strip()}
        preferred_artifacts = {str(a).strip() for a in expectations.get("preferred_artifacts") or [] if str(a).strip()}
        acceptable_artifacts = {str(a).strip() for a in expectations.get("acceptable_artifacts") or [] if str(a).strip()}
        forbidden_artifacts = {str(a).strip() for a in expectations.get("forbidden_artifacts") or [] if str(a).strip()}
        must_avoid_exact = bool(expectations.get("must_avoid_exact_numeric_thresholds"))

        metrics["semantic_target_property_count"] = len(target_properties)
        metrics["semantic_preferred_artifact_count"] = len(preferred_artifacts)
        metrics["semantic_acceptable_artifact_count"] = len(acceptable_artifacts)
        metrics["semantic_forbidden_artifact_count"] = len(forbidden_artifacts)

        action_artifacts = [
            _artifact_name_from_url(str(action.get("action_url") or ""))
            for action in actions
        ]
        condition_artifacts = [
            _artifact_name_from_url(str(cond.get("property_url") or ""))
            for cond in conditions
        ]

        matched_properties: set[str] = set()
        for artifact_name in condition_artifacts:
            matched_properties.update(_artifact_properties(artifact_name) & target_properties)
        metrics["semantic_matched_property_count"] = len(matched_properties)

        preferred_action_count = sum(1 for a in action_artifacts if a in preferred_artifacts)
        acceptable_action_count = sum(1 for a in action_artifacts if a in acceptable_artifacts)
        forbidden_action_count = sum(1 for a in action_artifacts if a in forbidden_artifacts)
        metrics["semantic_preferred_action_count"] = preferred_action_count
        metrics["semantic_acceptable_action_count"] = acceptable_action_count
        metrics["semantic_forbidden_action_count"] = forbidden_action_count

        metrics["score_semantic_property_match"] = round(
            1.0 if not target_properties else len(matched_properties) / len(target_properties), 4
        )
        metrics["score_semantic_artifact_relevance"] = round(
            1.0 if not actions else acceptable_action_count / len(actions), 4
        )
        metrics["score_semantic_preferred_coverage"] = round(
            1.0 if not preferred_artifacts else min(1.0, preferred_action_count / len(preferred_artifacts)), 4
        )
        metrics["score_semantic_forbidden_safety"] = round(
            1.0 if not actions else _clamp01(1.0 - (forbidden_action_count / len(actions))), 4
        )
        metrics["score_semantic_threshold_discipline"] = round(
            1.0
            if not must_avoid_exact or metrics["numeric_condition_count"] == 0
            else _clamp01(1.0 - (metrics["exact_numeric_condition_count"] / max(1, metrics["numeric_condition_count"]))),
            4,
        )

    semantic_score_cols = [
        "score_semantic_property_match",
        "score_semantic_artifact_relevance",
        "score_semantic_preferred_coverage",
        "score_semantic_forbidden_safety",
        "score_semantic_threshold_discipline",
    ]
    metrics["semantic_quality_avg"] = round(
        sum(float(metrics[c]) for c in semantic_score_cols) / len(semantic_score_cols), 4
    )
    metrics["plan_quality_avg"] = round(
        (metrics["structural_quality_avg"] + metrics["semantic_quality_avg"]) / 2.0,
        4,
    )
    return metrics


def _iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    suffix = input_path.suffix or ".csv"
    return input_path.with_name(f"{stem}.scored{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", help="Path to an input results.csv file")
    parser.add_argument(
        "--output",
        help="Output path for the augmented CSV (default: <input>.scored.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.results_csv)
    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    case_expectations = _load_case_expectations()

    rows = list(_iter_rows(input_path))
    if not rows:
        raise SystemExit(f"No rows found in {input_path}")

    augmented_rows: List[Dict[str, Any]] = []
    metric_keys: List[str] = []
    for row in rows:
        metrics = _score_plan(
            row.get("plan") or "",
            case_expectations.get(str(row.get("test_name") or "").strip(), {}),
        )
        if not metric_keys:
            metric_keys = list(metrics.keys())
        augmented = dict(row)
        augmented.update(metrics)
        augmented_rows.append(augmented)

    fieldnames = list(rows[0].keys()) + metric_keys
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(augmented_rows)

    avg = sum(float(r["plan_quality_avg"]) for r in augmented_rows) / len(augmented_rows)
    print(f"Wrote {len(augmented_rows)} rows to {output_path}")
    print(f"Average plan_quality_avg: {avg:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
