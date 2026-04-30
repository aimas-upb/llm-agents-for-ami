from typing import Any, Dict, List, Optional


def generate_shacl_shapes_from_conditions(structured_conditions: List[Dict[str, Any]]) -> Optional[str]:
    """
    Generate SHACL shapes (Turtle format) from structured_conditions.

    Maps structured conditions to SHACL constraints for context validation:
    - equals → sh:hasValue (exact match)
    - greater_than → sh:minExclusive
    - greater_than_or_equal → sh:minInclusive
    - less_than → sh:maxExclusive
    - less_than_or_equal → sh:maxInclusive

    Args:
        structured_conditions: List of condition dicts with artifact, property, value_conditions

    Returns:
        SHACL shapes as Turtle string, or None if no valid conditions
    """
    if not structured_conditions or not isinstance(structured_conditions, list):
        return None

    # Group conditions by artifact to create one NodeShape per artifact
    artifact_conditions: Dict[str, List[Dict[str, Any]]] = {}
    for cond in structured_conditions:
        if not isinstance(cond, dict):
            continue
        artifact = cond.get("artifact")
        if not artifact:
            continue
        if artifact not in artifact_conditions:
            artifact_conditions[artifact] = []
        artifact_conditions[artifact].append(cond)

    if not artifact_conditions:
        return None

    # Build SHACL Turtle
    shapes_lines = [
        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
    ]

    for idx, (artifact_uri, conditions) in enumerate(artifact_conditions.items()):
        shape_id = f"<urn:signifier:shape:{idx}>"
        shapes_lines.append(f"{shape_id} a sh:NodeShape ;")
        shapes_lines.append(f"    sh:targetNode <{artifact_uri}> ;")

        # Add property constraints
        for prop_idx, cond in enumerate(conditions):
            prop_uri = cond.get("property_affordance")
            value_conditions = cond.get("value_conditions", [])
            if not prop_uri or not value_conditions:
                continue

            is_last_property = (prop_idx == len(conditions) - 1)
            shapes_lines.append("    sh:property [")
            shapes_lines.append(f"        sh:path <{prop_uri}> ;")

            # Process each value condition
            for vc_idx, vc in enumerate(value_conditions):
                if not isinstance(vc, dict):
                    continue
                operator = vc.get("operator", "equals")
                value = vc.get("value")
                if value is None:
                    continue

                # Determine datatype and format value
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                    datatype = "xsd:boolean"
                elif isinstance(value, str):
                    # Escape quotes
                    escaped = value.replace('"', '\\"')
                    value_str = f'"{escaped}"'
                    datatype = "xsd:string"
                elif isinstance(value, int):
                    value_str = str(value)
                    datatype = "xsd:integer"
                elif isinstance(value, float):
                    value_str = str(value)
                    datatype = "xsd:double"
                elif isinstance(value, list):
                    # For lists, we can't easily represent in SHACL - skip
                    continue
                elif isinstance(value, dict):
                    # For dicts, we can't easily represent in SHACL - skip
                    continue
                else:
                    escaped = str(value).replace('"', '\\"')
                    value_str = f'"{escaped}"'
                    datatype = "xsd:string"

                # Add datatype constraint (first, before value constraints)
                shapes_lines.append(f"        sh:datatype {datatype} ;")

                # Map operator to SHACL constraint
                is_last_vc = (vc_idx == len(value_conditions) - 1)
                if operator == "equals":
                    shapes_lines.append(f"        sh:hasValue {value_str}{';' if not is_last_vc else ''}")
                elif operator == "greater_than":
                    shapes_lines.append(f"        sh:minExclusive {value_str}{';' if not is_last_vc else ''}")
                elif operator == "greater_than_or_equal":
                    shapes_lines.append(f"        sh:minInclusive {value_str}{';' if not is_last_vc else ''}")
                elif operator == "less_than":
                    shapes_lines.append(f"        sh:maxExclusive {value_str}{';' if not is_last_vc else ''}")
                elif operator == "less_than_or_equal":
                    shapes_lines.append(f"        sh:maxInclusive {value_str}{';' if not is_last_vc else ''}")
                # For not_equals, we could use sh:not but it's complex - skip for now

            shapes_lines.append(f"    ]{';' if not is_last_property else '.'}")

        shapes_lines.append("")

    return "\n".join(shapes_lines)