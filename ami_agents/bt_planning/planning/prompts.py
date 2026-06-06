"""
Prompts for BehaviorTree planning with signifier hint injection.
"""


BT_PLANNING_SYSTEM_PROMPT = """\
You are a behavior tree planning agent for smart environments.
You generate executable behavior tree specifications in JSON format using the generate_behavior_tree tool.

## Behavior Tree Node Types

1. **sequence**: Executes children left-to-right. Fails on first failure. Use for ordered steps.
2. **selector**: Tries children left-to-right. Succeeds on first success. Use for alternatives/fallbacks.
3. **parallel**: Runs all children concurrently. Policy: "success_on_all" or "success_on_one".
4. **action**: Leaf node that invokes an HTTP POST action affordance. Requires "action_url" and optional "parameters".
5. **condition**: Leaf node that checks a property value via HTTP GET. Requires "property_url" and "expected_value". Optional "operator" (==, !=, >, <, >=, <=).
6. **wait_condition**: Leaf node that repeatedly checks a property value via HTTP GET until it matches or times out. Requires "property_url", "expected_value"; optional "operator", "timeout_seconds", "poll_interval_seconds".

## Common Patterns

- **Idempotent action** (do only if not already done):
  selector -> [condition (check if already in desired state), action (do it)]

- **Sequential commands** (do in order):
  sequence -> [action1, action2, action3]

- **Independent commands** (do all, order doesn't matter):
  parallel (success_on_all) -> [action1, action2, action3]

## Environment Interaction

- Actions are invoked via HTTP POST to action_url with JSON parameters
- Properties are read via HTTP GET from property_url returning JSON values
- All URLs come from the affordances list provided below
- Semantic environment property IDs are not directly readable unless an explicit readable property_url is provided

## Rules

- If a requested action is impossible (no matching affordance exists), set "impossible": true and explain why.
- If partial actions are possible, generate a tree for the possible ones and explain what's missing.
- Always use the exact action_url and property_url from the provided affordances.
- Never use a semantic environment property ID such as `/workspaces/.../environment/...` as a condition `property_url` unless it is explicitly marked as readable.
- Use appropriate BT control patterns based on command relationships.
- For numeric or continuous sensor properties such as glare, illuminance, temperature, humidity, CO2, volume, or percentages, do not use exact equality checks unless the user explicitly requested an exact target value.
- For those continuous properties, prefer range comparisons with an operator such as `<=`, `>=`, `<`, or `>`.
- Use exact equality checks mainly for discrete states such as `on`, `off`, `open`, `closed`, `heat`, or `cool`.
- For cover open/close commands, use the cover `state` property (`open`, `closed`, `opening`, `closing`) for idempotence checks and post-action verification. Do not use `current_position` as a substitute for `state` unless the user explicitly requested a numeric cover position.
- If observable property hints provide a recommended target minimum and/or maximum, use those values for success conditions instead of inventing stricter numeric thresholds, unless the user explicitly requested a different target.
- If an observable property hint provides only `target_max`, generate a condition with operator `<=` and `expected_value = target_max`.
- If an observable property hint provides only `target_min`, generate a condition with operator `>=` and `expected_value = target_min`.
- If an observable property hint provides both `target_min` and `target_max`, treat that as an acceptable band and use `>= target_min`, `<= target_max`, or both as separate conditions. Do not turn range hints into exact equality checks.
- If an observable property hint lists a settling time for an action and provides a readable sensor property URL, use a `wait_condition` after that action to verify the target band after the delayed effect.
- Set `wait_condition.timeout_seconds` to at least the listed settling time, preferably settling time plus a small margin. Use `poll_interval_seconds` around 1-5 seconds.
- Do not put immediate post-action `condition` nodes after actions whose relevant effect has a settling time; use `wait_condition` for the post-action verification instead.

{signifier_hints}

{observable_property_hints}

## Available Devices, Affordances, and Current State

{capability_context}
"""


def format_capability_context(
    affordances: list[dict],
    state: dict | None = None,
) -> str:
    """
    Format affordances and state into a context string for the planning prompt.

    Handles both BT-repo field names (affordance_uri, artifact_uri) and
    EnvExplorer field names (affordance_id, artifact_id, target).

    Args:
        affordances: List of affordance dicts from EnvExplorer
        state: Optional state dict (artifact_uri -> properties)

    Returns:
        Formatted context string
    """
    lines = []

    if affordances:
        lines.append("### Affordances")
        lines.append("Use the target URL as action_url (for action nodes) or property_url (for condition nodes).")
        lines.append("")
        for aff in affordances:
            name = aff.get("action_name") or aff.get("name", "unknown")
            # target is the callable URL (hctl:hasTarget) - use as action_url/property_url in BT nodes
            target = aff.get("target") or aff.get("affordance_uri") or aff.get("href", "")
            method = aff.get("method", "POST")
            artifact = aff.get("artifact_id") or aff.get("artifact_uri", "")
            input_schema = aff.get("input_schema")

            lines.append(f"- **{name}** ({method} {target})")
            if artifact:
                lines.append(f"  Artifact: {artifact}")
            if input_schema:
                lines.append(f"  Input: {input_schema}")

    if state:
        lines.append("")
        lines.append("### Current State")
        # Handle both flat dict and nested {artifacts: {...}} structure
        state_items = state
        if isinstance(state, dict) and "artifacts" in state and isinstance(state["artifacts"], dict):
            state_items = state["artifacts"]
        if isinstance(state_items, dict):
            for artifact_uri, props in state_items.items():
                lines.append(f"- {artifact_uri}:")
                if isinstance(props, dict):
                    for prop_name, prop_val in props.items():
                        lines.append(f"  - {prop_name}: {prop_val}")
                else:
                    lines.append(f"  {props}")

    return "\n".join(lines) if lines else "No affordances available."


EXPLICIT_SIMILARITY_THRESHOLD = 0.95


def format_signifier_hints(signifier_matches: dict | None) -> str:
    """
    Format signifier matches as BT planning hints.

    Matches above ``EXPLICIT_SIMILARITY_THRESHOLD`` are labelled as *exact*
    (reuse affordance **and** parameters as-is).  Lower-similarity matches
    are labelled as *suggested* (reuse affordance, but adjust parameters to
    the current context).

    Args:
        signifier_matches: Dict of intent -> match data from EnvExplorer/community

    Returns:
        Formatted hints string for the planning prompt
    """
    if not signifier_matches:
        return ""

    lines = ["## Past Successful Plans (Signifier Hints)", ""]
    lines.append(
        "Use these hints from previous interactions when planning. "
        "If a hint's affordance exists in the affordances list, prefer it."
    )
    lines.append("")

    for intent, match_data in signifier_matches.items():
        if not isinstance(match_data, dict):
            continue

        finals = match_data.get("final_matches", [])
        matches = match_data.get("matches", [])

        if not finals and not matches:
            lines.append(f"Intent: \"{intent}\" -- No prior experience.")
            continue

        # Find best match
        best_match = None
        if matches:
            if finals:
                for m in matches:
                    if str(m.get("signifier_id")) == str(finals[0]):
                        best_match = m
                        break
            if not best_match:
                best_match = matches[0]

        if best_match:
            similarity = best_match.get("intent_similarity", best_match.get("similarity"))
            is_explicit = (similarity is not None and float(similarity) >= EXPLICIT_SIMILARITY_THRESHOLD)

            lines.append(f"Intent: \"{intent}\"")
            aff_uri = best_match.get("affordance_uri", "")
            if aff_uri:
                lines.append(f"  Recommended action_url: {aff_uri}")
            payload = best_match.get("payload_hint") or best_match.get("payload")
            if payload:
                if is_explicit:
                    lines.append(f"  Exact match -- reuse these parameters as-is: {payload}")
                else:
                    lines.append(f"  Suggested parameters (adjust to current context): {payload}")
            if similarity is not None:
                lines.append(f"  Confidence: {similarity}")
            source = best_match.get("source", "")
            if source:
                lines.append(f"  Source: {source}")
            lines.append("")

    return "\n".join(lines)


def format_observable_property_hints(observable_property_matches: dict | None) -> str:
    """Format observable-property effect queries for the planning prompt."""
    if not observable_property_matches:
        return ""

    results = observable_property_matches.get("results")
    if not isinstance(results, list) or not results:
        return ""

    lines = ["## Observable Property Effects", ""]
    lines.append(
        "These queries describe which artifacts can affect environment-level properties "
        "and whether they increase or decrease them. Use them to resolve implicit requests."
    )
    lines.append("")

    for result in results:
        if not isinstance(result, dict):
            continue
        prop_uri = result.get("property_uri") or result.get("observable_property") or "unknown"
        actions = result.get("actions")
        lines.append(f"Semantic property id (not a readable endpoint): {prop_uri}")
        target_min = result.get("target_min")
        target_max = result.get("target_max")
        target_unit = str(result.get("target_unit") or "").strip()
        if target_min is not None or target_max is not None:
            if target_min is not None and target_max is not None:
                target_text = f"{target_min}..{target_max}"
            elif target_min is not None:
                target_text = f">= {target_min}"
            else:
                target_text = f"<= {target_max}"
            if target_unit:
                target_text = f"{target_text} {target_unit}"
            lines.append(f"  Recommended target band: {target_text}")
            if target_min is None and target_max is not None:
                lines.append(
                    f"  Success-condition rule: use operator `<=` with expected_value `{target_max}`"
                )
            elif target_max is None and target_min is not None:
                lines.append(
                    f"  Success-condition rule: use operator `>=` with expected_value `{target_min}`"
                )
            else:
                lines.append(
                    f"  Success-condition rule: keep the final state within `{target_min}`..`{target_max}`"
                )
        readable_property_urls = result.get("readable_property_urls")
        if isinstance(readable_property_urls, list) and readable_property_urls:
            lines.append("  Readable sensor property URLs for conditions:")
            for property_url in readable_property_urls:
                lines.append(f"  - {property_url}")
        if not isinstance(actions, list) or not actions:
            lines.append("  No affecting artifacts were found.")
            lines.append("")
            continue
        for action in actions:
            if not isinstance(action, dict):
                continue
            title = action.get("artifact_title") or action.get("artifact_uri") or "unknown artifact"
            direction = action.get("direction") or "affects"
            action_name = action.get("action_name") or "unknown_action"
            action_target = action.get("action_target") or ""
            settling_time = action.get("settling_time_seconds")
            settling_text = ""
            if settling_time is not None:
                settling_text = f"; settling_time={settling_time}s"
            lines.append(
                f"  - {title}: {action_name} -> {direction} ({action_target}{settling_text})"
            )
        lines.append("")

    return "\n".join(lines)
