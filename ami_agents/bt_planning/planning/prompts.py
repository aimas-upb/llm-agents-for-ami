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

## Rules

- If a requested action is impossible (no matching affordance exists), set "impossible": true and explain why.
- If partial actions are possible, generate a tree for the possible ones and explain what's missing.
- Always use the exact action_url and property_url from the provided affordances.
- Use appropriate BT control patterns based on command relationships.

{signifier_hints}

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
