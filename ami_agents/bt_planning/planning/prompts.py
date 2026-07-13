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
4. **action**: Leaf node that invokes an action affordance. Requires "affordance_id" (an [action] id from the affordances list) and optional "parameters".
5. **condition**: Leaf node that checks a property value. Requires "affordance_id" (a [property] id from the affordances list, or a readable property id/URL from the hints) and "expected_value". Optional "operator" (==, !=, >, <, >=, <=).
6. **wait_condition**: Leaf node that repeatedly checks a property value until it matches or times out. Requires "affordance_id", "expected_value"; optional "operator", "timeout_seconds", "poll_interval_seconds".

## Common Patterns

- **Idempotent action** (do only if not already done):
  selector -> [condition (check if already in desired state), action (do it)]

- **Sequential commands** (do in order):
  sequence -> [action1, action2, action3]

- **Independent commands** (do all, order doesn't matter):
  parallel (success_on_all) -> [action1, action2, action3]

## Environment Interaction

- Affordances are referenced by their id (e.g. `light308/setBrightness`); the runtime resolves ids to HTTP endpoints.
- Action affordances are invoked with JSON parameters; property affordances are read and return JSON values.
- All affordance ids come from the affordances list provided below.
- For condition nodes you may also use a readable property id/URL taken verbatim from the observable-property hints.
- Semantic environment property IDs are not directly readable unless an explicit readable property id or URL is provided

## Rules

- If a requested action is impossible (no matching affordance exists), set "impossible": true and explain why.
- If partial actions are possible, generate a tree for the possible ones and explain what's missing.
- Always use exact affordance ids from the provided affordances list (or readable property ids/URLs from the hints). Never invent ids or URLs.
- Never use a semantic environment property ID such as `/workspaces/.../environment/...` as a condition `affordance_id` unless it is explicitly marked as readable.
- Use appropriate BT control patterns based on command relationships.
- Never start a sequence with an equality condition on the current value of a property before an action: if the value differs, the action never runs. Use the selector idempotent pattern or a range operator instead.
- For numeric or continuous sensor properties such as glare, illuminance, temperature, humidity, CO2, volume, or percentages, do not use exact equality checks unless the user explicitly requested an exact target value.
- For those continuous properties, prefer range comparisons with an operator such as `<=`, `>=`, `<`, or `>`.
- Use exact equality checks mainly for discrete states such as `on`, `off`, `open`, `closed`, `heat`, or `cool`.
- For cover open/close commands, use the cover `state` property (`open`, `closed`, `opening`, `closing`) for idempotence checks and post-action verification. Do not use `current_position` as a substitute for `state` unless the user explicitly requested a numeric cover position.
- If observable property hints provide a recommended target minimum and/or maximum, use those values for success conditions instead of inventing stricter numeric thresholds, unless the user explicitly requested a different target.
- If an observable property hint provides only `target_max`, generate a condition with operator `<=` and `expected_value = target_max`.
- If an observable property hint provides only `target_min`, generate a condition with operator `>=` and `expected_value = target_min`.
- If an observable property hint provides both `target_min` and `target_max`, treat that as an acceptable band and use `>= target_min`, `<= target_max`, or both as separate conditions. Do not turn range hints into exact equality checks.
- If an observable property hint lists a settling time for an action and provides a readable sensor property id/URL, use a `wait_condition` after that action to verify the target band after the delayed effect.
- Set `wait_condition.timeout_seconds` to at least the listed settling time, preferably settling time plus a small margin. Use `poll_interval_seconds` around 1-5 seconds.
- Do not put immediate post-action `condition` nodes after actions whose relevant effect has a settling time; use `wait_condition` for the post-action verification instead.

{signifier_hints}

{observable_property_hints}

## Available Devices, Affordances, and Current State

{capability_context}
"""


def _affordance_target(aff: dict) -> str:
    """Return the callable URL (hctl:hasTarget) of an affordance dict."""
    return str(aff.get("target") or aff.get("affordance_uri") or aff.get("href") or "")


def _artifact_short_name(aff: dict) -> str:
    """Derive a short artifact label from an affordance dict."""
    name = aff.get("artifact_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    artifact = str(aff.get("artifact_id") or aff.get("artifact_uri") or "")
    artifact = artifact.split("#")[0].rstrip("/")
    return artifact.rsplit("/", 1)[-1] if artifact else "env"


def build_affordance_index(affordances: list[dict]) -> dict[str, dict]:
    """
    Assign short, deterministic ids (``artifact/name``) to affordances.

    The ids are shown to the LLM instead of URLs/schemas and resolved back to
    target URLs by the planner after generation. Collisions get a ``-N`` suffix.
    """
    index: dict[str, dict] = {}
    for aff in affordances or []:
        if not isinstance(aff, dict):
            continue
        name = aff.get("action_name") or aff.get("name") or "affordance"
        base_ref = f"{_artifact_short_name(aff)}/{name}"
        ref = base_ref
        counter = 2
        while ref in index:
            ref = f"{base_ref}-{counter}"
            counter += 1
        index[ref] = aff
    return index


def build_url_to_ref(index: dict[str, dict]) -> dict[str, str]:
    """Map normalized affordance target URLs back to their short ids."""
    url_to_ref: dict[str, str] = {}
    for ref, aff in index.items():
        target = _affordance_target(aff).rstrip("/")
        if target and target not in url_to_ref:
            url_to_ref[target] = ref
    return url_to_ref


def _short_type(param_type) -> str:
    """Shorten schema type IRIs like ``...json-schema#NumberSchema`` to ``number``."""
    text = str(param_type or "")
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    if text.endswith("Schema"):
        text = text[: -len("Schema")]
    return text.lower()


def _schema_param_summary(schema) -> str:
    """Summarise a JSON Schema as compact ``name:type*`` parameter labels."""
    if not isinstance(schema, dict):
        return ""
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return ""
    required = set(schema.get("required") or [])
    parts = []
    for key, sub in props.items():
        param_type = _short_type(sub.get("type")) if isinstance(sub, dict) else ""
        label = f"{key}:{param_type}" if param_type else str(key)
        if key in required:
            label += "*"
        parts.append(label)
    return ", ".join(parts)


def format_capability_context(
    affordances: list[dict],
    state: dict | None = None,
    index: dict[str, dict] | None = None,
) -> str:
    """
    Format affordances and state into a compact context string for the
    planning prompt: short affordance ids, names, descriptions, and parameter
    names only (no URLs or full JSON Schemas).

    Args:
        affordances: List of affordance dicts from EnvExplorer
        state: Optional state dict (artifact_uri -> properties)
        index: Optional precomputed id index from ``build_affordance_index``

    Returns:
        Formatted context string
    """
    lines = []

    if index is None:
        index = build_affordance_index(affordances)

    # Only list property affordances of artifacts that also expose actions;
    # sensor-only artifacts would flood the prompt (hundreds in large homes)
    # and their readable properties arrive via observable-property hints.
    # The full index still resolves every ref, listed or not.
    actionable_artifacts = {
        str(aff.get("artifact_id") or aff.get("artifact_uri") or "")
        for aff in index.values()
        if str(aff.get("affordance_type") or "action") == "action"
    }

    if index:
        lines.append("### Affordances")
        lines.append(
            "Reference these by exact `affordance_id` in action, condition, and wait_condition nodes."
        )
        lines.append("")
        for ref, aff in index.items():
            aff_type = str(aff.get("affordance_type") or "action")
            if aff_type != "action":
                artifact = str(aff.get("artifact_id") or aff.get("artifact_uri") or "")
                if artifact not in actionable_artifacts:
                    continue
            entry = f"- {ref} [{aff_type}]"
            params = _schema_param_summary(aff.get("input_schema"))
            if params:
                entry += f" (params: {params})"
            name = str(aff.get("action_name") or aff.get("name") or "")
            description = str(aff.get("description") or "").strip()
            generic = {
                name.lower(),
                f"property affordance: {name.lower()}",
                f"action affordance: {name.lower()}",
            }
            if description and description.lower() not in generic:
                entry += f" -- {description}"
            lines.append(entry)

    if state:
        lines.append("")
        lines.append("### Current State")
        lines.append("(short ids; use affordance ids from the list above to act on or check these)")
        # Handle both flat dict and nested {artifacts: {...}} structure
        state_items = state
        if isinstance(state, dict) and "artifacts" in state and isinstance(state["artifacts"], dict):
            state_items = state["artifacts"]
        if isinstance(state_items, dict):
            for artifact_uri, props in state_items.items():
                artifact_key = _short_state_key(artifact_uri)
                if not isinstance(props, dict):
                    lines.append(f"- {artifact_key}: {props}")
                    continue
                flat = _flatten_state_props(props)
                if not flat:
                    continue
                if len(flat) == 1:
                    prop, val = next(iter(flat.items()))
                    lines.append(f"- {artifact_key}: {prop}={val}")
                else:
                    lines.append(f"- {artifact_key}:")
                    for prop, val in flat.items():
                        lines.append(f"  - {prop}: {val}")

    return "\n".join(lines) if lines else "No affordances available."


_NOISY_STATE_PROPS = {"metadata", "name", "workspace_id", "persistent"}


def _flatten_state_props(props: dict) -> dict:
    """
    Flatten artifact state for prompt rendering.

    Shortens URI-shaped keys, drops noisy bookkeeping entries, and inlines
    aggregate dicts keyed by property URIs (some adapters nest the actual
    property values inside a single 'state' entry).
    """
    flat: dict = {}
    for key, val in props.items():
        short = _short_state_key(key)
        if short in _NOISY_STATE_PROPS:
            continue
        if isinstance(val, dict) and any(
            isinstance(k, str) and k.startswith("http") for k in val
        ):
            for sub_key, sub_val in val.items():
                sub_short = _short_state_key(sub_key)
                if sub_short in _NOISY_STATE_PROPS:
                    continue
                flat.setdefault(sub_short, sub_val)
        else:
            flat.setdefault(short, val)
    return flat


def _short_state_key(key) -> str:
    """Shorten URI-shaped state keys to their last path segment."""
    text = str(key)
    if not text.startswith("http"):
        return text
    text = text.split("#")[0].rstrip("/")
    return text.rsplit("/", 1)[-1] or text


EXPLICIT_SIMILARITY_THRESHOLD = 0.95


def format_signifier_hints(
    signifier_matches: dict | None,
    url_to_ref: dict[str, str] | None = None,
) -> str:
    """
    Format signifier matches as BT planning hints.

    Matches above ``EXPLICIT_SIMILARITY_THRESHOLD`` are labelled as *exact*
    (reuse affordance **and** parameters as-is).  Lower-similarity matches
    are labelled as *suggested* (reuse affordance, but adjust parameters to
    the current context).

    Args:
        signifier_matches: Dict of intent -> match data from EnvExplorer/community
        url_to_ref: Optional map of affordance target URL -> short affordance id

    Returns:
        Formatted hints string for the planning prompt
    """
    if not signifier_matches:
        return ""
    url_to_ref = url_to_ref or {}

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
                ref = url_to_ref.get(str(aff_uri).rstrip("/"), aff_uri)
                lines.append(f"  Recommended affordance_id: {ref}")
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


def format_observable_property_hints(
    observable_property_matches: dict | None,
    url_to_ref: dict[str, str] | None = None,
) -> str:
    """Format observable-property effect queries for the planning prompt."""
    if not observable_property_matches:
        return ""
    url_to_ref = url_to_ref or {}

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
            lines.append("  Readable sensor property ids/URLs for condition affordance_id:")
            for property_url in readable_property_urls:
                ref = url_to_ref.get(str(property_url).rstrip("/"), property_url)
                lines.append(f"  - {ref}")
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
            action_target = str(action.get("action_target") or "")
            action_ref = url_to_ref.get(action_target.rstrip("/"), action_target)
            settling_time = action.get("settling_time_seconds")
            settling_text = ""
            if settling_time is not None:
                settling_text = f"; settling_time={settling_time}s"
            lines.append(
                f"  - {title}: {action_name} -> {direction} (affordance_id: {action_ref}{settling_text})"
            )
        lines.append("")

    return "\n".join(lines)
