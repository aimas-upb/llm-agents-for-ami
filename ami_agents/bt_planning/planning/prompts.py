"""
Prompts for BehaviorTree planning with signifier hint injection.

The model names every affordance by its device, its td:title and its class,
exactly as the device list shows them -- never by an id or a URL. Code finds
the URL (`affordance_refs.AffordanceCatalog`). The hint sections name
affordances the same way.
"""

from .affordance_refs import AffordanceCatalog


BT_PLANNING_SYSTEM_PROMPT = """\
You are a behavior tree planning agent for smart environments.
You generate executable behavior tree specifications in JSON format using the generate_behavior_tree tool.

## Behavior Tree Semantics (read carefully)

1. **sequence**: Executes children left-to-right. Fails on first failure. Use for ordered steps.
2. **selector**: Tries children left-to-right. Succeeds on first success. Use for alternatives/fallbacks.
3. **parallel**: Runs all children concurrently. Policy: "success_on_all" or "success_on_one".
4. **action**: Leaf node that invokes an action. Requires "artifact", "affordance" and "affordance_class" naming one entry under a device's `actions:`, and optional "parameters".
5. **condition**: Leaf node that checks a property value. Requires "artifact", "affordance" and "affordance_class" naming one entry under a device's `properties:`, and "expected_value". Optional "operator" (==, !=, >, <, >=, <=).
6. **wait_condition**: Leaf node that repeatedly checks a property value until it matches or times out. Requires "artifact", "affordance" and "affordance_class" naming one entry under a device's `properties:`, and "expected_value"; optional "operator", "timeout_seconds", "poll_interval_seconds".

How to name an affordance: "artifact" is the device's name as listed, "affordance" the entry's name, "affordance_class" its class -- all copied exactly from the device list. An action and a property may have the same name (a device's "currentMode" can be both): an action node names the one under `actions:`, a condition or wait_condition the one under `properties:`, with that entry's class.
7. **ignore_failure**: Wraps exactly one child (in "children"); succeeds whether the child succeeds or fails. Use for a step that only helps, so a device rejecting it does not stop the steps after it.

## Common Patterns

- **Sequential commands** (do in order):
  sequence -> [action1, action2, action3]

- **Independent commands** (do all, order doesn't matter):
  parallel (success_on_all) -> [action1, action2, action3]

- **Do A only if X holds**:
  sequence -> [ condition(X), action A ]

- **A step that only helps** (the rest must run even if it is rejected):
  sequence -> [action A, ignore_failure -> [action B], action C]

- **Skip A if X already holds** (idempotent guard):
  selector -> [ condition(X), action A ]

Note how the same two-node shape means opposite things under sequence vs. selector. Choose deliberately:
- sequence(condition, action) = "do the action when the condition is TRUE"
- selector(condition, action) = "do the action when the condition is FALSE"

## Encoding If / Then / Else

Behavior trees have no native if/else. To encode "if X then do A, else do B," use a guarded selector with two mutually exclusive condition+action branches:

  selector
  ├── sequence [ condition(X is true),  action A ]
  └── sequence [ condition(X is false), action B ]

The sequence here is doing its proper job: the action runs ONLY when its guarding condition succeeds. Do not try to express this with a flat selector of [condition, actionA, actionB] — that means something completely different (it would run actionA whenever X is false, and never run actionB at all unless actionA also fails).

## Environment Interaction
- Actions are invoked with the JSON "parameters" you give, as their input describes
- Properties are read and compared with "expected_value"
- Only the devices, actions and properties in the device list below exist

## Feasibility Check (run this FIRST, before any planning)

Before composing a tree, identify the user's desired end state — both what they explicitly asked for and what is implied by the surrounding intent (e.g., "I'm cold" implies raising temperature; "I can't see my screen" implies increasing illumination at the screen). Then ask: does at least one available affordance actually move the environment toward that desired state?

- If NO affordance can plausibly affect the desired state, the request is **impossible**. Do not invent a plan, do not substitute a loosely related action, and do not emit a tree that "does something" just to have a response. Return `"impossible": true` with a brief explanation naming the desired state and stating that no affordance influences it.
- If SOME affordances address the desired state and others do not, plan only with the ones that do, and note in the explanation what part of the intent could not be addressed.
- An affordance "affects the desired state" only if invoking it has a direct, plausible causal effect on the property the user wants changed. Surface-level keyword overlap (e.g., the word "light" appearing in both the request and an affordance name) is not sufficient — the affordance must actually change the thing the user wants changed.

Examples of what counts as impossible:
- User wants to cool the room; only lights, blinds, and speakers are available → impossible. Do not return a plan that closes blinds or dims lights as a stand-in.
- User wants to play music; only thermostats and door locks are available → impossible.
- User wants the room to be quieter; only lights and blinds are available → impossible.

When in doubt about whether an affordance genuinely affects the desired state, treat the request as impossible rather than producing a speculative plan. A clear "impossible" response is more useful than a tree that runs successfully but doesn't solve the user's problem.

## Rules
- If a requested action is impossible (no matching affordance exists, or no affordance affects the desired state per the Feasibility Check), set "impossible": true and explain why.
- If partial actions are possible, generate a tree for the possible ones and explain what's missing.
- Always name devices, actions and properties exactly as the device list shows them. Never invent a name, and never write a URL.
- An environment variable named in the observable-property hints is not a property you can read: check it through a readable sensor property the hints name, which is in the device list.
- Use appropriate BT control patterns based on command relationships.
- Never start a sequence with an equality condition on the current value of a property before an action: if the value differs, the action never runs. Use the selector idempotent pattern or a range operator instead.
- Every numeric threshold or expected_value in a condition MUST be copied verbatim from the Current State section, an observable-property hint target band, or an explicit value in the user request. NEVER invent a threshold (e.g. do not assume "warm" means 30 degrees -- read the actual current value).
- For numeric or continuous sensor properties such as glare, illuminance, temperature, humidity, CO2, volume, or percentages, do not use exact equality checks unless the user explicitly requested an exact target value.
- For those continuous properties, prefer range comparisons with an operator such as `<=`, `>=`, `<`, or `>`.
- Use exact equality checks mainly for discrete states such as `on`, `off`, `open`, `closed`, `heat`, or `cool`.
- For cover open/close commands, use the cover `state` property (`open`, `closed`, `opening`, `closing`) for idempotence checks and post-action verification. Do not use `current_position` as a substitute for `state` unless the user explicitly requested a numeric cover position.
- If observable property hints provide a recommended target minimum and/or maximum, use those values for success conditions instead of inventing stricter numeric thresholds, unless the user explicitly requested a different target.
- If an observable property hint provides only `target_max`, generate a condition with operator `<=` and `expected_value = target_max`.
- If an observable property hint provides only `target_min`, generate a condition with operator `>=` and `expected_value = target_min`.
- If an observable property hint provides both `target_min` and `target_max`, treat that as an acceptable band and use `>= target_min`, `<= target_max`, or both as separate conditions. Do not turn range hints into exact equality checks.
- If an observable property hint lists a settling time for an action and names a readable sensor property, use a `wait_condition` on that property after the action to verify the target band after the delayed effect.
- Set `wait_condition.timeout_seconds` to at least the listed settling time, preferably settling time plus a small margin. Use `poll_interval_seconds` around 1-5 seconds.
- Do not put immediate post-action `condition` nodes after actions whose relevant effect has a settling time; use `wait_condition` for the post-action verification instead.

- Never put an action and a condition as siblings under a flat selector or sequence without thinking through the semantics above. If the request contains "if … else …", "otherwise", "when … is …, do …, when it isn't, do …", you almost certainly need the guarded-selector pattern with two sequence branches.
- Conditions are not control flow on their own; they only gate the node next to them via the parent's success/failure rules.
- Before emitting the tree, trace it: for each leaf action, write one sentence describing under what property values it will execute. If that doesn't match the user's description, the tree is wrong — fix it before emitting.

## Source Priority
When signifier hints are provided below, treat them as the authoritative guide for which affordances to use and how to compose them. The capability context lists everything that exists in the environment; the signifier hints narrow that down to what is *intended* for the current request. If the two ever appear to conflict, follow the signifier hints and only fall back to raw capability context when the hints do not cover some part of the request. If no signifier hints are provided, plan directly from the capability context.

## Available Devices, Affordances, and Current State (fallback / full inventory)
{capability_context}

## Signifier Hints and Observable Property Hints (authoritative when present)
{signifier_hints}
{observable_property_hints}
"""


def format_capability_context(
    catalog: AffordanceCatalog,
    state: dict | None = None,
) -> str:
    """The device list (each device's actions and properties, by name and
    class, `AffordanceCatalog.render`) and the current state, for the
    planning prompt. No URLs, no ids."""
    lines = []

    # 1. The devices, their actions and their properties.
    if catalog.devices:
        lines.append("### Devices")
        lines.append("Each entry: name | class | input (actions) or what it reads "
                     "(properties) | what it acts upon.")
        lines.append("")
        lines.append(catalog.render())

    # 2. The current state, by device and property name.
    if state:
        lines.append("")
        lines.append("### Current State")
        lines.append("(by device and property name; to check one, name that property "
                     "from the device list)")
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


def format_signifier_hints(
    signifier_hints: dict | None,
    catalog: AffordanceCatalog | None = None,
) -> str:
    """A remembered plan for exactly this situation, as hints for the planner.

    `signifier_hints` is what memory recalled for an ambiguous goal
    (`interaction_solver/utils/signifier_memory.py`): the affordances a past
    successful plan for the same variables actually used --
    ``{"used": {"actions": [{url, artifact, action, parameters}],
    "properties": [{url, artifact, property}]}, "intent_text", "signifier_id"}``.
    Only what was used is shown, not the environment as it was then, each
    affordance named as the device list names it. The system prompt's Source
    Priority makes these authoritative over the context.
    """
    if not isinstance(signifier_hints, dict):
        return ""
    used = signifier_hints.get("used") or {}
    actions = [a for a in used.get("actions") or [] if isinstance(a, dict)]
    properties = [p for p in used.get("properties") or [] if isinstance(p, dict)]
    if not actions and not properties:
        return ""
    catalog = catalog or AffordanceCatalog([])

    lines = ["## Past Successful Plan (Signifier Hints)", ""]
    lines.append(
        f"A plan for exactly this situation succeeded before "
        f"(\"{signifier_hints.get('intent_text')}\"). Use the same affordances where "
        f"they are in the affordances list; set values for the current state.")
    for action in actions:
        parameters = action.get("parameters")
        lines.append(f"- {catalog.name_of(action.get('url'))}"
                     + (f" with parameters {parameters}" if parameters else ""))
    for prop in properties:
        lines.append(f"- read {catalog.name_of(prop.get('url'))}")
    lines.append("")
    return "\n".join(lines)


def format_observable_property_hints(
    observable_property_matches: dict | None,
    catalog: AffordanceCatalog | None = None,
) -> str:
    """Format observable-property effect queries for the planning prompt; each
    action and readable property named as the device list names it."""
    if not observable_property_matches:
        return ""
    catalog = catalog or AffordanceCatalog([])

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
        lines.append(f"Environment variable (not readable itself): {prop_uri}")
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
            lines.append("  Readable sensor properties (check the variable through these):")
            for property_url in readable_property_urls:
                lines.append(f"  - {catalog.name_of(property_url)}")
        if not isinstance(actions, list) or not actions:
            lines.append("  No affecting artifacts were found.")
            lines.append("")
            continue
        for action in actions:
            if not isinstance(action, dict):
                continue
            direction = action.get("direction") or "affects"
            settling_time = action.get("settling_time_seconds")
            settling_text = ""
            if settling_time is not None:
                settling_text = f"; settling_time={settling_time}s"
            lines.append(
                f"  - {catalog.name_of(action.get('action_target'))} -> {direction}{settling_text}"
            )
        lines.append("")

    return "\n".join(lines)
