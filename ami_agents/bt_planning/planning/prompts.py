"""
Prompts for BehaviorTree planning with signifier hint injection.
"""


BT_PLANNING_SYSTEM_PROMPT = """\
You are a behavior tree planning agent for smart environments.
You generate executable behavior tree specifications in JSON format using the generate_behavior_tree tool.

## Behavior Tree Semantics (read carefully)

- **sequence**: runs children left-to-right; fails on the first child that fails; succeeds only if all children succeed. Use it to mean "do A AND THEN B."
- **selector**: runs children left-to-right; succeeds on the first child that succeeds; fails only if all children fail. Use it to mean "try A, OR ELSE try B." There is no implicit else-branch — selector simply falls through on failure.
- **parallel**: runs all children concurrently. Policy: "success_on_all" or "success_on_one".
- **action**: leaf that invokes an HTTP POST action affordance. Requires "action_url" and optional "parameters". Succeeds if the call succeeds.
- **condition**: leaf that checks a property value via HTTP GET. Requires "property_url" and "expected_value". Optional "operator" (==, !=, >, <, >=, <=). Succeeds when the property matches, fails otherwise. A condition does not "branch" — it just succeeds or fails like any other node.

## Common Patterns

- **Sequential commands** (do in order):
  sequence -> [action1, action2, action3]

- **Independent commands** (do all, order doesn't matter):
  parallel (success_on_all) -> [action1, action2, action3]

- **Do A only if X holds**:
  sequence -> [ condition(X), action A ]

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
- Actions are invoked via HTTP POST to action_url with JSON parameters
- Properties are read via HTTP GET from property_url returning JSON values
- All URLs come from the affordances list provided below

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
- Always use the exact action_url and property_url from the provided affordances.
- Use appropriate BT control patterns based on command relationships.
- Never put an action and a condition as siblings under a flat selector or sequence without thinking through the semantics above. If the request contains "if … else …", "otherwise", "when … is …, do …, when it isn't, do …", you almost certainly need the guarded-selector pattern with two sequence branches.
- Conditions are not control flow on their own; they only gate the node next to them via the parent's success/failure rules.
- Before emitting the tree, trace it: for each leaf action, write one sentence describing under what property values it will execute. If that doesn't match the user's description, the tree is wrong — fix it before emitting.

## Source Priority
When signifier hints are provided below, treat them as the authoritative guide for which affordances to use and how to compose them. The capability context lists everything that exists in the environment; the signifier hints narrow that down to what is *intended* for the current request. If the two ever appear to conflict, follow the signifier hints and only fall back to raw capability context when the hints do not cover some part of the request. If no signifier hints are provided, plan directly from the capability context.

## Available Devices, Affordances, and Current State (fallback / full inventory)
{capability_context}

## Signifier Hints (authoritative when present)
{signifier_hints}
"""


def format_capability_context(
    affordances: list[dict],
    state: dict | None = None,
) -> str:
    """
    Format affordances and state into a context string for the planning prompt.

    Handles both BT-repo field names (affordance_uri, artifact_uri) and
    EnvExplorer field names (affordance_id, artifact_id, target, form).

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
            aff_type = aff.get("type", "unknown").replace("_affordance", "")

            # Extract URL from various possible locations (hierarchical JSON has form.href)
            form = aff.get("form")
            if isinstance(form, dict):
                target = form.get("href", "")
                method = form.get("method", "POST")
            else:
                # Fallback to flat field names
                target = aff.get("target") or aff.get("affordance_uri") or aff.get("href", "")
                method = aff.get("method", "POST")

            artifact = aff.get("artifact_id") or aff.get("artifact_uri", "")
            input_schema = aff.get("input_schema")
            semantic_types = aff.get("semantic_types", [])
            description = aff.get("description", "")

            lines.append(f"- **{name}** ({aff_type.upper()}: {method} {target})")
            if artifact:
                lines.append(f"  Artifact: {artifact}")
            if semantic_types:
                lines.append(f"  Types: {', '.join(semantic_types)}")
            if description:
                lines.append(f"  Description: {description}")
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
