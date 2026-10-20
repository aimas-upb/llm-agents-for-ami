"""The condition-guard check: decide, for each conditioned step of a plan, its guard.

Run on every plan once its steps are planned -- simple plans by
`ConditionGuardBehaviour`, dependency plans in their step 3c, after profiling.
The steps are the plan's actions on devices with a condition
(`planning_conditions.conditioned_steps`). For each:

    1. memory     the guard memory (`guard_memory.recall_guard`), by the step's
                  exact key. A hit is resolved again on today's TD graph
                  (`condition_guard.apply_remembered`); one that no longer
                  resolves counts as a miss.
    2. the model  every miss in one call to the small model (the guard model,
                  `agent.guard_llm`, when the agent has one -- the bench sets it
                  -- else the profiling model, `agent.explicit_llm`) with
                  CONDITION_GUARD_PROMPT: the
                  request, the plan's actions in order, the steps with their
                  conditions, and the TDs of their devices. Each answer is
                  resolved against the step's own device
                  (`condition_guard.resolve_guard`).
    3. fallback   a failed call, a missing or unusable answer: no guard, with
                  a note saying which, and `why: "check_failed"` -- the check
                  never stops a plan. The reply's JSON shape is enforced by the
                  API (strict structured output, REPLY_SCHEMA), so a reply
                  that does not parse is no longer one of them.

Example -- "set washer 1's spin to 1200", nothing remembered:

    steps      [S1 washer_1.spinSpeedCurrent 1200, laundry_washer_settings_not_running]
    memory     S1 key -> miss
    model      {"answers": [{"id": "S1", "guard": true,
                 "check": {"property": "operationalState", ..., "operator": "!=", "value": 1},
                 "action": {"action": "operationalState", ..., "value": 2}}]}
    decisions  {"S1": Decision(guard=Guard(check != 1, action 2, restore 1),
                               source="model")}

Which of these guards the plan already satisfies is decided afterwards, on the
whole tree (`condition_guard.drop_satisfied`).

    decide(agent, logger, graph, steps, request, plan_actions, trace)
        -> ({step id: Decision}, notes)
    guard_trees(agent, logger, graph, trees, request, conditions, trace)
        -> (the trees with guarded steps wrapped, decisions, notes)
    plan_actions(graph, trees, steps)   the plan's actions, in tree order
    render_condition_guard_prompt(...)  the prompt with its inputs filled in
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Graph

from ....shared.utils.demo_log import demo
from ..prompts import CONDITION_GUARD_PROMPT
from .condition_guard import (
    CHECK_FAILED,
    FROM_MODEL,
    WHY,
    Decision,
    NOT_APPLICABLE,
    apply_remembered,
    resolve_guard,
    wrap_steps,
)
from .dependency_context import artifact_of_url
from .guard_memory import recall_guard
from .planning_conditions import Step, _action_of, conditioned_steps
from .profiling import _is_reasoning, _json_object
from .specific_capability_context import (
    DIRECT_VALUE_KEY,
    _command_types,
    _title,
    artifact_affordances,
    device_class,
)


# -- the reply's shape -------------------------------------------------------

# A device value, as the TD types it.
_VALUE = {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"},
                    {"type": "null"}]}

# The reply, enforced by the API (OpenAI structured outputs, strict): the model
# cannot return anything that does not parse into this. Strict mode needs every
# property listed as required and no others; an absent part is null.
REPLY_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["answers"],
    "properties": {"answers": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", "guard", "check", "action", "why", "reason"],
        "properties": {
            "id": {"type": "string"},
            "guard": {"type": "boolean"},
            "check": {"anyOf": [{"type": "null"}, {
                "type": "object", "additionalProperties": False,
                "required": ["property", "property_class", "operator", "value"],
                "properties": {"property": {"type": "string"},
                               "property_class": {"type": "string"},
                               "operator": {"type": "string", "enum": ["==", "!="]},
                               "value": _VALUE}}]},
            "action": {"anyOf": [{"type": "null"}, {
                "type": "object", "additionalProperties": False,
                "required": ["action", "action_class", "value"],
                "properties": {"action": {"type": "string"},
                               "action_class": {"type": "string"},
                               "value": _VALUE}}]},
            "why": {"anyOf": [{"type": "null"},
                              {"type": "string", "enum": list(WHY)}]},
            "reason": {"type": "string"}}}}}}

RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "condition_guards", "strict": True,
                                   "schema": REPLY_SCHEMA}}


# -- the prompt's inputs -----------------------------------------------------

def plan_actions(graph: Graph, trees: List[Tuple[Optional[str], Any]],
                 steps: List[Step]) -> List[Dict[str, Any]]:
    """Every action of the plan, in tree order: (goal, device, action, value),
    with the step id where the action is a step. ``trees`` are (goal id or
    None, tree) -- one per goal of a dependency structure, else one.

    The device and the action are named by their td:title, looked up from the
    action's URL -- the names `steps` and `devices` use -- never by the tree
    node's name, which each planner makes up ("Currentmode")."""
    by_node = {id(s.node): s.id for s in steps}
    out: List[Dict[str, Any]] = []

    def walk(goal: Optional[str], node: Any) -> None:
        """Depth-first, children in order."""
        if not isinstance(node, dict):
            return
        if node.get("type") == "action" and node.get("action_url"):
            # The device and its action affordance, from the URL; the node's
            # name only when the URL is not in the TDs.
            artifact = artifact_of_url(graph, node["action_url"])
            action = _action_of(graph, artifact, node["action_url"]) \
                if artifact is not None else None
            out.append({**({"step": by_node[id(node)]} if id(node) in by_node else {}),
                        **({"goal": goal} if goal else {}),
                        "device": _title(graph, artifact) if artifact is not None else None,
                        "action": (_title(graph, action) if action is not None else None)
                        or node.get("name"),
                        "value": (node.get("parameters") or {}).get(DIRECT_VALUE_KEY)})
        for child in node.get("children") or []:
            walk(goal, child)

    for goal, tree in trees:
        walk(goal, tree)
    return out


# The fields of an affordance the model reads: what it is, what it takes or
# reports. URLs, forms and artifact ids are left out -- the answer names
# affordances by title and class, and code finds their URLs in the TDs.
_BRIEF_FIELDS = ("name", "semantic_types", "description", "input_schema",
                 "output_schema", "unit")
_KINDS = {"action_affordance": "action", "property_affordance": "property"}


def _brief_affordance(aff: Dict[str, Any]) -> Dict[str, Any]:
    """One affordance as the model reads it: its kind, name, classes,
    description and schema."""
    return {"kind": _KINDS.get(aff.get("type"), aff.get("type")),
            **{k: aff[k] for k in _BRIEF_FIELDS if aff.get(k) not in (None, "", [])}}


def device_context(graph: Graph, steps: List[Step]) -> Dict[str, Any]:
    """The `devices` input: per device a step acts on (by td:title), its
    affordances as `_brief_affordance` gives them."""
    devices: Dict[str, Any] = {}
    for step in steps:
        title = _title(graph, step.artifact) or str(step.artifact)
        if title not in devices:
            devices[title] = [_brief_affordance(a)
                              for a in artifact_affordances(graph, step.artifact)]
    return devices


def _step_brief(graph: Graph, step: Step) -> Dict[str, Any]:
    """What the model reads of one step."""
    commands = _command_types(graph, step.action)
    return {"id": step.id, **({"goal": step.goal} if step.goal else {}),
            "device_name": _title(graph, step.artifact),
            "device_class": device_class(graph, step.artifact),
            "action_name": _title(graph, step.action),
            "action_class": commands[0] if commands else None,
            "value": step.value,
            "conditions": [c.to_wire_dict() for c in step.conditions]}


def render_condition_guard_prompt(request: str, actions: str, steps: str, devices: str) -> str:
    """CONDITION_GUARD_PROMPT with its four inputs filled in, by plain
    replacement (the prompt quotes JSON, so str.format would need every brace
    doubled)."""
    return (CONDITION_GUARD_PROMPT
            .replace("{request}", request)
            .replace("{plan_actions}", actions)
            .replace("{steps}", steps)
            .replace("{devices}", devices))


# -- the model ---------------------------------------------------------------

async def _ask(agent, logger, graph: Graph, steps: List[Step], request: Any,
               actions: List[Dict[str, Any]], trace: Dict[str, Any]) -> Dict[str, Any]:
    """The model's answers, by step id. The model is ``agent.guard_llm`` when
    the agent has one, else ``agent.explicit_llm``. The reply's shape is
    enforced by the API (RESPONSE_FORMAT), so it always parses; raises on a
    failed call (or a refusal), and the caller falls back."""
    # 1. The prompt: the request, the plan's actions, the steps, their devices.
    devices = device_context(graph, steps)
    prompt = render_condition_guard_prompt(
        json.dumps(request, indent=1, default=str), json.dumps(actions, indent=1, default=str),
        json.dumps([_step_brief(graph, s) for s in steps], indent=1, default=str),
        json.dumps(devices, indent=1, default=str))

    # 2. The call, with the guard model's parameters (as `profiling.ask_profile`):
    #    `guard_llm` when the agent has one, else its small model.
    cfg = getattr(agent, "guard_llm", None) or agent.explicit_llm
    trace.update({"model": cfg.model, "steps": [s.id for s in steps], "answer": None})
    kwargs: Dict[str, Any] = {}
    if _is_reasoning(cfg.model):
        if cfg.reasoning_effort:
            kwargs["reasoning_effort"] = cfg.reasoning_effort
        if cfg.max_completion_tokens:
            kwargs["max_completion_tokens"] = cfg.max_completion_tokens
    else:
        kwargs["temperature"] = cfg.temperature
    logger.info(demo(f"Condition check: {len(steps)} step(s) for {cfg.model}"))
    response = await cfg.client.chat.completions.create(
        model=cfg.model, messages=[{"role": "user", "content": prompt}],
        response_format=RESPONSE_FORMAT, **kwargs)

    # 3. The raw reply kept for the entry, then its answers by id. `_json_object`
    #    still guards the parse: a refusal or a truncated reply has no JSON.
    choice = response.choices[0]
    trace["answer"] = choice.message.content or ""
    trace["finish_reason"] = getattr(choice, "finish_reason", None)
    output = _json_object(trace["answer"])
    return {str(a.get("id")): a for a in output.get("answers") or [] if isinstance(a, dict)}


def _decision(graph: Graph, step: Step, answer: Optional[Dict[str, Any]],
              failure: Optional[str] = None) -> Decision:
    """The model's answer for ``step`` as a decision: a resolved guard, a
    reasoned "no guard", or -- missing / unusable -- no guard with a note.
    ``failure`` is why the whole call gave nothing (the call failed, or its
    reply could not be read), for the note."""
    # 1. Nothing came back for this step: the call failed, or skipped it.
    if not isinstance(answer, dict):
        why = f"condition check failed: {failure}" if failure else \
            "the condition check returned no answer for it"
        return Decision(step=step, source=FROM_MODEL, why=CHECK_FAILED,
                        notes=[f"{step.id}: {why}; not guarded"])
    reason = answer.get("reason")
    # 2. "No guard", with its reason.
    if answer.get("guard") is not True:
        why = answer.get("why") if answer.get("why") in WHY else NOT_APPLICABLE
        return Decision(step=step, source=FROM_MODEL, why=why, reason=reason)
    # 3. A guard: resolved on the step's device, or no guard with why not.
    guard, why_not = resolve_guard(graph, step, answer)
    if guard is None:
        return Decision(step=step, source=FROM_MODEL, why=CHECK_FAILED, reason=reason,
                        notes=[f"{step.id}: guard not built ({why_not}); not guarded"])
    return Decision(step=step, source=FROM_MODEL, guard=guard, reason=reason)


# -- the check ---------------------------------------------------------------

async def decide(agent, logger, graph: Graph, steps: List[Step], request: Any,
                 actions: List[Dict[str, Any]], trace: Optional[Dict[str, Any]] = None
                 ) -> Tuple[Dict[str, Decision], List[str]]:
    """({step id: decision}, notes) for ``steps`` -- memory first, then one
    model call for the misses (see the module docstring). Never raises."""
    trace = {} if trace is None else trace
    decisions: Dict[str, Decision] = {}

    # 1. Memory, once per distinct key: a hit that still resolves decides.
    remembered: Dict[str, Optional[Dict[str, Any]]] = {}
    for step in steps:
        digest = json.dumps(step.key, sort_keys=True)
        if digest not in remembered:
            remembered[digest] = await recall_guard(agent, step.key, logger)
        record = remembered[digest]
        decision = apply_remembered(graph, step, record) if record else None
        if decision is not None:
            decisions[step.id] = decision
    misses = [s for s in steps if s.id not in decisions]

    # 2. The model, for the misses; a failure leaves them all unguarded.
    if misses:
        failure = None
        try:
            answers = await _ask(agent, logger, graph, misses, request, actions, trace)
        except Exception as exc:  # noqa: BLE001 - the check never stops a plan
            logger.warning(f"Condition check failed, steps left unguarded: {exc}")
            answers, failure = {}, str(exc)
            trace.setdefault("error", failure)
        for step in misses:
            decisions[step.id] = _decision(graph, step, answers.get(step.id), failure)

    # 3. The notes, for the plan's explanation and the run log.
    notes = [note for d in decisions.values() for note in d.notes]
    for decision in decisions.values():
        logger.info(demo(f"[CONDITION] {decision.step.id} {decision.step.key.get('device_name')}."
                         f"{decision.step.key.get('affordance_name')}: "
                         + (f"guard ({decision.source})" if decision.guard
                            else f"no guard ({decision.why or 'unresolved'}, {decision.source})")))
    return decisions, notes


# -- a whole plan ------------------------------------------------------------

async def guard_trees(agent, logger, graph: Graph, trees: List[Tuple[Optional[str], Any]],
                      request: Any, conditions, trace: Optional[Dict[str, Any]] = None
                      ) -> Tuple[List[Any], Dict[str, Decision], List[str]]:
    """(``trees`` with their guarded steps wrapped, the decisions, notes).

    ``trees`` are (goal id or None, tree): one per goal of a dependency
    structure (and per candidate device of a filtered goal), or the one tree
    of a simple plan. Their steps are numbered across all of them and decided
    in one check, so the model sees the whole plan. Guards the plan already
    satisfies are not removed here: the caller runs `drop_satisfied` on the
    final tree, where execution order across goals is known."""
    # 1. The steps of every tree, numbered across the plan.
    per_tree: List[List[Step]] = []
    for goal, tree in trees:
        found = conditioned_steps(graph, tree, conditions, goal=goal,
                                  first=1 + sum(len(s) for s in per_tree))
        per_tree.append(found)
    steps = [s for found in per_tree for s in found]
    if not steps:
        return [tree for _, tree in trees], {}, []

    # 2. One decision per step: memory, then the model.
    decisions, notes = await decide(agent, logger, graph, steps, request,
                                    plan_actions(graph, trees, steps), trace)

    # 3. Each tree with its guarded steps wrapped.
    return ([wrap_steps(tree, found, decisions) for (_, tree), found in zip(trees, per_tree)],
            decisions, notes)
