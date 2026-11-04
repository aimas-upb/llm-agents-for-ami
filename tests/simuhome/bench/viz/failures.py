"""The failure catalogue: one category and a one-line explanation per episode that
did not pass, derived from the record and its evaluation alone (no LLM).

    category, explanation = categorise(record, evaluation)

`CATEGORIES` lists every category in the order they are tested; the first whose rule
fires wins, so an episode has exactly one. The rules read, in this order: the score
(-1 is infrastructure), our pipeline's `outcome` / `error_step`, what segmentation
and structuring produced, the plan entries (`outcome`: clarify / impossible / error
with its `error` code), the execution (`plans[].status` / `error`, the actions), and
the evaluation (`error_type`, the execution check, the judge votes, the per-check
classes ok / missing / wrong_value / wrong_timing).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# (category, how it is detected). The order is the precedence.
CATEGORIES: List[Tuple[str, str]] = [
    ("infrastructure error",
     "score -1: the API, the simulator or the judge failed (eval error_type Infrastructure, "
     "Not Run, No Simulator, Judge Error, Evaluation Runtime Error), or the record's error "
     "holds an infrastructure marker."),
    ("pipeline error",
     "the record's outcome is `error`: our code raised, in `error_step`."),
    ("nothing understood",
     "segmentation produced no actionable intent (outcome `nothing`, no structures, no plans)."),
    ("empty structuring",
     "a structure carries an error (the model returned no JSON object / an unparsable one) "
     "or has no goals, so nothing reached the planner for that intent."),
    ("executed infeasible part",
     "infeasible request: an action ran on the part that cannot be done (execution check "
     "`Executed Infeasible Part`, with the offending actions)."),
    ("not reported infeasible",
     "infeasible request: nothing ran on the impossible part but the reply did not say it "
     "was impossible nor ask (execution check `Not Reported Infeasible`)."),
    ("judge rejected reply",
     "infeasible request: the execution check passed but SimuHome's judge panel voted the "
     "reply down (`Context Faithfulness Failure`, majority B)."),
    ("clarification asked",
     "feasible request answered with a question instead of acted on: a structure with "
     "`clarification`, or a plan entry with outcome `clarify` (e.g. `stated_time_conflict`)."),
    ("impossible verdict",
     "feasible request: a plan entry answered `impossible` (false impossibility: the planner "
     "found no affordance for the goal)."),
    ("invalid plan tree",
     "the general BT planner's tree was rejected at validation "
     "(`ambiguous_planning_failed`: wrong action classes, unknown devices)."),
    ("no plan path",
     "the planner raised before a tree existed: `unplannable_goal`, `unresolvable_predicate`, "
     "`dependency_compile_failed`, `unrouted`."),
    ("execution timeout",
     "a plan ran out of real time (`timeout after Ns`) or was still RUNNING when the episode ended."),
    ("action rejected",
     "the home rejected an action: a plan error with an HTTP status, or an action with success false."),
    ("plan execution failed",
     "a plan ended FAILURE without a rejected action or a timeout (a condition never held, "
     "a guard failed)."),
    ("wrong timing",
     "feasible qt4: every value was set but a check is `wrong_timing` (reached outside the window)."),
    ("wrong value",
     "feasible: a check is `wrong_value` (the attribute was set to another value)."),
    ("wrong device",
     "feasible: a check is `missing` while actions ran on devices the answer key does not "
     "name (the right thing on the wrong device)."),
    ("missing action",
     "feasible: a check is `missing` and no action was attempted on that device / attribute "
     "(the goal was not covered by any plan)."),
    ("required tools not invoked",
     "the reply or the actions were right but SimuHome's required tool calls "
     "(`get_room_devices`, `get_room_states`) were not made (`Required Actions Failure`)."),
    ("value mismatch",
     "qt1: a state was read but the value reported differs from the ground truth (`Value Mismatch`)."),
    ("other",
     "none of the rules above matched; see the eval error_type."),
]
CATEGORY_NAMES = [name for name, _ in CATEGORIES]

INFRA_TYPES = ("Infrastructure", "Not Run", "No Simulator", "Judge Error",
               "Evaluation Runtime Error")
NO_PLAN_ERRORS = ("unplannable_goal", "unresolvable_predicate", "dependency_compile_failed",
                  "unrouted", "no_plan")


def _clip(text: Any, n: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _checks(evaluation: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return ((evaluation or {}).get("breakdown") or {}).get("checks") or []


def _check_line(check: Dict[str, Any], actual: Optional[Any] = None) -> str:
    """"fan 80% on utility_room_air_purifier_1 (1.FanControl.PercentSetting): expected 80, got 50"."""
    what = check.get("description") or check.get("attribute") or check.get("expected")
    where = check.get("device") or check.get("room") or ""
    text = f"{what} on {where}" if where else str(what)
    if check.get("attribute") and check.get("description"):
        text += f" ({check['attribute']})"
    # qt1: the check is a reading, not a device value.
    if check.get("expected") in ("room_state", "device_attribute") and not check.get("attribute"):
        return f"{where}: {'value read but wrong' if check.get('class') == 'wrong_value' else 'state not read / not reported'}"
    if check.get("class") == "wrong_value":
        seen = check.get("values")
        final = seen[-1] if seen else actual
        # qt2: the expected value is a direction ("humidity increase"), not a number.
        if isinstance(check.get("expected"), str) and check.get("expected") == check.get("description"):
            text += f": not achieved (final {final!r})"
        else:
            text += f": expected {check.get('expected')!r}, got {final!r}"
    elif check.get("class") == "wrong_timing":
        text += f": reached at tick {check.get('reached_at')}, outside the window"
    elif check.get("class") == "missing":
        text += ": never set" + (" (an action was attempted)" if check.get("attempted") else "")
    return text


def _actual(evaluation: Optional[Dict[str, Any]], check: Dict[str, Any]) -> Optional[Any]:
    """The final value SimuHome saw for a check, from `targets_eval` when it has it."""
    for target in (evaluation or {}).get("targets_eval") or []:
        if target.get("device_id") != check.get("device"):
            continue
        for diff in target.get("diffs") or []:
            if diff.get("attribute") == check.get("attribute"):
                return diff.get("actual")
    return None


def _plan_entries(record: Dict[str, Any], outcome: str) -> List[Dict[str, Any]]:
    return [p for p in record.get("plans") or [] if p.get("outcome") == outcome]


def _judge_votes(evaluation: Optional[Dict[str, Any]]) -> str:
    votes = ((evaluation or {}).get("judge") or {}).get("judge") or []
    return "/".join(str(v) for v in votes)


def categorise(record: Dict[str, Any], evaluation: Optional[Dict[str, Any]]
               ) -> Tuple[Optional[str], str]:
    """(category, one-line explanation) for an episode that failed or errored; (None,
    "") for one that passed or is not scored yet."""
    evaluation = evaluation or {}
    score = evaluation.get("score")
    if score == 1 or (score is None and record.get("outcome") not in ("error",)):
        return None, ""
    error_type = evaluation.get("error_type")
    case = record.get("case")
    infeasible = case in ("infeasible", "mixed")
    structures = record.get("structures") or []
    plans = record.get("plans") or []
    execution = record.get("execution") or {}
    run_plans = execution.get("plans") or []
    actions = execution.get("actions") or []
    response = record.get("response") or ""

    # 1. Not the agents' fault.
    if score == -1 or error_type in INFRA_TYPES:
        return "infrastructure error", _clip(evaluation.get("detail") or record.get("error")
                                             or error_type or "score -1")
    if record.get("outcome") == "error":
        return "pipeline error", _clip(f"{record.get('error_step') or '?'}: {record.get('error')}")

    # 2. Nothing came out of the UA.
    if record.get("outcome") == "nothing" or (not record.get("intents") and not structures
                                              and not plans):
        return "nothing understood", _clip(f"no intents; reply: {response}")
    broken = [s for s in structures if s.get("error")
              or (s.get("structure") is not None and not (s.get("structure") or {}).get("goals")
                  and not s.get("clarification") and not s.get("question"))]
    if broken and not plans and not any(s.get("clarification") for s in structures):
        why = broken[0].get("error") or "no goals in the structure"
        return "empty structuring", _clip(f"{len(broken)} of {len(structures)} intent(s): {why}")

    # 3. Infeasible requests: the execution check and the judge.
    check = evaluation.get("execution_check") or {}
    if infeasible:
        if check.get("error_type") == "Executed Infeasible Part" or error_type == "Executed Infeasible Part":
            offending = check.get("offending_actions") or []
            names = ", ".join(_clip(a.get("name") or a.get("device_id") or a, 40) for a in offending[:3]) \
                if isinstance(offending, list) else str(offending)
            return "executed infeasible part", _clip(
                f"{len(offending)} action(s) on the impossible part ({names}); reply: {response}")
        if check.get("error_type") == "Not Reported Infeasible" or error_type == "Not Reported Infeasible":
            asked = [s.get("clarification") for s in structures if s.get("clarification")]
            detail = f"asked instead: {asked[0]}" if asked else f"reply: {response}"
            return "not reported infeasible", _clip(detail)
        judge = evaluation.get("judge") or {}
        if check.get("score") == 1 and judge.get("score") == 0:
            return "judge rejected reply", _clip(
                f"votes {_judge_votes(evaluation)}; reply: {response}")
        if error_type == "Required Actions Failure" or check.get("error_type") == "Required Actions Failure":
            missing = [f"{a.get('tool')}({', '.join(str(v) for v in (a.get('params') or {}).values())})"
                       for a in (evaluation.get("required_actions") or check.get("required_actions")
                                 or (judge.get("required_actions") or [])) if not a.get("invoked")]
            return "required tools not invoked", _clip("not invoked: " + ", ".join(missing))
        if check.get("error_type") == "Value Mismatch" or error_type == "Value Mismatch":
            return "value mismatch", _clip(f"reply: {response}")

    # 4. Refusals on a feasible request.
    if not infeasible:
        asked = [s.get("clarification") or s.get("question") for s in structures
                 if s.get("clarification") or s.get("question")]
        asked += [p.get("explanation") or p.get("question") for p in _plan_entries(record, "clarify")]
        if asked:
            return "clarification asked", _clip(f"“{asked[0]}”")
        impossible = _plan_entries(record, "impossible")
        if impossible:
            return "impossible verdict", _clip(impossible[0].get("explanation") or "impossible")

    # 5. The planner could not produce a tree.
    errored = _plan_entries(record, "error")
    for entry in errored:
        if entry.get("error") == "ambiguous_planning_failed":
            return "invalid plan tree", _clip(entry.get("explanation") or entry["error"])
    for entry in errored:
        if entry.get("error") in NO_PLAN_ERRORS or entry.get("error"):
            return "no plan path", _clip(f"{entry.get('error')}: {entry.get('explanation') or ''}")

    # 6. Execution.
    for plan in run_plans:
        error = str(plan.get("error") or "")
        if "timeout" in error.lower() or plan.get("status") == "RUNNING":
            return "execution timeout", _clip(error or f"plan {plan.get('plan')} still RUNNING "
                                              f"after {plan.get('ticks')} ticks")
    for plan in run_plans:
        error = str(plan.get("error") or "")
        if "HTTP" in error:
            return "action rejected", _clip(error)
    rejected = [a for a in actions if a.get("success") is False]
    if rejected:
        a = rejected[0]
        return "action rejected", _clip(f"{a.get('name')} -> HTTP {a.get('status_code')} "
                                        f"{a.get('error') or ''}")
    if any(p.get("status") == "FAILURE" for p in run_plans) and not infeasible \
            and not any(c.get("class") in ("wrong_value", "wrong_timing") for c in _checks(evaluation)):
        plan = next(p for p in run_plans if p.get("status") == "FAILURE")
        return "plan execution failed", _clip(
            f"plan {plan.get('plan')} ({plan.get('source')}) ended FAILURE after "
            f"{plan.get('ticks')} tick(s); {len(actions)} action(s) sent")

    # 7. The answer key, check by check.
    checks = _checks(evaluation)
    for klass, name in (("wrong_timing", "wrong timing"), ("wrong_value", "wrong value")):
        bad = [c for c in checks if c.get("class") == klass]
        if bad:
            return name, _clip("; ".join(_check_line(c, _actual(evaluation, c)) for c in bad[:2]))
    missing = [c for c in checks if c.get("class") == "missing"]
    if missing:
        expected = {c.get("device") for c in checks if c.get("device")}
        elsewhere = sorted({a.get("name", "").split(".")[0] for a in actions
                            if a.get("name", "").split(".")[0] not in expected})
        if elsewhere and any(not c.get("attempted") for c in missing):
            return "wrong device", _clip(
                f"{len(missing)} check(s) never set; acted on {', '.join(elsewhere)} instead")
        return "missing action", _clip("; ".join(_check_line(c) for c in missing[:2]))

    # 8. What is left: the evaluator's own label.
    if error_type == "Required Actions Failure":
        missing_tools = [f"{a.get('tool')}" for a in evaluation.get("required_actions") or []
                         if not a.get("invoked")]
        return "required tools not invoked", _clip("not invoked: " + ", ".join(missing_tools))
    if error_type == "Value Mismatch":
        wrong = [g for g in evaluation.get("goals") or [] if not g.get("ok")]
        return "value mismatch", _clip(
            f"{len(wrong)} reading(s) wrong: " + ", ".join(
                f"{g.get('room') or ''} {g.get('device') or ''}".strip() for g in wrong[:3])
            + f"; reply: {response}")
    return "other", _clip(f"{error_type or 'failed'}; outcome {record.get('outcome')}; reply: {response}")
