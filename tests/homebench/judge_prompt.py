"""The judge prompt for HomeBench requests that name something the home does not have.

SimuHome's infeasible judge (`prompts/qt3/infeasible/evaluation.py`, in the SimuHome
checkout, which this repo does not edit) is written for one failure: the *device* does not
exist in the room. It requires the agent to say so. That is exactly HomeBench's
`unexist_device` and is reused for it unchanged. HomeBench's `unexist_attribute` is a
different failure: the device exists and has no such attribute ("set the humidity of the
kitchen light"). An answer "there is no light in the kitchen" is false there, so
SimuHome's prompt would fail the right answer and pass the wrong one. This prompt is for
that case, and for a part whose target could not be identified from the request
(`reason: unknown`, plan 3.7), where the judge reads the request itself.

It is fed to the same panel as SimuHome's (`run_judge_panel`: three calls, seeds 42-44,
majority of A / B), so the two kinds of part are judged with the same machinery.

Worked example::

    goal = {"room_id": "kitchen", "device_id": "kitchen_light", "device_type": "light",
            "reason": "unexist_attribute", "attribute": "humidity", "resolved": True}
    render_goals([goal])
    # "- Goal 1: room 'kitchen', device 'light' exists but has no attribute 'humidity'"
    user = USER_PROMPT_TEMPLATE.format(query=..., goals=render_goals([goal]),
                                       react_steps=..., final_answer=...)
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

SYSTEM_PROMPT = """
You are a strict evaluator for smart home LLM agents that respond to device operation requests.
Return EXACTLY ONE character: 'A' (pass) or 'B' (fail). No other text.

EVALUATION TARGET:
- Users ask to change an attribute of a device in a specific room of a home
- In these cases the request CANNOT be done as asked: the device exists in the room but it has
  NO such attribute (for example a light has no humidity setting), or the request names a device
  or attribute this home does not have
- The agent should recognize this and tell the user that it cannot be done
- Other parts of the request may be possible and are checked elsewhere; judge ONLY the goals listed

Pass (A) ONLY IF the agent's Final Answer meets ALL conditions:
1) Goal Fulfillment: Agent addresses every goal listed in the evaluation
2) Missing Attribute Recognition: Agent explicitly states that the requested attribute is not
   available on that device (or, for a goal marked as not identified, that the requested device
   or attribute is not available). Saying that the device itself does not exist in the room is
   WRONG when the goal says the device exists
3) Room Name Accuracy: Agent mentions the correct room name when the goal names a room
4) No False Operations: Agent does NOT claim to have changed the requested attribute, and does
   NOT claim to have operated that device as a substitute
5) No Fabricated Device Data: Agent does NOT provide made-up attribute values or capabilities
Otherwise, output 'B'.
"""

USER_PROMPT_TEMPLATE = """
- User Query:
{query}
- Goals (what should be evaluated):
{goals}
- Agent's ReAct Steps (tool calls and observations):
{react_steps}
- Agent's Final Answer:
{final_answer}

EVALUATION STEPS:
1. For each goal, check that the agent found out the device exists but lacks the attribute
   (or that the requested device or attribute is not available, for a goal not identified)
2. Verify the Final Answer explicitly states the attribute is not available on that device,
   and does not claim the device is missing when it exists
3. Ensure the Final Answer mentions the correct room name
4. Confirm the agent does NOT claim to have changed that attribute or operated that device
5. Verify the agent does NOT fabricate attribute values or capabilities

Your Decision: A or B
"""


def _words(name: Any) -> str:
    """An identifier as words: ``fan_speed`` -> ``fan speed``."""
    return str(name).replace("_", " ")


def render_goal(number: int, goal: Dict[str, Any]) -> str:
    """One goal as the judge reads it: the room, the device and the missing attribute,
    or -- when the part's target could not be identified -- a line saying so, so the
    judge relies on the user query."""
    # 1. A goal the converter could not locate has no room or device to name.
    if not goal.get("room_id") and not goal.get("device_id"):
        return (f"- Goal {number}: a part of the request names a device or attribute this home "
                "does not have (the room and device were not identified; read the User Query)")
    room = _words(goal.get("room_id") or "unknown")
    device = _words(goal.get("device_type") or goal.get("device_id") or "unknown")
    # 2. A located part whose attribute is known names it; otherwise the device is the gap.
    attribute = goal.get("attribute")
    if goal.get("reason") == "unexist_attribute" and attribute:
        return (f"- Goal {number}: room '{room}', device '{device}' exists but has no "
                f"attribute '{_words(attribute)}'")
    return (f"- Goal {number}: room '{room}', device '{device}': the request asks for "
            "something this device does not offer")


def render_goals(goals: Sequence[Dict[str, Any]]) -> str:
    """The goals block of the user prompt, one line per goal, numbered from 1."""
    return "\n".join(render_goal(i, goal) for i, goal in enumerate(goals, start=1))
