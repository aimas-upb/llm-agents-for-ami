"""The profiling call: one small-model call answering a checklist.

Some facts a plan needs are not in the goal structure: which property tells a
cycle's remaining time, whether the next goal waits for a device to reach a
state, how a maintained value is kept. The planner lists them as a
checklist of questions; one call to the small (profiling) model answers them
all; each answer is checked, and a question without a usable answer gets its
fallback.

Used by the dependency planner, for its timing and maintenance questions.

Example -- "start the dryer so it finishes with the dishwasher":
    checklist  [{"id": "Q1", "question": "completion_timing", "target": "G1",
                 "ref": "P1"}]
    answer     {"answers": [{"id": "Q1", "answer": {"ref_remaining": {...},
                                                    "target_duration": {...}}}],
                "unresolved": []}
    profile    {"Q1": {...the item, "answer": {...}, "source": "model"}}
    trace      {"model": "gpt-5-nano", "questions": [...the checklist...],
                "answer": "<raw reply text>", "finish_reason": "stop"}

    ask_profile(agent, logger, structure, checklist, context, trace=None)
        -> (resolved profile, fallback log)
        The GOAL_PROFILE_PROMPT with the structure, the per-id context and the
        checklist; the answer resolved by `goal_profile.resolve_profile`. A
        failed call, or unusable output, puts every question on its fallback.
        InconsistentStructure (the model reported one) is raised to the caller.
        `trace`, when given, receives the call: the model, the questions sent,
        the model's raw answer text and why it stopped -- kept on the plan entry
        as `profile_call`, so each fallback can be traced to what was asked and
        what came back.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ....shared.models.goal_structure import GoalStructure
from ....shared.utils.demo_log import demo
from .dependency_context import known_properties
from .goal_profile import (
    InconsistentStructure,
    fallback_profile,
    render_goal_profile_prompt,
    resolve_profile,
)


# -- call helpers ------------------------------------------------------------

def _is_reasoning(model: str) -> bool:
    """True for reasoning models (o-series, gpt-5): they reject `temperature`
    and take `reasoning_effort` / `max_completion_tokens` instead."""
    return model.startswith("o") or model.startswith("gpt-5")


def _json_object(text: str) -> Dict[str, Any]:
    """The JSON object in the model's reply.

    Small models often wrap JSON in a ```json fence despite the prompt; the
    fence is tolerated. Raises (JSONDecodeError / ValueError) when the reply is
    not one JSON object -- the caller then falls back on every question."""
    # Strip a markdown fence: drop the opening line, cut at the closing one.
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    # Parse; a list or a bare value is not an answer.
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("the profile answer is not a JSON object")
    return data


# -- the profiling call ------------------------------------------------------

async def ask_profile(agent, logger, structure: GoalStructure,
                      checklist: List[Dict[str, Any]],
                      context: Dict[str, Any], trace: Optional[Dict[str, Any]] = None
                      ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Ask the profiling model ``checklist``; returns (resolved profile, log).

    The profile maps each checklist id to its item plus the answer used and its
    source ("model", or "fallback" with a reason); the log lists the fallbacks
    applied, for the plan's notes. A failed call or unreadable reply puts every
    question on its fallback -- profiling never stops a plan. Only a reported
    inconsistency (InconsistentStructure) is raised: the structure cannot be
    carried out, which the caller turns into an "impossible" reply."""
    trace = {} if trace is None else trace

    # 1. The prompt: the structure, each id's device context, the checklist.
    prompt = render_goal_profile_prompt(
        json.dumps(structure.to_wire_dict(), indent=1),
        json.dumps(context, indent=1), json.dumps(checklist, indent=1))
    # 2. The trace starts with what is asked; "answer" stays None if the call
    #    itself fails, so a failed call is told apart from an empty reply.
    cfg = agent.explicit_llm
    trace.update({"model": cfg.model, "questions": checklist, "answer": None})

    # 3. Model parameters: reasoning models take effort and a completion
    #    budget, the others a temperature (see `_is_reasoning`).
    kwargs: Dict[str, Any] = {}
    if _is_reasoning(cfg.model):
        if cfg.reasoning_effort:
            kwargs["reasoning_effort"] = cfg.reasoning_effort
        if cfg.max_completion_tokens:
            kwargs["max_completion_tokens"] = cfg.max_completion_tokens
    else:
        kwargs["temperature"] = cfg.temperature
    logger.info(demo(f"Profiling {structure.intent_text!r}: {len(checklist)} "
                     f"question(s) for {cfg.model}"))

    # 4. Call, keep the raw reply and why it stopped ("length" = ran out of
    #    tokens, so later answers are missing), then check every answer
    #    against its question and the devices of its context.
    try:
        response = await cfg.client.chat.completions.create(
            model=cfg.model, messages=[{"role": "user", "content": prompt}], **kwargs)
        choice = response.choices[0]
        trace["answer"] = choice.message.content or ""
        trace["finish_reason"] = getattr(choice, "finish_reason", None)
        output = _json_object(trace["answer"])
        profile, log = resolve_profile(checklist, output, known_properties(context))
    except InconsistentStructure:
        raise
    except Exception as exc:  # noqa: BLE001 - a bad call or bad output: fallbacks
        # 5. No usable reply at all: every question on its fallback.
        logger.warning(f"Profiling failed, every question on its fallback: {exc}")
        profile, log = fallback_profile(checklist, f"profiling failed: {exc}")

    # 6. Each fallback is logged, so a run log shows which answers were not used.
    for line in log:
        logger.info(demo(f"[PROFILE] {line}"))
    return profile, log
