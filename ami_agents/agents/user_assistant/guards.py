"""Remembering the condition-guard decisions of a plan that ran successfully.

The InteractionSolver puts, on each plan entry it checked, `guards`: one
decision per step on a device with a condition
(`interaction_solver/utils/condition_guard.guards_wire`). After the plan
succeeds, the decisions worth keeping go to the EnvExplorer's guard store, so
the next plan on that action of that device skips the model:

    recorded    a guard the model made (`source: "model"`), still in the plan
                when it ran; a "no guard" the model made because the condition
                does not concern that command (`why: "not_applicable"`)
    never       a decision from memory (`source: "memory"`): already stored;
                a guard the plan did not need (`dropped`): it says nothing
                about the device; `why: "established_by_plan"`: it depends on
                that one plan; anything from a plan that did not succeed

Example -- "set washer 1's spin to 1200" succeeded, its entry carrying

    guards  [{"step": "S1", "key": {...washer_1 spinSpeedCurrent...},
              "protection": true, "source": "model",
              "check": {"title": "operationalState", ..., "operator": "!=", "value": 1},
              "action": {"title": "operationalState", ..., "value": 2},
              "restore": {...}}]
    sent    GUARD_RECORD_REQUEST {"guard": {key, protection, check, action}}
            -- the restore is not sent: it is derived again on every use.

Plan execution's whole involvement is one call, next to the signifiers:

    agent.add_behaviour(GuardRecordingBehaviour(context, outcome))   # after
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ...shared.models.messages import MessageType
from ...shared.utils.demo_log import demo
from ...shared.utils.logger import LoggerFactory
from ...shared.utils.spade_rpc import rpc_call

# The decision fields the guard store keeps (`GuardStore.record`).
RECORDED_FIELDS = ("key", "protection", "check", "action", "why")


def guards_to_record(entry: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The decisions of a successful plan entry worth remembering, each as
    the guard store takes it (see the module docstring for the rules)."""
    out = []
    for guard in (entry or {}).get("guards") or []:
        # 1. Only the model's decisions: memory hits are stored already.
        if not isinstance(guard, dict) or guard.get("source") != "model":
            continue
        # 2. A guard the plan ran with; a "no guard" about the device itself.
        if guard.get("protection") is True:
            if guard.get("dropped"):
                continue
        elif guard.get("why") != "not_applicable":
            continue
        out.append({k: guard[k] for k in RECORDED_FIELDS if k in guard})
    return out


class GuardRecordingBehaviour(OneShotBehaviour):
    """Send a successful plan's guard decisions to the EnvExplorer."""

    def __init__(self, context, outcome, logger=None):
        """``context`` is the plan's `ExecutionContext` (its entry carries the
        guards), ``outcome`` its `ExecutionOutcome`."""
        super().__init__()
        self.context = context
        self.outcome = outcome
        self.logger = logger or LoggerFactory.get_logger("UserAssistant")
        self.recorded: List[str] = []

    async def run(self) -> None:
        """Record each decision; best-effort, a failure is only logged."""
        # 1. Nothing is remembered from a plan that did not succeed.
        if not self.outcome.success:
            return
        guards = guards_to_record(self.context.entry)
        explorer_jid = self.agent.target_jids.get("explorer")
        if not guards or not explorer_jid:
            return

        # 2. One request per decision, on the plan's conversation thread.
        for guard in guards:
            try:
                response = await rpc_call(
                    self.agent, to_jid=str(explorer_jid),
                    request_type=MessageType.GUARD_RECORD_REQUEST.value,
                    body={"guard": guard},
                    expect_type=MessageType.GUARD_RECORD_RESPONSE.value,
                    timeout=self.agent.rpc_call_timeout,
                    thread=(self.context.thread
                            if self.context.thread != "__default__" else None))
                payload = json.loads(response.body or "{}")
                if isinstance(payload, dict) and payload.get("guard_id"):
                    self.recorded.append(payload["guard_id"])
            except Exception as exc:  # noqa: BLE001 - recording is best-effort
                self.logger.info(demo(f"Failed to record a guard decision: {exc}"))
        self.logger.info(demo(f"[MEMORY] Recorded {len(self.recorded)} guard decision(s) "
                              f"for plan={self.context.plan_id}"))
