"""Per-request planning workflow: context, then one LLM plan per intent.

Spawned by ``GoalRequestBehaviour`` for what the structured-goal paths do not
take (dependency structures, maintenance goals, plain-text intents). It gathers
the environment context from EnvExplorer, plans each intent with the general
LLM planner, and writes the reply onto ``self.reply_envelope``. The caller
sends the SPADE reply.

Memory is not consulted here: it is recalled by exact key on the
structured-goal paths (`utils/signifier_memory.py`).
"""

from typing import Any, Dict, List, Optional

from spade.behaviour import OneShotBehaviour

from ....shared.models.intents import Intent
from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ..utils.plan_envelope import envelope_error, envelope_llm_plans
from .bt_plan_generation import BTPlanGenerationBehaviour
from .env_context_query import EnvContextQueryBehaviour


class PlanningWorkflowBehaviour(OneShotBehaviour):
    """Coordinate context gathering and LLM planning per request."""

    def __init__(
        self,
        intents: List[Intent],
        workspace_id: Optional[str] = None,
        logger=None,
    ) -> None:
        super().__init__()
        self.intents = intents
        self.workspace_id = workspace_id
        self.logger = logger or LoggerFactory.get_logger("InteractionSolver")
        # Final response envelope, set by ``run``.
        self.reply_envelope: Optional[Dict[str, Any]] = None

    async def run(self) -> None:
        self.logger.info(demo(
            f"PlanningWorkflow starting: {len(self.intents)} intent(s) "
            f"workspace_id={self.workspace_id!r}"))

        # 1. The environment context.
        self.logger.info(demo(f"Step 1: Gathering environment context from EnvExplorer (workspace_id={self.workspace_id!r})"))
        ctx = EnvContextQueryBehaviour(
            workspace_id=self.workspace_id, logger=self.logger
        )
        self.agent.add_behaviour(ctx)
        await ctx.join()
        if ctx.error:
            self.logger.error(demo(f"Step 1: Context gathering failed: {ctx.error}"))
            self.reply_envelope = envelope_error(
                "context_gathering_failed",
                ctx.error,
                self.intents,
            )
            return

        self.logger.info(demo(f"Step 1: Context gathered - affordances={len(ctx.affordances)} artifact_count={len(ctx.state_payload.get('artifacts', {})) if isinstance(ctx.state_payload, dict) else 0}"))

        # 2. Plan each atomic intent separately.
        self.logger.info(demo(f"Step 2: LLM-based planning for {len(self.intents)} intent(s)"))
        plans_by_intent = {}
        plan_success_count = 0
        plan_error_count = 0
        successful_intents = []
        failed_intents = []

        for intent_obj in self.intents:
            intent_str = intent_obj.to_query_string()
            intent_label = f"({intent_str[:40]}...)" if len(intent_str) > 40 else f"({intent_str})"
            self.logger.debug(demo(f"Step 2: {intent_label} BT planner input: "
                                   f"affordances={len(ctx.affordances)}"))

            plan_gen = BTPlanGenerationBehaviour(
                intent_strings=[intent_str],  # Plan single intent at a time
                affordances=ctx.affordances,
                state=(
                    ctx.state_payload.get("state")
                    if isinstance(ctx.state_payload, dict)
                    else ctx.state_payload
                ),
                signifier_hints=None,
                affected_env_vars=None,
                logger=self.logger,
            )
            self.agent.add_behaviour(plan_gen)
            await plan_gen.join()

            if plan_gen.error:
                self.logger.error(demo(f"Step 2: {intent_label} planning failed: {plan_gen.error}"))
                self.reply_envelope = envelope_error(
                    "plan_generation_failed",
                    plan_gen.error,
                    [intent_obj],
                )
                failed_intents.append(intent_label)
                plan_error_count += 1
                return

            # Extract BT depth/node count for logging
            bt_result = plan_gen.result
            node_count = "?"
            if isinstance(bt_result, dict):
                # Count nodes in the BT
                def count_nodes(node):
                    if not isinstance(node, dict):
                        return 0
                    count = 1
                    if "children" in node and isinstance(node["children"], list):
                        count += sum(count_nodes(child) for child in node["children"])
                    return count
                node_count = count_nodes(bt_result)

            self.logger.info(demo(f"Step 2: {intent_label} planning succeeded (nodes={node_count})"))
            plans_by_intent[intent_str] = bt_result
            successful_intents.append(f"{intent_label}[{node_count}]")
            plan_success_count += 1

        self.logger.info(demo(f"Step 2 complete: successful={plan_success_count} [{', '.join(successful_intents)}] failed={plan_error_count} [{', '.join(failed_intents)}]"))

        # Combine all plans into a single tree with parallel execution
        self.logger.info(demo(f"Step 2: Combining {len(plans_by_intent)} LLM-generated plan(s) with parallel execution"))
        self.reply_envelope = envelope_llm_plans(
            plans_by_intent=plans_by_intent,
            intents=self.intents,
            workspace_id=self.workspace_id,
        )
        self.logger.info(demo(f"PlanningWorkflow complete: {len(plans_by_intent)} plan(s) generated successfully via LLM path"))
