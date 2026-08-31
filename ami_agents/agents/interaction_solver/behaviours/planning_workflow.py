"""Per-request planning workflow: orchestrates the four sub-Behaviours.

Spawned by ``GoalRequestBehaviour`` once per GOAL_REQUEST. It runs the
context queries, decides between the signifier-only fast path and the
LLM path, and writes the final reply onto ``self.reply_envelope``. The
caller (GoalRequestBehaviour) sends the SPADE reply, so the workflow
doesn't need access to the original message envelope itself.
"""

from typing import Any, Dict, List, Optional

from rdflib import Graph
from spade.behaviour import OneShotBehaviour

from ....shared.utils.demo_log import demo
from ....shared.utils.logger import LoggerFactory
from ....shared.models.intents import ImplicitGoalIntent, ExplicitGoalIntent
from ....shared.models.messages import MessageType
from ....shared.models.intents import Intent
from ..utils.plan_envelope import (
    envelope_error,
    envelope_llm_plan,
    envelope_llm_plans,
    envelope_signifier_reuse,
)
from ..utils.signifier_fast_path import (
    collect_signifier_ids,
    try_build_signifier_only_tree,
)
from ..utils.signifier_matching import merge_signifier_matches
from ....bt_planning import explicit_intent_handler
from .bt_plan_generation import BTPlanGenerationBehaviour
from .community_signifier_query import CommunitySignifierQueryBehaviour
from .env_context_query import EnvContextQueryBehaviour
from .implicit_goal_desire import ImplicitGoalDesireInferenceBehaviour
from .signifier_match_query import SignifierMatchQueryBehaviour


class PlanningWorkflowBehaviour(OneShotBehaviour):
    """Coordinate signifier matching, fast path, and LLM planning per request."""

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
        intent_strings = [i.to_query_string() for i in self.intents]
        intent_details = []
        for i in self.intents:
            if isinstance(i, ExplicitGoalIntent):
                intent_details.append(f"EXPLICIT({i.target.artifact_type}:{i.action.affordance_type}:{i.action.verb})")
            elif isinstance(i, ImplicitGoalIntent):
                intent_details.append(f"IMPLICIT({i.subtype}:{i.text_intent[:30]}...)" if len(i.text_intent) > 30 else f"IMPLICIT({i.subtype}:{i.text_intent})")
            else:
                intent_details.append(str(type(i).__name__))
        self.logger.info(
            demo(f"PlanningWorkflow starting: intents={intent_details} workspace_id={self.workspace_id!r}")
        )

        # Step 0 — Infer affected environment variables for implicit_intent subtypes
        self.logger.info(demo("Step 0: Inferring environment variables for implicit_intent subtypes"))
        affected_env_vars_by_intent: Dict[str, Optional[List[Dict[str, str]]]] = {}
        inferred_count = 0
        skipped_count = 0
        failed_count = 0
        inferred_vars = []

        for intent_obj in self.intents:
            intent_str = intent_obj.to_query_string()

            # Explicit intents: skip env var inference, skip signifier matching
            if isinstance(intent_obj, ExplicitGoalIntent):
                affected_env_vars_by_intent[intent_str] = None
                skipped_count += 1
                self.logger.debug(demo(f"Step 0: EXPLICIT intent ({intent_obj.target.artifact_type}:{intent_obj.action.affordance_type}), skipping env var inference"))
            # Implicit intents with subtype != "implicit_intent": skip env var inference, skip signifier matching
            elif isinstance(intent_obj, ImplicitGoalIntent) and intent_obj.subtype != "implicit_intent":
                affected_env_vars_by_intent[intent_str] = None
                skipped_count += 1
                self.logger.debug(demo(f"Step 0: IMPLICIT {intent_obj.subtype}, skipping env var inference: {intent_str}"))
            # Implicit intents with subtype == "implicit_intent": infer env vars
            elif isinstance(intent_obj, ImplicitGoalIntent) and intent_obj.subtype == "implicit_intent":
                self.logger.debug(demo(f"Step 0: Running env var inference for implicit_intent: {intent_str}"))
                inference = ImplicitGoalDesireInferenceBehaviour(
                    intent_text=intent_obj.text_intent,
                    logger=self.logger,
                )
                self.agent.add_behaviour(inference)
                await inference.join()

                if inference.error:
                    self.logger.warning(demo(f"Step 0: Env var inference failed for {intent_str}: {inference.error}"))
                    affected_env_vars_by_intent[intent_str] = None
                    failed_count += 1
                else:
                    env_vars = inference.result.get("affected_env_vars")
                    # Check if result is only unknown pairs
                    if env_vars == [{"variable": "unknown", "direction": "unknown"}]:
                        self.logger.debug(demo(f"Step 0: Env var inference found only unknown pairs for {intent_str}"))
                        affected_env_vars_by_intent[intent_str] = None
                    else:
                        formatted_vars = " | ".join([f"{v['variable']}({v['direction']})" for v in (env_vars or [])])
                        self.logger.info(demo(f"Step 0: Inferred env vars for {intent_str}: [{formatted_vars}]"))
                        affected_env_vars_by_intent[intent_str] = env_vars
                        inferred_vars.append(formatted_vars)
                        inferred_count += 1

        self.logger.info(demo(f"Step 0 complete: inferred={inferred_count} [{', '.join(inferred_vars) if inferred_vars else 'none'}] skipped={skipped_count} failed={failed_count}"))

        # Step 0.5 — EXPLICIT intent SPARQL path (before signifier matching)
        self.logger.info(demo("Step 0.5: Processing EXPLICIT intents via SPARQL path"))
        explicit_intents = [i for i in self.intents if isinstance(i, ExplicitGoalIntent)]
        implicit_intents = [i for i in self.intents if not isinstance(i, ExplicitGoalIntent)]
        explicit_results = {}
        # Track candidate affordances for ambiguous EXPLICIT intents
        explicit_intent_affordance_constraints: Dict[str, List[Dict[str, Any]]] = {}

        if explicit_intents:
            explicit_details = []
            for e in explicit_intents:
                explicit_details.append(
                    f"{e.target.workspace_type}::{e.target.artifact_type}::{e.action.affordance_type}[{e.action.verb}]"
                )
            self.logger.info(demo(f"Step 0.5: Found {len(explicit_intents)} EXPLICIT intent(s): [{', '.join(explicit_details)}]"))
            self.logger.debug(demo(f"Step 0.5: Fetching RDF graph from EnvExplorer"))
            graph = await self._get_environment_rdf_graph()
            self.logger.debug(demo(f"Step 0.5: RDF graph fetched, processing intents via SPARQL"))

            deterministic_count = 0
            ambiguous_count = 0
            impossible_count = 0
            deterministic_intents = []
            ambiguous_intents = []

            for intent in explicit_intents:
                intent_str = intent.to_query_string()
                intent_detail = f"{intent.target.artifact_type}@{intent.action.verb}"
                self.logger.debug(demo(f"Step 0.5: SPARQL query for {intent_detail}"))
                try:
                    result = await explicit_intent_handler.handle_explicit_intent(
                        intent, graph, logger=self.logger
                    )

                    if result.get("tree"):
                        # Deterministic BT succeeded
                        self.logger.info(demo(f"Step 0.5: {intent_detail} → DETERMINISTIC BT (source={result.get('source')})"))
                        explicit_results[intent_str] = result
                        deterministic_count += 1
                        deterministic_intents.append(intent_detail)
                    elif result.get("impossible"):
                        # Return error envelope
                        reason = result.get("reason", "Unknown reason")
                        self.logger.error(demo(f"Step 0.5: {intent_detail} → IMPOSSIBLE ({result.get('source')}): {reason}"))
                        self.reply_envelope = envelope_error(
                            "explicit_intent_impossible",
                            reason,
                            [intent],
                        )
                        impossible_count += 1
                        return
                    else:
                        # Ambiguous: fall back to LLM planning with filtered context
                        context = result.get("context", {})
                        candidates = len(context.get("candidate_affordances", []))
                        self.logger.info(demo(f"Step 0.5: {intent_detail} → AMBIGUOUS ({result.get('source')}, {candidates} candidates)"))
                        implicit_intents.append(intent)
                        explicit_results[intent_str] = result.get("context")
                        ambiguous_count += 1
                        ambiguous_intents.append(intent_detail)
                        # Store candidate affordances for context filtering later
                        candidate_affordances = context.get("candidate_affordances", [])
                        if candidate_affordances:
                            self.logger.debug(demo(f"Step 0.5: Storing {len(candidate_affordances)} candidate affordances for LLM filtering"))
                            explicit_intent_affordance_constraints[intent_str] = candidate_affordances

                except Exception as e:
                    self.logger.warning(demo(f"Step 0.5: {intent_detail} → SPARQL ERROR: {e}, falling back to implicit flow"))
                    implicit_intents.append(intent)

            self.logger.info(demo(f"Step 0.5 complete: deterministic={deterministic_count} [{', '.join(deterministic_intents)}] ambiguous={ambiguous_count} [{', '.join(ambiguous_intents)}] impossible={impossible_count}"))
        else:
            self.logger.debug(demo("Step 0.5: No EXPLICIT intents, skipping SPARQL path"))

        # Update intent_strings to reflect implicit-only list for signifier matching
        intent_strings = [i.to_query_string() for i in implicit_intents]

        # If no implicit intents remain and we have explicit results, build reply (early exit optimization)
        if not implicit_intents and explicit_results:
            # All intents were EXPLICIT and succeeded deterministically
            # Keep the result dicts intact (not just the trees) for envelope_llm_plans
            plans_by_intent = {k: v for k, v in explicit_results.items() if v.get("tree")}
            if plans_by_intent:
                self.logger.info(demo(f"All intents were EXPLICIT and resolved deterministically, skipping remaining steps"))
                self.reply_envelope = envelope_llm_plans(
                    plans_by_intent=plans_by_intent,
                    intents=explicit_intents,
                    workspace_id=self.workspace_id,
                )
                self.logger.info(demo(f"PlanningWorkflow complete: all EXPLICIT intents produced deterministic BTs"))
                return

        # Step 1 — signifier matching (local + community in parallel).
        # For intents with affected_env_vars, use v3 matching; others skip matching
        matching_intents = [i for i in implicit_intents if i.to_query_string() in intent_strings]
        self.logger.info(demo(f"Step 1: Signifier matching for {len(matching_intents)} intent(s) (v3={len([i for i in matching_intents if isinstance(i, ImplicitGoalIntent) and i.to_query_string() in affected_env_vars_by_intent and affected_env_vars_by_intent.get(i.to_query_string())])} with env_vars)"))
        merged_matches = await self._gather_signifier_matches(
            intent_strings,
            affected_env_vars_by_intent=affected_env_vars_by_intent,
        )
        self.logger.debug(demo(f"Step 1: Signifier matching complete, analyzing results"))
        self._log_signifier_search(merged_matches)

        # Count results by type
        exact_count = 0
        hint_count = 0
        error_count = 0
        for payload in merged_matches.values():
            if isinstance(payload, dict):
                if payload.get("error"):
                    error_count += 1
                else:
                    exact = payload.get("exact_matches", [])
                    hints = payload.get("affordance_hints", [])
                    exact_count += len(exact) if isinstance(exact, list) else 0
                    hint_count += len(hints) if isinstance(hints, list) else 0
        self.logger.info(demo(f"Step 1 complete: exact_matches={exact_count} affordance_hints={hint_count} errors={error_count}"))

        # Step 2 — try the signifier-only fast path.
        self.logger.info(demo(f"Step 2: Attempting signifier-only fast path"))
        fast_tree = try_build_signifier_only_tree(merged_matches, self.intents)
        if fast_tree is not None:
            self.logger.info(
                demo(
                    "Step 2: SUCCESS - BT recovered from signifiers (no EnvExplorer context queries, no LLM)"
                )
            )
            # Populate affected_env_vars back into intent objects for storage in signifiers
            for intent_obj in self.intents:
                if isinstance(intent_obj, ImplicitGoalIntent):
                    intent_str = intent_obj.to_query_string()
                    if intent_str in affected_env_vars_by_intent:
                        intent_obj.affected_env_vars = affected_env_vars_by_intent[intent_str]

            self.reply_envelope = envelope_signifier_reuse(
                tree=fast_tree,
                signifier_ids=collect_signifier_ids(merged_matches, self.intents),
                intents=self.intents,
                workspace_id=self.workspace_id,
            )
            self.logger.info(demo(f"PlanningWorkflow complete: signifier-only fast path succeeded"))
            return
        else:
            self.logger.info(demo(f"Step 2: Fast path did not produce a tree, proceeding to LLM path"))

        # Step 3 — LLM path: gather context, then generate.
        self.logger.info(demo(f"Step 3: Gathering environment context from EnvExplorer (workspace_id={self.workspace_id!r})"))
        ctx = EnvContextQueryBehaviour(
            workspace_id=self.workspace_id, logger=self.logger
        )
        self.agent.add_behaviour(ctx)
        await ctx.join()
        if ctx.error:
            self.logger.error(demo(f"Step 3: Context gathering failed: {ctx.error}"))
            # Populate affected_env_vars even on error for complete intent structure
            for intent_obj in self.intents:
                if isinstance(intent_obj, ImplicitGoalIntent):
                    intent_str = intent_obj.to_query_string()
                    if intent_str in affected_env_vars_by_intent:
                        intent_obj.affected_env_vars = affected_env_vars_by_intent[intent_str]

            self.reply_envelope = envelope_error(
                "context_gathering_failed",
                ctx.error,
                self.intents,
            )
            return

        self.logger.info(demo(f"Step 3: Context gathered - affordances={len(ctx.affordances)} artifact_count={len(ctx.state_payload.get('artifacts', {})) if isinstance(ctx.state_payload, dict) else 0}"))

        # Step 4 — Plan each atomic intent separately
        self.logger.info(demo(f"Step 4: LLM-based planning for {len(self.intents)} intent(s)"))
        plans_by_intent = {}
        plan_success_count = 0
        plan_error_count = 0
        successful_intents = []
        failed_intents = []

        for intent_obj in self.intents:
            intent_str = intent_obj.to_query_string()
            intent_type_str = "EXPLICIT" if isinstance(intent_obj, ExplicitGoalIntent) else "IMPLICIT"
            intent_label = f"{intent_type_str}({intent_str[:40]}...)" if len(intent_str) > 40 else f"{intent_type_str}({intent_str})"

            # Convert affected_env_vars to only populated entries for BT planner
            affected_env_vars_for_planner = {}
            has_env_vars = False
            if intent_str in affected_env_vars_by_intent and affected_env_vars_by_intent[intent_str] is not None:
                affected_env_vars_for_planner[intent_str] = affected_env_vars_by_intent[intent_str]
                has_env_vars = True

            # Filter affordances for constrained EXPLICIT intents (ambiguous cases)
            # Keep entire artifacts (both action AND property affordances)
            affordances_for_planner = ctx.affordances
            affordance_filtering_applied = False
            if intent_str in explicit_intent_affordance_constraints:
                candidate_affordances = explicit_intent_affordance_constraints[intent_str]
                # Extract artifact URIs from SPARQL results
                candidate_artifact_uris = {
                    str(a.get("artifact", "")).strip()
                    for a in candidate_affordances
                    if a.get("artifact")
                }
                if candidate_artifact_uris:
                    # Filter to include all affordances (action + property) from candidate artifacts
                    # artifact_id in ctx.affordances is an RDF URI that matches SPARQL ?artifact
                    affordances_for_planner = [
                        a for a in ctx.affordances
                        if str(a.get("artifact_id", "")).strip() in candidate_artifact_uris
                    ]
                    affordance_filtering_applied = True
                    self.logger.info(
                        demo(
                            f"Step 4: {intent_label} filtered affordances: "
                            f"{len(affordances_for_planner)} out of {len(ctx.affordances)} "
                            f"(candidate artifacts: {len(candidate_artifact_uris)})"
                        )
                    )

            signifier_hints_count = 0
            if merged_matches and isinstance(merged_matches, dict):
                for payload in merged_matches.values():
                    if isinstance(payload, dict):
                        exact = payload.get("exact_matches", [])
                        hints = payload.get("affordance_hints", [])
                        signifier_hints_count += len(exact) if isinstance(exact, list) else 0
                        signifier_hints_count += len(hints) if isinstance(hints, list) else 0

            self.logger.debug(demo(f"Step 4: {intent_label} BT planner input: affordances={len(affordances_for_planner)} env_vars={has_env_vars} signifier_hints={signifier_hints_count} filtered={affordance_filtering_applied}"))

            plan_gen = BTPlanGenerationBehaviour(
                intent_strings=[intent_str],  # Plan single intent at a time
                affordances=affordances_for_planner,
                state=(
                    ctx.state_payload.get("state")
                    if isinstance(ctx.state_payload, dict)
                    else ctx.state_payload
                ),
                signifier_hints=merged_matches,
                affected_env_vars=affected_env_vars_for_planner if affected_env_vars_for_planner else None,
                logger=self.logger,
            )
            self.agent.add_behaviour(plan_gen)
            await plan_gen.join()

            if plan_gen.error:
                self.logger.error(demo(f"Step 4: {intent_label} planning failed: {plan_gen.error}"))
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

            self.logger.info(demo(f"Step 4: {intent_label} planning succeeded (nodes={node_count})"))
            plans_by_intent[intent_str] = bt_result
            successful_intents.append(f"{intent_label}[{node_count}]")
            plan_success_count += 1

        self.logger.info(demo(f"Step 4 complete: successful={plan_success_count} [{', '.join(successful_intents)}] failed={plan_error_count} [{', '.join(failed_intents)}]"))

        # Populate affected_env_vars back into intent objects for storage in signifiers
        for intent_obj in self.intents:
            if isinstance(intent_obj, ImplicitGoalIntent):
                intent_str = intent_obj.to_query_string()
                if intent_str in affected_env_vars_by_intent:
                    intent_obj.affected_env_vars = affected_env_vars_by_intent[intent_str]
        self.logger.debug(demo(f"Step 4: Populated affected_env_vars for {len([i for i in self.intents if isinstance(i, ImplicitGoalIntent)])} implicit intent(s)"))

        # Combine all plans into a single tree with parallel execution
        self.logger.info(demo(f"Step 4: Combining {len(plans_by_intent)} LLM-generated plan(s) with parallel execution"))
        self.reply_envelope = envelope_llm_plans(
            plans_by_intent=plans_by_intent,
            intents=self.intents,
            workspace_id=self.workspace_id,
        )
        self.logger.info(demo(f"PlanningWorkflow complete: {len(plans_by_intent)} plan(s) generated successfully via LLM path"))

    async def _gather_signifier_matches(
        self,
        intent_strings: List[str],
        affected_env_vars_by_intent: Optional[Dict[str, Optional[List[Dict[str, str]]]]] = None,
    ) -> Dict[str, Any]:
        """Spawn local signifier behaviour and optionally community signifier behaviour.

        Community signifier matching is only queried AFTER local matching if:
        1. There is a community client available
        2. At least one intent is an implicit goal with subtype == "implicit_intent"
        3. Local matching found no (or weak) matches for those implicit intents

        Explicit intents never trigger community queries (they are fully specified).
        """
        # Only pass affected_env_vars if they are populated (for v3 matching)
        affected_env_vars_for_matching = {
            intent_str: env_vars
            for intent_str, env_vars in (affected_env_vars_by_intent or {}).items()
            if env_vars is not None
        } if affected_env_vars_by_intent else None

        local_b = SignifierMatchQueryBehaviour(
            intents=self.intents,
            workspace_id=self.workspace_id,
            affected_env_vars=affected_env_vars_for_matching,
            logger=self.logger,
        )
        self.agent.add_behaviour(local_b)

        # Wait for local matching to complete first
        await local_b.join()
        local_matches = local_b.matches

        # Decide whether to query community based on:
        # - Local matching results for implicit_intent subtypes
        # - Availability of community client
        community_b: Optional[CommunitySignifierQueryBehaviour] = None
        implicit_intents_needing_community = []

        if getattr(self.agent, "community_client", None) is not None:
            # Check which implicit_intent subtypes had insufficient local matches
            for intent in self.intents:
                if isinstance(intent, ImplicitGoalIntent) and intent.subtype == "implicit_intent":
                    intent_str = intent.to_query_string()
                    match_data = local_matches.get(intent_str, {})
                    exact_matches = match_data.get("exact_matches", [])
                    # If no exact matches, consult community
                    if not exact_matches:
                        implicit_intents_needing_community.append(intent_str)

            # Only spawn community behaviour if there are implicit intents without local matches
            if implicit_intents_needing_community:
                community_b = CommunitySignifierQueryBehaviour(
                    intent_strings=implicit_intents_needing_community, logger=self.logger
                )
                self.agent.add_behaviour(community_b)
                await community_b.join()

        return merge_signifier_matches(
            local_matches,
            community_b.matches if community_b is not None else {},
            intent_strings,
        )

    def _log_signifier_search(self, results: Dict[str, Any]) -> None:
        """Demo-friendly per-intent summary of signifier search outcomes."""
        for intent, payload in results.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("error"):
                self.logger.info(
                    demo(f"Signifier search failed: intent={intent!r} error={payload.get('error')}")
                )
                continue

            exact = payload.get("exact_matches") if isinstance(payload.get("exact_matches"), list) else []
            hints = payload.get("affordance_hints") if isinstance(payload.get("affordance_hints"), list) else []
            total = payload.get("total_signifiers")
            community_count = sum(
                1 for m in exact + hints if isinstance(m, dict) and m.get("source") == "community"
            )

            if exact:
                top_signifier = exact[0].get("signifier_id", "?") if exact else "?"
                self.logger.info(
                    demo(
                        f"Signifier search: intent={intent!r} exact={len(exact)} affordance_hints={len(hints)} "
                        f"(community={community_count}) top={top_signifier}"
                    )
                )
            else:
                self.logger.info(
                    demo(
                        f"Signifier search: intent={intent!r} exact=0 affordance_hints={len(hints)} "
                        f"(community={community_count}) stored_total={total if total is not None else '?'}"
                    )
                )

    async def _get_environment_rdf_graph(self) -> Graph:
        """
        Fetch RDF graph from EnvExplorer using ENV_CAPABILITIES_REQUEST with detail_level="detailed".
        Cache it to avoid redundant RPC calls within a single planning workflow.
        """
        if not hasattr(self, "_cached_rdf_graph"):
            try:
                import json

                # Use agent's _query_env_explorer which handles response parsing correctly
                response_body = await self.agent._query_env_explorer(
                    message_type=MessageType.ENV_CAPABILITIES_REQUEST.value,
                    body={"detail_level": "detailed"},
                    expect_type=MessageType.ENV_CAPABILITIES_RESPONSE.value,
                )

                # Response is a JSON object with payload field containing RDF Turtle
                rdf_ttl = ""
                try:
                    response_data = json.loads(response_body)
                    if isinstance(response_data, dict):
                        rdf_ttl = response_data.get("payload", "")
                except (json.JSONDecodeError, ValueError):
                    # If not JSON, assume it's raw Turtle
                    rdf_ttl = response_body

                self._cached_rdf_graph = Graph()

                if rdf_ttl:
                    # Log first and last 300 chars to see prefix declarations and content
                    if len(rdf_ttl) > 600:
                        self.logger.info(demo(f"RDF start: {rdf_ttl[:300]}"))
                        self.logger.info(demo(f"RDF end: {rdf_ttl[-300:]}"))
                    else:
                        self.logger.info(demo(f"RDF content: {rdf_ttl}"))

                    self._cached_rdf_graph.parse(data=rdf_ttl, format="turtle")
                    self.logger.info(demo(f"RDF graph parsed successfully"))

            except Exception as e:
                self.logger.error(demo(f"Failed to fetch RDF graph: {e}"))
                self._cached_rdf_graph = Graph()

        return self._cached_rdf_graph

