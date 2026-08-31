"""
The UA's request pipeline — the LLM and RPC steps, independent of SPADE.

These were methods on `UserMessageBehaviour`, which had grown to own the whole
request lifecycle: receive, segment, parse, plan, summarise, confirm, execute.
Splitting the *steps* from the *orchestration* is what lets the orchestration
become a per-request FSM without dragging 1400 lines along, and it removes the
duplicated pipeline that `run()` and `_handle_confirmation` had each grown a
copy of.

Everything here is a free function taking `agent` and `logger` explicitly, so a
`State`, a `CyclicBehaviour`, or a test can call it without inheriting
anything. Nothing in this module sends, receives, or touches conversation
state — that stays with the behaviours.
"""

from __future__ import annotations

import json

from ...shared.models.messages import MessageType
from ...shared.utils.spade_rpc import rpc_call
from ...shared.utils.demo_log import demo
from ...shared.utils.logger import LoggerFactory
from ...shared.utils.namespaces import domain_types
from .models import AtomicIntent
from .prompts import (
    ATOMIC_SEGMENTATION_SYSTEM_PROMPT,
    ENV_CAPABILITIES_REQUEST_PARSER_PROMPT,
    ENV_CAPABILITIES_RESPONSE_PROMPT,
    ENV_STATE_REQUEST_PARSER_PROMPT,
    GOAL_REQUEST_PARSER_PROMPT,
    PLAN_SUMMARY_SYSTEM_PROMPT,
    QUERY_RESPONSE_SYSTEM_PROMPT,
)
from .utils import loose_json_loads
from .utils.llm_client import build_behaviour_llm_client, build_llm_call_kwargs

# The two `_filter_capabilities_json*` helpers are module-level and predate the
# per-call logger, so they keep using the module logger they were written with.
logger = LoggerFactory.get_logger("UserAssistant")


def filter_capabilities_json(caps_dict: dict) -> dict:
    """Filter capabilities JSON to reduce size for ENV_CAPABILITIES_RESPONSE_PROMPT.

    Rules:
    1. Omit 'description' if empty
    2. Omit 'form' from action/property affordances
    3. Exclude action affordances without ex: semantic types
    4. Omit output_schema from property affordances
    5. Omit input_schema from action affordances
    6. Omit parameters if empty

    Args:
        caps_dict: Hierarchical capabilities dict from EnvExplorer

    Returns:
        Filtered dict with reduced size
    """
    filtered = {}

    # Copy top-level fields
    for key in ("discovery_complete",):
        if key in caps_dict:
            filtered[key] = caps_dict[key]

    # Filter workspaces
    filtered["workspaces"] = []
    for ws in (caps_dict.get("workspaces") or []):
        filtered_ws = {
            k: v for k, v in ws.items()
            if k not in ("form",) and not (k == "description" and not v)
        }

        # Filter artifacts
        filtered_ws["artifacts"] = []
        for artifact in (ws.get("artifacts") or []):
            filtered_artifact = {
                k: v for k, v in artifact.items()
                if k not in ("form",) and not (k == "description" and not v)
            }

            # Filter affordances
            filtered_artifact["affordances"] = []
            for aff in (artifact.get("affordances") or []):
                aff_type = aff.get("type", "")

                # Rule 3: For action affordances, keep ONLY those with ex: namespace semantic types
                if aff_type == "action_affordance":
                    semantic_types = aff.get("semantic_types", [])
                    # Keep only home-ontology types (the device/room vocabulary
                    # the user reasons about), not protocol types like td:Thing.
                    ex_types = domain_types(semantic_types)
                    logger.debug(
                        f"Action affordance {aff.get('name')}: semantic_types={semantic_types}, ex_types={ex_types}"
                    )
                    if not ex_types:
                        logger.debug(
                            f"Filtering out action affordance {aff.get('name')} (no ex: semantic types)"
                        )
                        continue

                # Build filtered affordance
                filtered_aff = {}

                # Copy all fields except form, schemas, parameters, and empty descriptions
                for key, val in aff.items():
                    if key == "form":
                        continue  # Rule 2
                    elif key == "output_schema" and aff_type == "property_affordance":
                        continue  # Rule 4
                    elif key == "input_schema" and aff_type == "action_affordance":
                        continue  # Rule 5
                    elif key == "parameters":
                        # Rule 6: Omit if empty
                        if val and len(val) > 0:
                            filtered_aff[key] = val
                    elif key == "description" and not val:
                        # Rule 1: Omit if empty
                        continue
                    else:
                        filtered_aff[key] = val

                filtered_artifact["affordances"].append(filtered_aff)

            filtered_ws["artifacts"].append(filtered_artifact)

        filtered["workspaces"].append(filtered_ws)

    return filtered


def filter_capabilities_json_for_state(caps_dict: dict) -> dict:
    """Filter capabilities JSON for ENV_STATE_REQUEST parsing.

    Rules:
    1. Include ONLY property affordances (exclude all actions)
    2. Omit 'form', 'output_schema'
    3. Include 'description' only if non-empty
    4. Omit 'parameters' if empty

    Purpose: Minimal property list for state-query parser to align property names to artifact/property URIs.

    Args:
        caps_dict: Hierarchical capabilities dict from EnvExplorer

    Returns:
        Filtered dict containing only properties (no actions, no schemas/forms)
    """
    filtered = {}

    # Copy top-level fields
    for key in ("discovery_complete",):
        if key in caps_dict:
            filtered[key] = caps_dict[key]

    # Filter workspaces
    filtered["workspaces"] = []
    for ws in (caps_dict.get("workspaces") or []):
        filtered_ws = {
            k: v for k, v in ws.items()
            if k != "form" and not (k == "description" and not v)
        }

        # Filter artifacts
        filtered_ws["artifacts"] = []
        for artifact in (ws.get("artifacts") or []):
            filtered_artifact = {
                k: v for k, v in artifact.items()
                if k != "form" and not (k == "description" and not v)
            }

            # Filter affordances: ONLY property affordances (no actions)
            filtered_artifact["affordances"] = []
            for aff in (artifact.get("affordances") or []):
                aff_type = aff.get("type", "")

                # Skip action affordances entirely
                if aff_type != "property_affordance":
                    continue

                # Build filtered property affordance
                filtered_aff = {}

                # Copy fields except: form, output_schema, and empty description
                for key, val in aff.items():
                    if key == "form":
                        continue  # Omit form
                    elif key == "output_schema":
                        continue  # Omit output schema
                    elif key == "description" and not val:
                        continue  # Omit empty description
                    elif key == "parameters":
                        # Only include if non-empty
                        if val and len(val) > 0:
                            filtered_aff[key] = val
                    else:
                        filtered_aff[key] = val

                filtered_artifact["affordances"].append(filtered_aff)

            filtered_ws["artifacts"].append(filtered_artifact)

        filtered["workspaces"].append(filtered_ws)

    return filtered


# ============================================================================
# UserMessageBehaviour class
# ============================================================================


async def segment_into_atomic_intents(agent, logger, user_text: str, capabilities_ctx: str = "") -> list[AtomicIntent]:
    """Call LLM to segment user message into atomic intents with categories."""
    logger.info(demo(f"[LLM CALL] Calling ATOMIC_SEGMENTATION_SYSTEM_PROMPT for: {user_text[:100]!r}"))
    prompt = ATOMIC_SEGMENTATION_SYSTEM_PROMPT.format(capabilities=capabilities_ctx)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_text},
    ]
    try:
        response = await agent.llm_client.chat.completions.create(
            model=agent.llm_model,
            messages=messages,
            **agent.build_llm_kwargs(),
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = loose_json_loads(raw)
        if not isinstance(parsed, dict):
            logger.warning("LLM atomic segmentation returned non-dict: %r", raw[:200])
            return []

        intents_data = parsed.get("intents", [])
        if not isinstance(intents_data, list):
            logger.warning("LLM atomic segmentation returned non-list intents")
            return []

        result = []
        for item in intents_data:
            if not isinstance(item, dict):
                continue
            span = item.get("span", "").strip()
            category = item.get("category", "").strip()
            reason = item.get("reason", "").strip()
            if span and category in ("GOAL_REQUEST", "ENV_STATE_REQUEST", "ENV_CAPABILITIES_REQUEST") and reason:
                result.append(AtomicIntent(span=span, category=category, reason=reason))
        return result
    except Exception as exc:
        logger.error("LLM atomic segmentation failed: %s", exc)
        return []


def build_capabilities_hierarchical_text(caps_summary: dict) -> str:
    """Build human-readable hierarchical text from capabilities summary JSON.

    Filters ACTION affordances to only show those with ex: semantic types,
    but includes ALL PROPERTY affordances. Delegates to format_capabilities_hierarchical_text()
    from env_explorer's data_formatting module, which handles the full formatting.

    Args:
        caps_summary: Hierarchical dict from format_capabilities_summary_hierarchical()

    Returns:
        Human-readable hierarchical string
    """
    if not caps_summary or not caps_summary.get("discovery_complete"):
        return "Environment discovery not yet complete."

    lines = []

    def format_workspace(ws_node: dict, indent: str = "") -> None:
        """Recursively format a workspace and its contents."""
        ws_name = ws_node.get("name", "Unknown")
        lines.append(f"{indent}Workspace: {ws_name}")
        semantic_types = ws_node.get("semantic_types", [])
        if semantic_types:
            lines.append(f"{indent}  semantic_types: {', '.join(semantic_types)}")
        description = ws_node.get("description", "")
        if description:
            lines.append(f"{indent}  description: {description}")

        # Format artifacts in this workspace
        for artifact in (ws_node.get("artifacts") or []):
            artifact_name = artifact.get("name", "Unknown")
            lines.append(f"{indent}  Artifact: {artifact_name}")
            artifact_types = artifact.get("semantic_types", [])
            if artifact_types:
                lines.append(f"{indent}    semantic_types: {', '.join(artifact_types)}")
            artifact_desc = artifact.get("description", "")
            if artifact_desc:
                lines.append(f"{indent}    description: {artifact_desc}")

            # Format affordances: ACTION only if has ex: types, PROPERTY always
            for aff in (artifact.get("affordances") or []):
                aff_type = aff.get("type", "").replace("_affordance", "")

                # For ACTION affordances, only include if they have ex: semantic types
                if aff_type == "action":
                    semantic_types = aff.get("semantic_types", [])
                    homeont_types = domain_types(semantic_types)
                    if not homeont_types:
                        continue  # Skip actions without homeont types

                aff_name = aff.get("name", "Unknown")
                lines.append(f"{indent}    {aff_type}: {aff_name}")

                # Include semantic types if present
                aff_semantic_types = aff.get("semantic_types", [])
                if aff_semantic_types:
                    lines.append(f"{indent}      semantic_types: {', '.join(aff_semantic_types)}")

                # Include description if present
                aff_desc = aff.get("description", "")
                if aff_desc:
                    lines.append(f"{indent}      description: {aff_desc}")

                # Include parameters if present
                parameters = aff.get("parameters", [])
                if parameters:
                    lines.append(f"{indent}      parameters: {', '.join(parameters)}")

        # Format sub-workspaces
        for sub_ws in (ws_node.get("sub_workspaces") or []):
            format_workspace(sub_ws, indent + "  ")

    # Process all root workspaces
    for ws in (caps_summary.get("workspaces") or []):
        format_workspace(ws)

    return "\n".join(lines) if lines else "No environment capabilities found."


async def parse_atomic_intent(agent, logger, span: str, category: str, capabilities_hierarchical: str) -> dict:
    """Parse a single atomic intent span using LLM per-span prompts.

    Args:
        span: The verbatim user text for this atomic intent
        category: One of "GOAL_REQUEST", "ENV_STATE_REQUEST", "ENV_CAPABILITIES_REQUEST"
        capabilities_hierarchical: Hierarchical text description of environment capabilities

    Returns:
        Dict with parsed structured intent fields (LLM output), or fallback dict on error
    """
    logger.info(demo(f"[LLM CALL] Parsing {category}: {span[:80]!r}"))

    # Choose prompt and template variables by category
    ontology = agent.ontology_ttl

    if category == "GOAL_REQUEST":
        prompt_template = GOAL_REQUEST_PARSER_PROMPT
        prompt = prompt_template.format(
            capabilities_hierarchical=capabilities_hierarchical,
            ontology=ontology,
        )
    elif category == "ENV_STATE_REQUEST":
        prompt_template = ENV_STATE_REQUEST_PARSER_PROMPT
        prompt = prompt_template.format(
            capabilities_hierarchical=capabilities_hierarchical,
            ontology=ontology,
        )
    elif category == "ENV_CAPABILITIES_REQUEST":
        prompt_template = ENV_CAPABILITIES_REQUEST_PARSER_PROMPT
        prompt = prompt_template.format(
            capabilities_hierarchical=capabilities_hierarchical,
            ontology=ontology,
        )
    else:
        # Unknown category — return minimal fallback
        return {"text_intent": span}

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": span},
    ]

    try:
        response = await agent.llm_client.chat.completions.create(
            model=agent.llm_model,
            messages=messages,
            **agent.build_llm_kwargs(),
        )
        raw = (response.choices[0].message.content or "").strip()
        logger.info(demo("Raw intent extraction output: %s"), raw[:1000])
        parsed = loose_json_loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        logger.warning("LLM per-span parsing failed for [%s]: %s", category, exc)

    # Fallback: minimal dict with only text_intent
    return {"text_intent": span}

# ------------------------------------------------------------------
# NLG: plan summary (LLM call)
# ------------------------------------------------------------------


async def summarize_plan(agent, logger, plan_json: str) -> str:
    """Call LLM to produce a user-friendly plan summary."""
    messages = [
        {"role": "system", "content": PLAN_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": plan_json},
    ]
    try:
        response = await agent.llm_client.chat.completions.create(
            model=agent.llm_model,
            messages=messages,
            **agent.build_llm_kwargs(),
        )
        return (response.choices[0].message.content or "").strip() or (
            "Plan ready. Does this plan look good to you?"
        )
    except Exception as exc:
        logger.error("LLM plan summary failed: %s", exc)
        return "I have a plan ready. Does this plan look good to you?"

# ------------------------------------------------------------------
# NLG: query response formatting (LLM call)
# ------------------------------------------------------------------


async def format_query_response(agent, logger, raw_data: str, user_text: str) -> str:
    """Call LLM to format raw query results for the user."""
    messages = [
        {"role": "system", "content": QUERY_RESPONSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"User asked: {user_text}\n\nRaw data:\n{raw_data}"},
    ]
    try:
        response = await agent.llm_client.chat.completions.create(
            model=agent.llm_model,
            messages=messages,
            **agent.build_llm_kwargs(),
        )
        return (response.choices[0].message.content or "").strip() or raw_data
    except Exception as exc:
        logger.error("LLM query formatting failed: %s", exc)
        return raw_data


async def format_capabilities_response(agent, logger, capabilities_ctx: str, extraction: dict) -> str:
    """Call LLM to format capabilities response based on structured extraction."""
    prompt = ENV_CAPABILITIES_RESPONSE_PROMPT.format(capabilities_hierarchical=capabilities_ctx)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(extraction)},
    ]
    try:
        # Build LLM client with capabilities_analysis config
        llm_cfg = build_behaviour_llm_client(agent.config, "capabilities_analysis")
        kwargs = build_llm_call_kwargs(llm_cfg)

        response = await llm_cfg.client.chat.completions.create(
            model=llm_cfg.model,
            messages=messages,
            **kwargs,
        )
        return (response.choices[0].message.content or "").strip() or "Unable to process capabilities query."
    except Exception as exc:
        logger.error("LLM capabilities formatting failed: %s", exc)
        return "Unable to process capabilities query."

# ------------------------------------------------------------------
# DETERMINISTIC: handle goal → request plan → summarize
# ------------------------------------------------------------------


async def fetch_capabilities(agent, logger, detail_level: str = "summary") -> str:
    """Fetch environment capabilities from EnvExplorer via RPC.

    Args:
        detail_level: One of "summary" (default, lightweight hierarchical JSON)
                     or "detailed" (full RDF/Turtle graph).
    """
    explorer_jid = agent.target_jids.get("explorer")
    if not explorer_jid:
        logger.info(demo("Capability fetch skipped: no explorer JID configured"))
        return ""
    try:
        result = await rpc_call(
            agent,
            to_jid=str(explorer_jid),
            request_type=MessageType.ENV_CAPABILITIES_REQUEST.value,
            body={"query": "all", "detail_level": detail_level},
            expect_type=MessageType.ENV_CAPABILITIES_RESPONSE.value,
            timeout=agent.rpc_call_timeout,
        )
        body = result.body or ""
        logger.info(
            demo("Capabilities fetched: body_len=%d"),
            len(body),
        )
        try:
            payload = json.loads(body)
            summary = payload.get("summary") if isinstance(payload, dict) else None
            if isinstance(summary, str) and summary.strip():
                return summary
        except (json.JSONDecodeError, TypeError):
            pass
        return body
    except Exception as exc:
        logger.warning("Failed to fetch capabilities: %s", exc)
        return ""


def collect_action_urls(tree_spec: dict) -> list[str]:
    action_urls: list[str] = []
    seen: set[str] = set()

    def walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "action":
            action_url = str(node.get("action_url") or "").strip()
            if action_url and action_url not in seen:
                seen.add(action_url)
                action_urls.append(action_url)
        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child)

    walk(tree_spec)
    return action_urls

# ------------------------------------------------------------------
# DETERMINISTIC: query handlers
# ------------------------------------------------------------------
