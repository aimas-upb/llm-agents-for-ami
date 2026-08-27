"""EXPLICIT intent handler: SPARQL query builder + deterministic BT construction."""

from typing import Dict, List, Optional
from rdflib import Graph

from ami_agents.shared.models.intents import ExplicitGoalIntent
from ami_agents.shared.utils.logger import LoggerFactory


def build_sparql_query(explicit_intent: ExplicitGoalIntent) -> str:
    """Build SPARQL SELECT query from ExplicitGoalIntent semantic constraints."""

    def has_value(v):
        return v is not None and v != "NA"

    target = explicit_intent.target
    action = explicit_intent.action

    # Build conditional blocks based on intent constraints
    workspace_type_block = (
        f"?workspace a {target.workspace_type} ." if has_value(target.workspace_type) else ""
    )

    artifact_type_block = (
        f"?artifact a {target.artifact_type} ." if has_value(target.artifact_type) else ""
    )

    affordance_type_block = (
        f"?affordance a {action.affordance_type} ." if has_value(action.affordance_type) else ""
    )

    if has_value(action.parameter):
        parameter_block = f"""?affordance td:hasInputSchema ?inputSchema .
    ?inputSchema jsonschema:properties ?property .
    ?property jsonschema:propertyName ?parameter_name ;
              a ?parameter_schema_type .
    FILTER (?parameter_name = "{action.parameter}")"""
    else:
        parameter_block = ""

    # Build full SPARQL query with standard prefixes
    # Note: We don't define 'ex' prefix here because it may vary per environment;
    # rdflib will use the graph's namespace bindings
    query = f"""
    PREFIX hmas: <https://purl.org/hmas/>
    PREFIX td: <https://www.w3.org/2019/wot/td#>
    PREFIX hctl: <https://www.w3.org/2019/wot/hypermedia#>
    PREFIX jsonschema: <https://json-schema.org/>

    SELECT ?workspace ?artifact ?affordance_name ?target_uri ?parameter_name ?parameter_schema_type
    WHERE {{
        ?workspace hmas:contains ?artifact .
        {workspace_type_block}
        ?artifact td:hasActionAffordance ?affordance .
        {artifact_type_block}
        ?affordance td:name ?affordance_name ;
                    td:hasForm ?form .
        {affordance_type_block}
        ?form hctl:hasTarget ?target_uri .
        {parameter_block}
    }}
    """

    return query


async def query_environment_graph(
    graph: Graph,
    explicit_intent: ExplicitGoalIntent,
) -> List[Dict]:
    """Execute SPARQL query against RDF graph and return result bindings."""
    sparql_query = build_sparql_query(explicit_intent)

    # Graph already has namespaces bound from parsing the Turtle,
    # so just execute the query directly
    results = graph.query(sparql_query)

    # Convert SPARQL results to list of dicts
    result_list = []
    for row in results:
        row_dict = {}
        for var in results.vars:
            row_dict[str(var)] = row[var]
        result_list.append(row_dict)

    return result_list


async def handle_explicit_intent(
    explicit_intent: ExplicitGoalIntent,
    graph: Graph,
    logger=None,
) -> Dict:
    """
    Main router for EXPLICIT intents.

    Routes based on verb (set/modify) and SPARQL query result count.
    Returns dict with "impossible", "tree", "source", and "context" fields.
    """
    if logger is None:
        logger = LoggerFactory.get_logger("InteractionSolver")

    # Execute SPARQL query
    results = await query_environment_graph(graph, explicit_intent)
    logger.debug(f"SPARQL query for explicit intent returned {len(results)} result(s)")

    if explicit_intent.action.verb == "set":
        return await handle_explicit_set(
            explicit_intent, results, logger
        )
    elif explicit_intent.action.verb == "modify":
        return await handle_explicit_modify(explicit_intent, results, logger)
    else:
        logger.warning(f"Unknown verb: {explicit_intent.action.verb}")
        return {
            "impossible": True,
            "reason": f"Unknown verb: {explicit_intent.action.verb}",
            "tree": None,
            "source": "explicit_unknown_verb",
            "context": {},
        }


async def handle_explicit_set(
    intent: ExplicitGoalIntent,
    query_results: List[Dict],
    logger,
) -> Dict:
    """
    Handle EXPLICIT intent with verb="set".

    Routes based on SPARQL result count:
    - 1 result: impossible=False, return deterministic BT (fast path)
    - 0 results: impossible=False, proceed to signifier matching and LLM planning
    - >1 results: impossible=False, return context with candidate affordances for LLM path
    """

    if len(query_results) == 1:
        # FAST PATH: Single deterministic BT from query result
        logger.info("EXPLICIT SET intent: single affordance found, building deterministic BT")
        return {
            "impossible": False,
            "tree": await build_bt_from_query_result(intent, query_results[0], logger),
            "source": "explicit_sparql_deterministic",
            "context": {},
        }

    elif len(query_results) == 0:
        # NO SPARQL MATCH: Let signifier matching and LLM planning decide
        logger.info("EXPLICIT SET intent: no SPARQL matches, proceeding to signifier matching and LLM planning")
        return {
            "impossible": False,
            "tree": None,
            "source": "explicit_sparql_no_match",
            "context": {
                "reason": f"SPARQL query found no affordances matching: "
                f"artifact_type={intent.target.artifact_type}, "
                f"workspace_type={intent.target.workspace_type}, "
                f"affordance_type={intent.action.affordance_type}, "
                f"parameter={intent.action.parameter}",
            },
        }

    else:
        # MULTIPLE RESULTS: Ambiguous, will be disambiguated by signifier matching in Step 1
        # or by LLM planning with filtered context
        logger.info(
            f"EXPLICIT SET intent: {len(query_results)} affordances match, will disambiguate in next steps"
        )
        return {
            "impossible": False,
            "tree": None,
            "source": "explicit_sparql_ambiguous",
            "context": {
                "candidate_affordances": query_results,
                "ambiguous": True,
                "reason": "Multiple affordances match; will be disambiguated by signifier matching or LLM",
            },
        }


async def handle_explicit_modify(
    intent: ExplicitGoalIntent,
    query_results: List[Dict],
    logger,
) -> Dict:
    """
    Handle EXPLICIT intent with verb="modify".

    Always proceeds to LLM because state lookup is needed to compute delta.
    Query results become the filtered context for planning.
    """
    logger.debug(f"EXPLICIT MODIFY intent: {len(query_results)} candidate affordances, proceeding to LLM")

    return {
        "impossible": False,
        "tree": None,
        "source": "explicit_modify_requires_llm",
        "context": {
            "candidate_affordances": query_results,
            "requires_state_lookup": True,
            "reason": "MODIFY action requires current state to compute delta",
        },
    }


async def build_bt_from_query_result(
    intent: ExplicitGoalIntent,
    query_result: Dict,
    logger,
) -> Optional[Dict]:
    """
    Build deterministic BT JSON IR action node from single SPARQL result.

    Action node name format:
    - If parameter present: "set_{parameter_name}"
    - Otherwise: "set_{affordance_name}"
    """
    target_uri = query_result.get("target_uri")
    affordance_name = query_result.get("affordance_name", "action")
    parameter_name = query_result.get("parameter_name")

    if not target_uri:
        logger.warning("Query result missing target_uri")
        return None

    # Determine action node name
    if intent.action.parameter and parameter_name:
        action_name = f"set_{parameter_name.replace(' ', '_')}"
    else:
        action_name = f"set_{affordance_name.replace(' ', '_')}"

    action_node = {
        "type": "action",
        "action_url": str(target_uri),
        "name": action_name,
    }

    # Inject parameter value if intent specifies both parameter name and value
    if intent.action.parameter and intent.action.value is not None:
        action_node["parameters"] = {intent.action.parameter: intent.action.value}
        logger.debug(f"Injected parameter: {intent.action.parameter}={intent.action.value}")

    return action_node
