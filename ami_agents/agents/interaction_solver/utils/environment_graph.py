"""The discovered TD graph, as the InteractionSolver plans against it.

Fetched from EnvExplorer as one merged Turtle dump (ENV_CAPABILITIES_REQUEST,
`detail_level: "detailed"`) and parsed here, then closed under subclass
inference up to the class roots (`shared/utils/vocabulary.close_types`): a
node typed with a specific class is also, as a plain triple, every ancestor of
it up to its root. An unreachable EnvExplorer or an
unparseable dump yields an empty graph and a logged error: a planner on an
empty graph finds nothing, which callers report rather than crash on.
"""

from __future__ import annotations

import json

from rdflib import Graph

from ....shared.models.messages import MessageType
from ....shared.utils.demo_log import demo
from ....shared.utils.vocabulary import close_types


async def fetch_environment_graph(agent, logger) -> Graph:
    """The whole discovered environment as one rdflib graph."""
    graph = Graph()
    try:
        response_body = await agent._query_env_explorer(
            message_type=MessageType.ENV_CAPABILITIES_REQUEST.value,
            body={"detail_level": "detailed"},
            expect_type=MessageType.ENV_CAPABILITIES_RESPONSE.value,
        )
        try:
            response = json.loads(response_body)
            turtle = response.get("payload", "") if isinstance(response, dict) else ""
        except (json.JSONDecodeError, ValueError):
            turtle = response_body          # raw Turtle
        if turtle:
            graph.parse(data=turtle, format="turtle")
        # Every typed node also gets its superclasses, up to the class roots,
        # so queries filter on a class of any generality with a plain `a`.
        inferred = close_types(graph)
        logger.info(demo(f"Environment graph fetched: {len(graph)} triples "
                         f"({inferred} inferred types)"))
    except Exception as exc:
        logger.error(demo(f"Failed to fetch the environment graph: {exc}"))
        graph = Graph()
    return graph
