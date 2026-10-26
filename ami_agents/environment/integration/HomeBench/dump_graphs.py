#!/usr/bin/env python3
"""Dump the RDF graph HBTD serves for a HomeBench home, one Turtle file per home and mode.

The HomeBench counterpart of the full dump Ioan sent for lab308e: it starts the same
engine + HBTD pair a worker starts (``verify_phase2.spawn_pair``), crawls it the way
the EnvExplorer does (``YggdrasilIntegration``: platform, home workspace, every room
workspace, every artifact), merges every document into one graph and writes
``<out>/home<N>.<mode>.ttl``. Those files are what the HBTD environment model
(one ``#environment`` per room, declared variables, TD-SOSA effects) looks like to an
agent, and they can be opened in any RDF viewer or loaded with ``rdflib``.

Worked example (a small and a large home, both modes, the defaults)::

    python ami_agents/environment/integration/HomeBench/dump_graphs.py
    # -> tests/homebench/results/graphs/home17.on.ttl   home17.off.ttl
    #    tests/homebench/results/graphs/home5.on.ttl    home5.off.ttl
    # stderr: one line per file with its triple count

The ports of the spawned pair are random, so the served IRIs are rewritten to the
stable ``--base`` (default ``http://localhost:8197``) before the merge: two dumps of
the same home differ only when the model does. ``--environment`` limits the dump to
one mode.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import List

if __package__ in {None, ""}:  # allow `python .../HomeBench/dump_graphs.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from rdflib import Graph

from ami_agents.environment.integration.HomeBench.td_builder import ENVIRONMENT_MODES
from ami_agents.environment.integration.HomeBench.verify_phase2 import (
    DEFAULT_DATA, REPO, crawl, spawn_pair,
)

LOGGER = logging.getLogger("dump_graphs")

DEFAULT_OUT = REPO / "tests" / "homebench" / "results" / "graphs"
DEFAULT_HOMES = (17, 5)  # the smallest (35 devices) and a largest (52) of the 100 homes
DEFAULT_BASE = "http://localhost:8197"


def _fetch(url: str) -> str:
    """The Turtle body of ``url`` (the crawler does not keep the platform document)."""
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8")


async def _crawl_documents(base: str) -> List[str]:
    """Every Turtle document the EnvExplorer's crawl of ``base`` sees, as text.

    The platform document (``/``) and the home workspace are read directly, then the
    crawler's own maps supply every workspace and every artifact: the same set an
    agent builds its context from.
    """
    # 1. The crawl, with the engine the agents use.
    integration = await crawl(base)
    # 2. The platform document, which the integration engine does not keep.
    documents = [_fetch(base + "/")]
    # 3. Every workspace (home and rooms) and every artifact's Thing Description.
    documents += [ws.rdf for ws in integration.workspace_map.values()]
    documents += [art.thing_description.rdf for art in integration.artifact_map.values()]
    return documents


def dump_home(home_id: int, environment: str, data: Path, out: Path, base: str) -> Path:
    """Write ``home<home_id>.<environment>.ttl`` under ``out`` and return its path.

    The pair is started in ``environment`` mode, crawled, and every document merged
    into one graph; ``base`` replaces the random port in every IRI.
    """
    # 1. Start the pair for this home and mode, and read everything it serves.
    config = json.loads((data / "homes" / f"home{home_id}.json").read_text(encoding="utf-8"))
    with spawn_pair(home_id, config, environment) as (_sim_url, served):
        documents = asyncio.run(_crawl_documents(served))
    # 2. One graph, IRIs rewritten to the stable base (documents are parsed one at a
    #    time: a bnode id in one document must not collide with another's).
    graph = Graph()
    for text in documents:
        graph.parse(data=text.replace(served, base), format="turtle")
    # 3. Write it, namespaces bound from the documents.
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"home{home_id}.{environment}.ttl"
    graph.serialize(destination=str(path), format="turtle")
    LOGGER.info("%s  %d triples  (%d documents)", path, len(graph), len(documents))
    return path


def main() -> int:
    """CLI: see the module docstring."""
    parser = argparse.ArgumentParser(description="Dump the HBTD graph of HomeBench homes.")
    parser.add_argument("--home", type=int, nargs="+", default=list(DEFAULT_HOMES),
                        help="home ids, 0-99")
    parser.add_argument("--environment", choices=ENVIRONMENT_MODES, default=None,
                        help="one mode only (default: both)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="the IRI base written in the dump (no trailing slash)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    logging.getLogger("ami_agents").setLevel(logging.WARNING)

    modes: List[str] = [args.environment] if args.environment else list(ENVIRONMENT_MODES)
    for home_id in args.home:
        for mode in modes:
            dump_home(
                home_id, mode, args.data, args.out, args.base.rstrip("/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
