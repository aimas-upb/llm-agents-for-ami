"""Home Assistant lifecycle and HASP TDs for the shared reasoning pipeline.

Only environment setup, discovery, real-time execution, and fixture checks
live here. Reasoning stays in tests.simuhome.bench.pipeline, unchanged.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import unquote, urldefrag

import httpx
from rdflib import Graph, RDF, URIRef
from ami_agents.shared.utils.namespaces import HOMEONT

# Reuse only the HA provisioning helpers of the first script. None of its
# benchmark/worker functions (or run_cases.py) are imported or called.
from tests.hasp.hasp_lab308e_benchmark import HA_DIR, ROOT, ha_urls, install_home, prepare_environment
from tests.simuhome.bench.execution import run_immediate
from tests.simuhome.bench.stubs import Home, DEFAULT_MAX_ARTIFACTS, DEFAULT_MAX_ACTION_AFFORDANCES
from ami_agents.agents.env_explorer.utils.data_formatting import format_capabilities_detailed_rdf

RESET = json.loads((Path(__file__).parent / "lab308e-reset.json").read_text())


def execution_succeeded(record):
    plans = (record.get("execution") or {}).get("plans") or []
    return (bool(plans) and all(plan.get("status") == "SUCCESS" for plan in plans)
            and not any(plan.get("outcome") in {"impossible", "clarify", "error", "no_plan"}
                        for plan in record.get("plans", [])))


class HASPHome(Home):
    """The same stand-in Home, populated from HASP's actual RDF documents."""

    def __init__(self, base, workspace_ttl, artifact_ttls, config, logger):
        self.shtd_base, self.home_id = base.rstrip("/"), "lab308e"
        self.rooms = {"lab308e": list(artifact_ttls)}
        self.device_room = {name: "lab308e" for name in artifact_ttls}
        self.sensors = {name: "lab308e" for name in artifact_ttls if "sensing_308e" in name}
        actuation = (config.get("env_explorer") or {}).get("actuation_context") or {}
        self.agent = SimpleNamespace(
            logger=logger, discovery_complete=True,
            environment_map={"lab308e": SimpleNamespace(rdf=workspace_ttl)},
            artifacts={name: SimpleNamespace(thing_description=SimpleNamespace(rdf=ttl))
                       for name, ttl in artifact_ttls.items()},
            actuation_max_artifacts=int(actuation.get("max_artifacts", DEFAULT_MAX_ARTIFACTS)),
            actuation_max_actions=int(actuation.get("max_action_affordances", DEFAULT_MAX_ACTION_AFFORDANCES)))
        self.turtle = format_capabilities_detailed_rdf(self.agent)

    def rooms_of_location(self, location_class):
        if not location_class:
            return []
        local = lambda value: str(value).rsplit("#", 1)[-1].rsplit("/", 1)[-1].rsplit(":", 1)[-1].lower()
        wanted = local(location_class)
        graph = self.graph()
        workspace = URIRef(f"{self.shtd_base}/workspaces/lab308e#workspace")
        types = [kind for space in graph.objects(workspace, HOMEONT.hasSpace)
                 for kind in graph.objects(space, RDF.type)]
        return ["lab308e"] if wanted == "lab308e" or any(local(t) == wanted for t in types) else []


class Lab308eBackend:
    clock = {"source": "system"}

    def __init__(self, args, output):
        self.args, self.output = args, Path(output)
        self.base, _ = ha_urls(args.ha_url)
        if args.hasp_port is None:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                args.hasp_port = sock.getsockname()[1]
        self.hasp = f"http://127.0.0.1:{args.hasp_port}"
        self.headers = {"Authorization": f"Bearer {os.environ['HA_TOKEN']}"}
        self.hasp_process = self.simulator = None
        self.logs = []
        self.case = None

    def _start(self, name, command):
        log = (self.output / f"{name}.log").open("a")
        self.logs.append(log)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(HA_DIR), env.get("PYTHONPATH", "")])
        return subprocess.Popen(command, cwd=HA_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)

    @staticmethod
    async def _stop(process):
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)

    async def start(self):
        prepare_environment(self.args)
        self.home_info = await install_home(self.args, set(RESET["states"]))
        # Never treat an already-running, unrelated server as our own HASP.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.args.hasp_port))
        self.hasp_process = self._start("hasp", [sys.executable, "-m", "uvicorn", "hasp:app",
                                                "--host", "127.0.0.1", "--port", str(self.args.hasp_port)])
        deadline = time.monotonic() + self.args.startup_timeout
        async with httpx.AsyncClient(timeout=5) as client:
            while True:
                if self.hasp_process.poll() is not None:
                    raise RuntimeError(f"HASP exited; see {self.output / 'hasp.log'}")
                try:
                    response = await client.get(f"{self.hasp}/workspaces/lab308e/artifacts")
                    response.raise_for_status()
                    return
                except httpx.HTTPError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"HASP startup timed out; see {self.output / 'hasp.log'}")
                    await asyncio.sleep(.5)

    async def stop(self):
        try:
            await self._stop(self.simulator)
        finally:
            await self._stop(self.hasp_process)
            for log in self.logs:
                log.close()

    async def reset(self, client):
        await self._stop(self.simulator)
        self.simulator = None
        # Services change Virtual Components' internal actuator states; POSTing
        # fake actuator states alone would be overwritten by the next update.
        for calls in RESET["services"].values():
            for call in calls:
                response = await client.post(f"{self.base}/api/services/{call['domain']}/{call['service']}",
                                             headers=self.headers, json=call["data"])
                response.raise_for_status()
        deadline = time.monotonic() + self.args.startup_timeout
        while True:
            response = await client.get(f"{self.base}/api/states", headers=self.headers)
            response.raise_for_status()
            states = {s["entity_id"]: s for s in response.json()}
            pending = [name for name in RESET["services"]
                       if states.get(name, {}).get("state") != RESET["states"][name]["state"]]
            if not pending:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Reset did not settle: {pending}")
            await asyncio.sleep(.5)
        for name, seed in RESET["states"].items():
            if name in RESET["services"]:
                continue
            payload = {**seed, "attributes": {**states.get(name, {}).get("attributes", {}),
                                               **seed.get("attributes", {})}}
            response = await client.post(f"{self.base}/api/states/{name}", headers=self.headers, json=payload)
            response.raise_for_status()

    def url(self, item):
        url = item.get("url") or self.hasp + item["path"]
        return url.replace("http://localhost:8123", self.base, 1) if url.startswith("http://localhost:8123/") else url

    async def setup_steps(self, client):
        for step in self.case.get("initial_state", []):
            if step.get("type", "request") == "sleep":
                await asyncio.sleep(float(step.get("seconds", 1)))
                continue
            headers = {key: os.path.expandvars(value) for key, value in step.get("headers", {}).items()}
            response = await client.request(step.get("method", "POST"), self.url(step),
                                            json=step.get("json"), headers=headers)
            if response.status_code != step.get("expect_status", 200):
                raise RuntimeError(f"Initial state request failed: HTTP {response.status_code}")

    async def start_dynamics(self):
        """Start this trial's dynamics and wait for daylight sensors when requested."""
        # A fresh process anchors every daylight trial at noon, regardless of
        # the previous trial's duration or the host's current time of day.
        command = [sys.executable, str(HA_DIR / "simulate_lab308e.py")]
        daylight = getattr(self.args, "daylight", False)
        ready = self.output / ".daylight-ready" if daylight else None
        if ready is not None:
            ready.unlink(missing_ok=True)
            command += ["--daylight", "--ready-file", str(ready)]
        self.simulator = self._start("simulate_lab308e", command)

        # Don't discover stale nighttime values while the first HA update is
        # still in progress. Surface startup failures instead of scoring them.
        if ready is not None:
            deadline = time.monotonic() + self.args.startup_timeout
            while not ready.exists():
                if self.simulator.poll() is not None:
                    raise RuntimeError("lab308e dynamics process exited; see simulate_lab308e.log")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Daylight startup timed out; see simulate_lab308e.log")
                await asyncio.sleep(.1)

    async def load(self, config, logger):
        """Reset the trial, start dynamics, and discover its live HASP descriptions."""
        # Apply fixture seeds before the simulator starts responding to actuators.
        async with httpx.AsyncClient(timeout=20) as client:
            await self.reset(client)
            await self.setup_steps(client)
            await self.start_dynamics()
            await asyncio.sleep(float(self.case.get("pre_query_settle_seconds", 2)))
            if self.simulator.poll() is not None:
                raise RuntimeError("lab308e dynamics process exited; see simulate_lab308e.log")
            response = await client.post(f"{self.hasp}/_graph/refresh-states", json={})
            response.raise_for_status()
            for assertion in self.case.get("pre_query_assertions", []):
                result = await self.assertion(client, assertion, {}, poll=True)
                if not result["passed"]:
                    raise RuntimeError(f"Initial-state assertion failed: {result}")
            # Discovery must reflect the settled sensors and current room state.
            workspace = await client.get(f"{self.hasp}/workspaces/lab308e")
            workspace.raise_for_status()
            index = await client.get(f"{self.hasp}/workspaces/lab308e/artifacts")
            index.raise_for_status()
            graph = Graph().parse(data=index.text, format="turtle")
            prefix = f"{self.hasp}/workspaces/lab308e/artifacts/"
            urls = sorted({urldefrag(str(s))[0] for s in graph.subjects(RDF.type, URIRef("https://purl.org/hmas/Artifact"))
                           if str(s).startswith(prefix)})
            if not urls:
                raise RuntimeError("HASP discovery returned no lab308e artifacts")
            responses = await asyncio.gather(*(client.get(url, headers={"Accept": "text/turtle"}) for url in urls))
            documents = {}
            for url, response in zip(urls, responses):
                response.raise_for_status()
                documents[unquote(url[len(prefix):])] = response.text
            return HASPHome(self.hasp, workspace.text, documents, config, logger)

    def execute(self, runnable, tools, config, options):
        bt = config.get("bt_execution") or {}
        return run_immediate(runnable, tools, tick_period=float(bt.get("min_tick_yield", .01)),
                             tick_budget=int((bt.get("max_ticks") or {}).get("user_assistant", 100)),
                             timeout=options.exec_timeout)

    def home_state(self):
        with httpx.Client(timeout=20) as client:
            response = client.get(f"{self.base}/api/states", headers=self.headers)
            response.raise_for_status()
            return {s["entity_id"]: s for s in response.json() if "308e" in s["entity_id"]}

    async def assertion(self, client, assertion, record, *, poll):
        kind = assertion["type"]
        if kind in {"all_of", "any_of"}:
            children = [await self.assertion(client, child, record, poll=poll)
                        for child in assertion.get("assertions", [])]
            passed = (all if kind == "all_of" else any)(r["passed"] for r in children)
            return {"type": kind, "passed": passed, "results": children}
        if kind in {"final_reply_contains", "assistant_reply_contains"}:
            # The shared harness executes directly; there is no confirmation
            # conversation. Derive its completion notice from executor results.
            actual = ("Done! The plan was executed successfully." if execution_succeeded(record)
                      else f"Execution failed: {record.get('outcome', 'not executed')}")
            if kind == "assistant_reply_contains":
                actual = record.get("response", "") + "\n" + actual
            return {"type": kind, "passed": str(assertion["value"]) in actual, "actual": actual}
        if kind not in {"property_equals", "property_gte", "property_lte", "property_in"}:
            raise ValueError(f"Unsupported assertion type: {kind}")
        deadline = time.monotonic() + (float(assertion.get("timeout_seconds", 30)) if poll else 0)
        while True:
            response = await client.get(self.url(assertion))
            response.raise_for_status()
            value = response.json()
            try:
                passed = (value == assertion["equals"] if kind == "property_equals" else
                          float(value) >= float(assertion["gte"]) if kind == "property_gte" else
                          float(value) <= float(assertion["lte"]) if kind == "property_lte" else
                          value in assertion["values"])
            except (TypeError, ValueError):
                passed = False
            if passed or time.monotonic() >= deadline:
                return {"type": kind, "passed": passed, "actual": value,
                        "expected": {k: assertion[k] for k in ("equals", "gte", "lte", "values") if k in assertion},
                        "path": assertion.get("path")}
            await asyncio.sleep(float(assertion.get("poll_interval_seconds", 1)))

    async def check(self, record):
        failed_call = any(not call.get("ok", True) for call in (record.get("meter") or {}).get("calls", []))
        if record.get("error") or failed_call:
            return {"score": -1, "error_type": record.get("error") or "Failed LLM call", "check": "lab308e"}
        executed = record.get("outcome") == "executed"
        if executed:
            await asyncio.sleep(float(self.case.get("settle_seconds", 2)))
        async with httpx.AsyncClient(timeout=20) as client:
            results = [await self.assertion(client, assertion, record, poll=executed)
                       for assertion in self.case.get("pass_criteria", [])]
        return {"score": int(executed and bool(results) and all(r["passed"] for r in results)),
                "error_type": None, "check": "lab308e", "assertions": results}
