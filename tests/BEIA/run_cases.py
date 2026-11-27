#!/usr/bin/env python3
"""BEIA automation benchmark: six standing rules for the BEIA lab, scored twice.

Each case (`cases/A*.json`) is a request BEIA suggested ("whenever the garage AC
turns on, close the garage window"), the facts a right plan has (`ground_truth`,
see `ground_truth.py`), and a timeline of simulator settings that makes the
rule's condition happen while the plan runs (`stimuli`). A case runs as:

    1. static     the query goes through the real agents; the plan they propose
                  is scored against `ground_truth` (Check 1, no execution)
    2. behaviour  the plan is confirmed and runs against HASP + the BEIA virtual
                  lab; each stimulus is played on the simulated clock and the
                  device states asserted after it (Check 2)

`--static-only` stops after step 1: the proposal is declined, nothing runs, no
device moves, and the simulator need not be running. A full run also skips
step 2 for a plan that failed step 1 (unless `--behaviour-anyway`): it never
waits on a plan already known to be wrong.

Time. The plan's waits ("held for 5 minutes") read the agents' plan clock. For
step 2 it is the simulator's: the virtual lab publishes `sensor.sim_time`, HASP
serves it at `/_clock` (CLOCK_ENTITY), and the agents read it
(CLOCK_SOURCE=http, CLOCK_URL=<hasp>/_clock). Stimuli times and assertion
timeouts are simulated seconds too, so `number.sim_speed` (a case's
`sim_speed`) shortens the whole run.

Built on the lab308e harness (`tests/e2e-lab308e/run_cases.py`): the same agents,
headless user, LLM counters and HASP handling; this file adds the simulator,
the clock, the two checks and three assertions (`property_stays`,
`alert_raised`, `no_alert`).

Run from the repo root (see README.md):
    conda run -n ami-agents python tests/BEIA/run_cases.py --static-only
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
HA_DIR = PROJECT_ROOT / "ami_agents" / "environment" / "integration" / "HomeAssistant"

# -- 0. The BEIA home, before the lab308e harness reads its own defaults ------------------
# It fills only what is unset, so these win over prepare-adapter-env.sh's lab308e.
os.environ.setdefault("AREAS", "peroni_22,garaj,telemetrie,invertoare,solaredge")
os.environ.setdefault("SEMANTIC_CONFIG", str(HA_DIR / "homelab.yaml"))
os.environ.setdefault("CLOCK_ENTITY", "sensor.sim_time")
# The headless user logs in as an account the local Prosody already has (it
# refuses in-band registration of a new one): the manual full-flow test's.
os.environ.setdefault("E2E_RUNNER_JID", f"orchestrator@{os.getenv('SPADE_SERVER', 'localhost')}")

_spec = importlib.util.spec_from_file_location(
    "lab308e_run_cases", PROJECT_ROOT / "tests" / "e2e-lab308e" / "run_cases.py")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base          # its dataclasses look their module up here
_spec.loader.exec_module(base)          # also puts PROJECT_ROOT on sys.path

from ami_agents.agents.user_assistant.behaviours.plan_execution import (  # noqa: E402
    PlanExecutionBehaviour,
)
from tests.BEIA.ground_truth import score  # noqa: E402

RESULTS_DIR = HERE / "results"


# -- time ------------------------------------------------------------------------------

def at_seconds(text: Any) -> float:
    """A stimulus time ("12:30", "45", 90) as seconds."""
    if isinstance(text, (int, float)):
        return float(text)
    minutes, _, seconds = str(text).rpartition(":")
    return float(minutes or 0) * 60 + float(seconds)


class SimClock:
    """The simulated clock, as the agents see it: HASP's GET /_clock."""

    def __init__(self, hasp_url: str) -> None:
        """The clock HASP serves at `<hasp_url>/_clock`."""
        self.url = f"{hasp_url.rstrip('/')}/_clock"
        self.origin: Optional[datetime] = None

    async def now(self) -> datetime:
        """The simulated moment; RuntimeError when HASP serves no clock."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as resp:
                body = await resp.json(content_type=None)
                if resp.status != 200:
                    raise RuntimeError(
                        f"no simulated clock at {self.url} ({resp.status}: {body}); is the "
                        f"BEIA virtual lab running, with sensor.sim_time, and HASP started "
                        f"with CLOCK_ENTITY=sensor.sim_time?")
        return datetime.fromisoformat(body["iso"])

    async def start(self) -> None:
        """Time 0 of a case's stimuli."""
        self.origin = await self.now()

    async def elapsed(self) -> float:
        """Simulated seconds since `start`."""
        return (await self.now() - self.origin).total_seconds()

    async def wait_until(self, seconds: float, poll: float = 0.5) -> None:
        """Until `seconds` of simulated time have passed since `start`."""
        while await self.elapsed() < seconds:
            await asyncio.sleep(poll)


# -- the harness -----------------------------------------------------------------------

class BEIAHarness(base.Lab308eHarness):
    """The lab308e harness for the BEIA home: simulator steps, the sim clock,
    the static check, and the three BEIA assertions."""

    def __init__(self, args: argparse.Namespace, logger) -> None:
        """The lab308e harness, plus the alert collector and the sim clock."""
        super().__init__(args, logger)
        self.alerts: List[Tuple[float, str, str]] = []      # (when, plan, text)
        self.clock = SimClock(self.yggdrasil_url)

    # -- lifecycle ------------------------------------------------------------------------

    async def start(self, cases=None) -> None:
        """The lab308e start, plus a collector on the UA's alert hook."""
        await super().start(cases)
        self.assistant.alert_handlers.append(self._collect_alert)

    def _collect_alert(self, behaviour, alert) -> None:
        """P2's hook: every alert a plan raises, for `alert_raised` / `no_alert`."""
        self.alerts.append((time.monotonic(), behaviour.plan_id, alert.text))

    async def _hard_reset_lab308e(self) -> None:
        """BEIA's reset: the simulator's own "Reset world", physics on, speed 1."""
        await self._set_sim({"button.sim_reset": "press", "switch.sim_running": "on",
                             "number.sim_speed": 1})
        await asyncio.sleep(2)

    async def _stop_simulator_for_setup(self) -> None:
        """The virtual lab runs on its own; nothing to stop."""

    async def _start_simulator_for_execution(self) -> None:
        """The virtual lab runs on its own; nothing to start."""

    # -- Home Assistant and the simulator -------------------------------------------------

    async def _ha_service(self, domain: str, service: str, data: Dict[str, Any]) -> None:
        """One HA service call (the simulator's entities are not HASP's)."""
        token = base._require_env("HA_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}
        async with aiohttp.ClientSession(headers=headers) as session:
            url = f"{self.ha_base_url}/api/services/{domain}/{service}"
            async with session.post(url, json=data) as resp:
                if resp.status >= 400:
                    raise AssertionError(f"{domain}.{service} {data}: HTTP {resp.status}: "
                                         f"{await resp.text()}")

    async def _set_sim(self, values: Dict[str, Any]) -> None:
        """Set simulator entities: a number's value, a switch on/off, a button press."""
        for entity_id, value in values.items():
            domain = entity_id.split(".")[0]
            if domain in ("number", "input_number"):
                await self._ha_service(domain, "set_value",
                                       {"entity_id": entity_id, "value": value})
            elif domain in ("switch", "input_boolean"):
                on = str(value).lower() in ("on", "true", "1")
                await self._ha_service(domain, "turn_on" if on else "turn_off",
                                       {"entity_id": entity_id})
            elif domain == "button":
                await self._ha_service("button", "press", {"entity_id": entity_id})
            else:
                raise ValueError(f"cannot set simulator entity {entity_id}")

    async def _run_beia_steps(self, steps: List[Dict[str, Any]], *, phase: str) -> None:
        """A case's steps: `sim` (simulator settings), `request` (HASP), `sleep`."""
        for step in steps or []:
            if step.get("type") == "sim":
                await self._set_sim(step.get("set") or {})
            else:
                await self._run_steps([step], phase=phase)

    def _cancel_plans(self) -> None:
        """Stop every plan the case started: they are standing rules, they never
        end on their own. Each finishes CANCELLED at its next run()."""
        for behaviour in list(getattr(self.assistant, "behaviours", []) or []):
            if isinstance(behaviour, PlanExecutionBehaviour):
                behaviour._cancelled = True

    # -- one case -------------------------------------------------------------------------

    async def run_case(self, case_path: Path):
        """Static check, then (unless static-only or the plan is wrong) the
        behavioural check; one CaseResult with both."""
        case = json.loads(case_path.read_text())
        name = case.get("name") or case_path.stem
        details: Dict[str, Any] = {"case": case, "transcript": [], "assertions": [],
                                   "model_name": self.model_name}
        timeout = float(case.get("response_timeout_seconds", self.response_timeout))
        started, started_perf = datetime.now().astimezone(), time.perf_counter()
        base.LLMCallCounter.reset()
        self._clear_in_memory_case_state()
        self.logger.info("Running BEIA case %s", name)
        behavioural = not self.args.static_only

        try:
            # 1. A known world (behavioural runs only: a static run moves nothing).
            if behavioural and not self.args.no_hard_reset:
                await self._hard_reset_lab308e()
            if behavioural:
                await self._run_beia_steps(case.get("initial_state") or [], phase="initial_state")
                await asyncio.sleep(float(case.get("pre_query_settle_seconds", 3)))

            # 2. The request, and the plan the agents propose for it.
            with base.LLMCallCounter.phase("initial_prompt"):
                await self.user.ask(str(case["query"]))
                proposal = await self.user.wait_for_reply(timeout)
            details["transcript"].append({"role": "assistant", "phase": "proposal",
                                          "text": proposal})
            self._capture_plan(details)

            # 3. Check 1: the plan against the ground truth.
            details["static"] = score(details.get("plan"), case["ground_truth"])
            static_pass = bool(details["static"]["static_pass"])
            self.logger.info("Case %s static: %s %s", name,
                             "PASS" if static_pass else "FAIL", details["static"]["missing"])

            # 4. Stop here: static-only, or a plan known to be wrong.
            if not behavioural or (not static_pass and not self.args.behaviour_anyway):
                details["behaviour"] = ("not run (--static-only)" if not behavioural
                                        else "skipped (static failed)")
                await self.user.ask("no")              # decline: nothing runs
                with _suppress_timeout():
                    await self.user.wait_for_reply(30)
                return self._result(case_path, details, static_pass, started, started_perf)

            # 5. Check 2: confirm, then play the stimuli on the simulated clock.
            await self._set_sim({"number.sim_speed": float(case.get("sim_speed", 10))})
            self.alerts.clear()
            with base.LLMCallCounter.phase("confirmation"):
                await self.user.ask(str(case.get("confirmation_text", "yes")))
                with _suppress_timeout():
                    reply = await self.user.wait_for_reply(min(timeout, 120))
                    details["transcript"].append({"role": "assistant",
                                                  "phase": "confirmation", "text": reply})
            behaviour_pass = await self._play_stimuli(case, details)
            details["behaviour"] = "passed" if behaviour_pass else "failed"
            return self._result(case_path, details, static_pass and behaviour_pass,
                                started, started_perf)
        except Exception as exc:  # noqa: BLE001 - reported on the row
            details["error"] = str(exc)
            self.logger.exception("BEIA case %s failed", name)
            return self._result(case_path, details, False, started, started_perf)
        finally:
            # 6. Standing rules never end: stop them, and the clock's speed-up.
            self._cancel_plans()
            if behavioural:
                with _suppress_errors():
                    await self._set_sim({"number.sim_speed": 1})

    async def _play_stimuli(self, case: Dict[str, Any], details: Dict[str, Any]) -> bool:
        """Each stimulus at its simulated time, then its assertions; all must hold."""
        await self.clock.start()
        all_passed = True
        for stimulus in case.get("stimuli") or []:
            # 1. Wait for its moment, then make the change.
            await self.clock.wait_until(at_seconds(stimulus.get("at", 0)))
            await self._run_beia_steps(stimulus.get("steps") or [], phase="stimulus")
            # 2. What must follow from it.
            for assertion in stimulus.get("then") or []:
                result = await self._evaluate_assertion(assertion, details)
                result["at"] = stimulus.get("at")
                details["assertions"].append(result)
                all_passed = all_passed and bool(result.get("passed"))
        return all_passed

    def _result(self, case_path: Path, details: Dict[str, Any], passed: bool,
                started: datetime, started_perf: float):
        """The CaseResult, with timing and LLM counts."""
        details["started_at"] = started.isoformat()
        details["duration_seconds"] = round(time.perf_counter() - started_perf, 3)
        details.update(base.LLMCallCounter.snapshot())
        return base.CaseResult(case_path, passed, details)

    # -- assertions, on the simulated clock -----------------------------------------------

    async def _read(self, path: str) -> Any:
        """A HASP property's current value (None when HASP has none)."""
        url = self._resolve_url(None, path)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    return None
        try:
            return json.loads(body)
        except ValueError:
            return body

    @staticmethod
    def _holds(value: Any, assertion: Dict[str, Any]) -> bool:
        """Whether a value meets an assertion's equals / not_equals / in / gte / lte."""
        try:
            if "equals" in assertion:
                return str(value).lower() == str(assertion["equals"]).lower()
            if "not_equals" in assertion:
                return str(value).lower() != str(assertion["not_equals"]).lower()
            if "values" in assertion:
                return str(value).lower() in [str(v).lower() for v in assertion["values"]]
            if "gte" in assertion:
                return float(value) >= float(assertion["gte"])
            if "lte" in assertion:
                return float(value) <= float(assertion["lte"])
        except (TypeError, ValueError):
            return False
        return True

    async def _evaluate_assertion(self, assertion: Dict[str, Any], details: Dict[str, Any]):
        """Property, `property_stays` and alert assertions, with timeouts in
        simulated seconds; anything else as lab308e evaluates it."""
        kind = assertion.get("type")
        window = float(assertion.get("timeout_seconds", assertion.get("seconds", 60)))
        start = await self.clock.elapsed()

        # 1. Eventually: a property reaches a value within the window.
        if kind in ("property_equals", "property_in", "property_gte", "property_lte"):
            value = None
            while True:
                value = await self._read(assertion["path"])
                if self._holds(value, assertion):
                    return {**assertion, "passed": True, "actual": value}
                if await self.clock.elapsed() - start >= window:
                    return {**assertion, "passed": False, "actual": value}
                await asyncio.sleep(1)

        # 2. Throughout: a property keeps a value (or, with none given, its first).
        if kind == "property_stays":
            first = await self._read(assertion["path"])
            wanted = assertion if any(k in assertion for k in
                                      ("equals", "not_equals", "values", "gte", "lte")) \
                else {"equals": first}
            while await self.clock.elapsed() - start < window:
                value = await self._read(assertion["path"])
                if not self._holds(value, wanted):
                    return {**assertion, "passed": False, "actual": value, "first": first}
                await asyncio.sleep(1)
            return {**assertion, "passed": True, "first": first}

        # 3. Alerts, from the UA's alert hook.
        if kind in ("alert_raised", "no_alert"):
            seen = len(self.alerts)
            words = [w.lower() for w in assertion.get("contains_any") or []]

            def new_matching():
                """Alerts raised since the assertion began, saying one of the words."""
                return [text for _, _, text in self.alerts[seen:]
                        if not words or any(w in text.lower() for w in words)]
            while await self.clock.elapsed() - start < window:
                if new_matching() and kind == "alert_raised":
                    return {**assertion, "passed": True, "alerts": new_matching()}
                await asyncio.sleep(1)
            raised = new_matching()
            return {**assertion, "passed": (bool(raised) if kind == "alert_raised"
                                            else not raised), "alerts": raised}

        return await super()._evaluate_assertion(assertion, details)


class _suppress_timeout:
    """`with`: a reply that does not come is not an error here."""

    def __enter__(self):
        """Nothing to set up."""
        return self

    def __exit__(self, kind, exc, tb):
        """Swallow a timeout; let anything else through."""
        return kind is asyncio.TimeoutError


class _suppress_errors(_suppress_timeout):
    """`with`: best-effort cleanup."""

    def __exit__(self, kind, exc, tb):
        """Swallow any error: cleanup must not hide the case's own result."""
        return kind is not None and issubclass(kind, Exception)


# -- results ---------------------------------------------------------------------------

def write_results(results, path: Path) -> Path:
    """One JSON line per case: what `score_plans.py` re-scores, and a reader reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for result in results:
            d = result.details
            fh.write(json.dumps({
                "case": (d.get("case") or {}).get("id") or result.path.stem,
                "query": (d.get("case") or {}).get("query"),
                "passed": result.passed,
                "static": d.get("static"),
                "behaviour": d.get("behaviour"),
                "assertions": d.get("assertions"),
                "error": d.get("error"),
                "plan": d.get("plan"),
                "transcript": d.get("transcript"),
                "model_name": d.get("model_name"),
                "duration_seconds": d.get("duration_seconds"),
                "llm_calls": d.get("llm_calls"),
                "input_tokens": d.get("input_tokens"),
                "output_tokens": d.get("output_tokens"),
                "started_at": d.get("started_at"),
            }, default=str) + "\n")
    return path


def _parse_args(argv: List[str]) -> argparse.Namespace:
    """The lab308e harness's options (it reads them), plus BEIA's."""
    parser = argparse.ArgumentParser(description="Run the BEIA automation benchmark.")
    parser.add_argument("--case-dir", default=str(HERE / "cases"))
    parser.add_argument("--case", help="Run one case file.")
    parser.add_argument("--case-name", help="Run one case by its JSON 'name'.")
    parser.add_argument("--static-only", action="store_true",
                        help="Score the proposed plans only; decline them, run nothing.")
    parser.add_argument("--behaviour-anyway", action="store_true",
                        help="Run the behavioural check even for a plan that failed the static one.")
    parser.add_argument("--results", default=None,
                        help="JSONL results file (default results/beia-<timestamp>.jsonl).")
    parser.add_argument("--response-timeout", type=float, default=900.0)
    parser.add_argument("--discovery-timeout", type=float, default=300.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--clear-signifiers", action="store_true")
    parser.add_argument("--clear-signifiers-per-case", action="store_true")
    parser.add_argument("--skip-prewarm", action="store_true")
    parser.add_argument("--dump-prompts", action="store_true")
    parser.add_argument("--prompt-dump-dir")
    parser.add_argument("--no-hard-reset", action="store_true")
    parser.add_argument("--manage-hasp", action="store_true")
    parser.add_argument("--no-td-sosa", action="store_true")
    parser.add_argument("--manage-simulator", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--hasp-host", default="0.0.0.0")
    parser.add_argument("--service-start-timeout", type=float, default=30.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> int:
    """Every case, a summary, the results file; exit 1 when any case failed."""
    # 1. The plan clock: the simulator's for a behavioural run.
    if not args.static_only:
        hasp = os.getenv("HASP_URL", "http://localhost:8080/").rstrip("/")
        os.environ.setdefault("CLOCK_SOURCE", "http")
        os.environ.setdefault("CLOCK_URL", f"{hasp}/_clock")
    # 2. The cases, through one harness.
    # The lab308e harness sets up logging; this run's lines are BEIA's own.
    base._configure_logging(args.log_level)
    logger = logging.getLogger("BEIARunner")
    base._install_openai_call_counter()
    cases = base._discover_cases(args)
    harness = BEIAHarness(args, logger)
    results = []
    try:
        await harness.start(cases)
        for case_path in cases:
            results.append(await harness.run_case(case_path))
    finally:
        await harness.stop()
    # 3. The summary and the file.
    print("\n=== BEIA benchmark ===")
    for r in results:
        static = (r.details.get("static") or {})
        print(f"{'PASS' if r.passed else 'FAIL'} {r.path.stem:<32} "
              f"static={'pass' if static.get('static_pass') else 'fail'} "
              f"behaviour={r.details.get('behaviour')}"
              + (f"  missing={static.get('missing')}" if static.get("missing") else "")
              + (f"  error={r.details['error']}" if r.details.get("error") else ""))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = write_results(results, Path(args.results) if args.results
                        else RESULTS_DIR / f"beia-{stamp}.jsonl")
    print(f"Results: {out}")
    return 0 if all(r.passed for r in results) else 1


def main() -> int:
    """Parse the options and run the benchmark under SPADE; the exit code."""
    args = _parse_args(sys.argv[1:])
    holder = {"code": 2}

    async def run() -> None:
        """The whole run, inside SPADE's event loop."""
        holder["code"] = await _async_main(args)

    base.spade.run(run())
    return holder["code"]


if __name__ == "__main__":
    raise SystemExit(main())
