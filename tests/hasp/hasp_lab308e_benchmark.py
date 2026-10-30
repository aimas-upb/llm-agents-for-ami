#!/usr/bin/env python3
"""Load lab308e into Home Assistant, then run its five agent scenarios N times.

    export HA_TOKEN=... OPENAI_API_KEY=...
    python tests/hasp/hasp_lab308e_benchmark.py --runs 3 \
        --virtual-dir /path/to/ha/config/custom_components/virtual

Home Assistant must already be running with the Virtual Components integration
installed. --virtual-dir is the local/shared directory visible to HA as
/config/custom_components/virtual; --ha-yaml-path changes the HA-side path.
An existing lab308e integration is reused. HASP, the lab308e dynamics simulator,
and fresh agents are managed per trial. A local XMPP server is started by SPADE
when localhost:5222 is free; --xmpp-mode external uses your existing server.

Uses the existing E2E scenario assertions and actual SPADE agents. No SimuHome
installation is needed. Results, logs, and isolated memory live below --out.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit
import uuid

ROOT = Path(__file__).resolve().parents[2]
# Workers re-execute this file directly. Keep project imports independent of
# the script's new directory and the caller's current working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

HA_DIR = ROOT / "ami_agents/environment/integration/HomeAssistant"
CASE_DIR = ROOT / "tests/e2e-lab308e/cases"
SCENARIOS = (
    "make_it_brighter", "cool_down_the_room", "the_air_is_so_stuffy_in_here",
    "reduce_glare", "make_screen_easier_to_see",
)
METRICS = ("duration_seconds", "llm_calls", "input_tokens", "output_tokens")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def positive_float(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", "-n", type=positive_int, default=1,
                        help="Repetitions of EACH of the five scenarios (default: 1).")
    parser.add_argument("--ha-url", default=os.getenv("HA_URL", "http://localhost:8123"),
                        help="Home Assistant HTTP or WebSocket URL; defaults to HA_URL.")
    parser.add_argument("--virtual-dir", type=Path, default=os.getenv("HA_VIRTUAL_DIR"),
                        help="Host/shared HA virtual YAML directory (or HA_VIRTUAL_DIR).")
    parser.add_argument("--ha-yaml-path", default="/config/custom_components/virtual/lab308e.yaml",
                        help="Path to lab308e.yaml as seen by Home Assistant.")
    parser.add_argument("--hasp-port", type=int, default=8090)
    parser.add_argument("--xmpp-mode", choices=("auto", "external"), default="auto")
    parser.add_argument("--model", help="Override every agent LLM model; otherwise use agents.yaml.")
    parser.add_argument("--response-timeout", type=positive_float, default=900)
    parser.add_argument("--startup-timeout", type=positive_float, default=120)
    parser.add_argument("--trial-timeout", type=positive_float, default=2400,
                        help="Overall seconds per trial, including startup and cleanup.")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "tests/e2e-lab308e/results/benchmark")
    parser.add_argument("--dump-prompts", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without connecting, loading HA, or calling models.")
    parser.add_argument("--worker-case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.hasp_port <= 65535:
        parser.error("--hasp-port must be between 1 and 65535")
    if not args.ha_yaml_path.startswith("/"):
        parser.error("--ha-yaml-path must be an absolute path inside Home Assistant")
    return args


def ha_urls(value):
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.hostname:
        raise ValueError("--ha-url must be an HTTP(S) or WS(S) Home Assistant URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use HA_TOKEN for authentication, not credentials or query parameters in HA_URL")
    path = parsed.path.rstrip("/")
    if path.endswith("/api/websocket"):
        path = path[:-len("/api/websocket")]
    secure = parsed.scheme in {"https", "wss"}
    base = urlunsplit(("https" if secure else "http", parsed.netloc, path, "", ""))
    return base, base.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/api/websocket"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def redact(text):
    for key in ("HA_TOKEN", "OPENAI_API_KEY"):
        value = os.getenv(key, "")
        if value:
            text = text.replace(value, "[REDACTED]")
    return text


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(redact(json.dumps(data, indent=2, default=str)) + "\n")
    temporary.replace(path)


def choose_entry(entries, entities, expected):
    """Reuse only a virtual entry; never commandeer unrelated physical entities."""
    by_id = {entry["entry_id"]: entry for entry in entries}
    present = [entity for entity in entities if entity["entity_id"] in expected]
    ids = {entity.get("config_entry_id") for entity in present}
    named = {entry["entry_id"] for entry in entries if entry.get("domain") == "virtual"
             and entry.get("title", "").lower() in {"lab308e", "lab308e - virtual"}}
    candidates = ids | named
    if len(candidates) > 1 or any(by_id.get(key, {}).get("domain") != "virtual" for key in ids):
        raise RuntimeError("lab308e entity IDs conflict with other HA integrations; resolve the duplicates first")
    return next(iter(candidates), None)


async def install_home(args, expected):
    """Provision the Virtual config entry and put its entities in lab308e."""
    import httpx
    from tests.simuhome import add_virtual_devices as virtual

    base, ws_url = ha_urls(args.ha_url)
    token = os.environ["HA_TOKEN"]
    ws = await asyncio.wait_for(virtual._ws_handshake(ws_url, token), args.startup_timeout)

    async def call(payload):
        return await asyncio.wait_for(virtual._ws_call(ws, payload), 20)

    try:
        async with httpx.AsyncClient(base_url=base, timeout=20,
                                    headers={"Authorization": f"Bearer {token}"}) as client:
            response = await client.get("/api/config/config_entries/entry")
            response.raise_for_status()
            entities = await call({"type": "config/entity_registry/list"})
            entry_id = choose_entry(response.json(), entities, expected)
            if args.virtual_dir:
                directory = args.virtual_dir.expanduser().resolve()
                if not directory.is_dir():
                    raise ValueError(f"Virtual YAML directory does not exist: {directory}")
                destination = directory / Path(args.ha_yaml_path).name
                source = HA_DIR / "lab308e.yaml"
                if destination.resolve() != source.resolve():
                    shutil.copy2(source, destination)
            elif entry_id is None:
                raise ValueError("First import requires --virtual-dir (or HA_VIRTUAL_DIR), the directory "
                                 "mapped to HA's virtual YAML directory")

            area_id = await asyncio.wait_for(virtual._ensure_area(ws, "lab308e"), 30)
            # The existing case URLs and simulator are deliberately scoped to this workspace.
            if area_id != "lab308e":
                raise RuntimeError(f"The lab308e area has ID {area_id!r}; this suite requires area_id 'lab308e'")
            if entry_id is None:
                flow = await asyncio.wait_for(virtual._run_virtual_flow(
                    client, area_name="lab308e", area_id=area_id,
                    group_name="lab308e", ha_yaml_path=args.ha_yaml_path), args.startup_timeout)
                entry_id = (flow.get("result") or {}).get("entry_id")
                if flow.get("type") != "create_entry" or not entry_id:
                    raise RuntimeError(f"Virtual Components import failed: {flow}")
            elif args.virtual_dir:
                response = await client.post(f"/api/config/config_entries/entry/{entry_id}/reload")
                response.raise_for_status()

            deadline = time.monotonic() + args.startup_timeout
            while True:
                entities = await call({"type": "config/entity_registry/list"})
                registered = {e["entity_id"]: e for e in entities if e.get("config_entry_id") == entry_id}
                missing = set(expected) - registered.keys()
                if not missing:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Virtual lab308e import missing entities: {', '.join(sorted(missing))}. "
                                       "Check the YAML mount and Virtual integration; suffixed IDs are not supported.")
                await asyncio.sleep(1)
            devices = {registered[name].get("device_id") for name in expected} - {None}
            for device_id in sorted(devices):
                await call({"type": "config/device_registry/update", "device_id": device_id,
                            "area_id": area_id})
            for name in sorted(expected):
                await call({"type": "config/entity_registry/update", "entity_id": name,
                            "area_id": area_id})
            # Registries can appear before the virtual platforms have finished setting up.
            while True:
                response = await client.get("/api/states")
                response.raise_for_status()
                ready = {s["entity_id"] for s in response.json() if s.get("state") != "unavailable"}
                missing = set(expected) - ready
                if not missing:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"HA entities not ready: {', '.join(sorted(missing))}")
                await asyncio.sleep(1)
            return {"area_id": area_id, "entry_id": entry_id, "entities": sorted(expected)}
    finally:
        await ws.close()


def prepare_environment(args):
    base, ws = ha_urls(args.ha_url)
    hasp = f"http://127.0.0.1:{args.hasp_port}"
    os.environ.update(HA_URL=ws, HA_BASE_URL=base, AREAS="lab308e",
                      YGGDRASIL_URL=hasp, HASP_URL=hasp, BASE_WS_URI=hasp + "/",
                      SEMANTIC_CONFIG=str(HA_DIR / "lab308e.yaml"), PYTHONUNBUFFERED="1",
                      CURRENT_ENVIRONMENT="lab308", CLOCK_SOURCE="system", COMMUNITY_API_URL="")
    # HASP derives variables, effects and ranges from SEMANTIC_CONFIG now;
    # the legacy TD-SOSA JSON sidecars were removed from the repository.


def port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def prepare_cases(out, ha_base):
    paths = []
    for scenario in SCENARIOS:
        case = json.loads((CASE_DIR / f"{scenario}.json").read_text())
        case["workspace_id"] = "lab308e"
        # A legacy fixture contains localhost:8123 URLs. Keep credentials bound
        # to the HA instance selected for this benchmark.
        for step in case.get("initial_state", []):
            if step.get("url", "").startswith("http://localhost:8123/"):
                step["url"] = ha_base + step["url"][len("http://localhost:8123"):]
        path = out / "cases" / f"{scenario}.json"
        write_json(path, case)
        paths.append(path)
    return paths


def make_harness(runner, args, logger):
    """Adapt only configuration/lifecycle; reasoning and scoring stay in the agents."""
    class BenchmarkHarness(runner.Lab308eHarness):
        def _agent_config(self, config, section):
            # Agent constructors consume their own llm/planning sections at the
            # top level, as in the production launcher.
            return runner.ConfigLoader.merge_configs(copy.deepcopy(config), copy.deepcopy(config.get(section, {})))

        def _load_config(self):
            config = super()._load_config()
            config.setdefault("experience_engine", {})["storage_dir"] = str(args.out / "memory")
            config["community"] = {"enabled": False, "api_url": None}
            config.setdefault("interaction_solver", {}).setdefault("planning", {})["community"] = {
                "enabled": False, "api_url": None}
            config.setdefault("logging", {}).setdefault("file_logging", {})["enabled"] = False
            if args.model:
                def override(node):
                    if isinstance(node, dict):
                        for key, value in node.items():
                            if key in {"model", "default_model"}:
                                node[key] = args.model
                            else:
                                override(value)
                override(config)
            # Record model choices without serializing API credentials.
            self.model_choices = {
                "user_assistant": config.get("user_assistant", {}).get("llm", {}),
                "planning": {k: v for k, v in config.get("interaction_solver", {}).get("planning", {}).items()
                             if k in {"llm_planning", "explicit_planning"}},
                "default_model": config.get("default_llm", {}).get("providers", {}).get("openai", {}).get("model"),
            }
            return config

        async def _wait_for_yggdrasil(self, timeout_s=20):
            # Avoid considering an unrelated process on the same port healthy.
            await asyncio.sleep(0.2)
            if self.hasp_service and self.hasp_service.process.returncode is not None:
                raise RuntimeError("Managed HASP exited during startup; see trial.log")
            await super()._wait_for_yggdrasil(timeout_s)
            if self.hasp_service and self.hasp_service.process.returncode is not None:
                raise RuntimeError("Managed HASP exited during startup; see trial.log")

    runner_args = runner._parse_args([
        "--case", str(args.worker_case), "--manage-hasp", "--manage-simulator",
        "--hasp-host", "127.0.0.1", "--response-timeout", str(args.response_timeout),
        "--service-start-timeout", str(args.startup_timeout),
        "--discovery-timeout", str(args.startup_timeout),
        "--prompt-dump-dir", str(args.out / "prompts"),
        *(["--dump-prompts"] if args.dump_prompts else []),
    ])
    harness = BenchmarkHarness(runner_args, logger)
    if args.model:
        harness.model_name = args.model
    suffix = uuid.uuid4().hex[:10]
    for attr in ("explorer_jid", "assistant_jid", "solver_jid", "runner_jid"):
        name, server = getattr(harness, attr).split("@", 1)
        setattr(harness, attr, f"{name}_{suffix}@{server}")
    return harness


def worker_main(args):
    prepare_environment(args)
    runner = load_module("lab308e_e2e", ROOT / "tests/e2e-lab308e/run_cases.py")
    logger = runner._configure_logging("INFO")
    runner._install_openai_call_counter()
    if port_open("127.0.0.1", args.hasp_port):
        raise RuntimeError(f"HASP port {args.hasp_port} is occupied; choose another --hasp-port")
    # Separate callback listener for this trial, including concurrent benchmarks.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        callback_port = sock.getsockname()[1]
    os.environ["INTEGRATION_ENGINE_PORT"] = str(callback_port)
    os.environ["NOTIFICATION_CALLBACK_URL"] = f"http://127.0.0.1:{callback_port}/webhook"
    server = os.getenv("SPADE_SERVER", "localhost")
    embedded = args.xmpp_mode == "auto" and server == "localhost" and not port_open(server, 5222)
    if not embedded and not port_open(server, 5222):
        raise RuntimeError(f"XMPP server is unreachable at {server}:5222")
    result = {"passed": False, "failure_stage": "startup", "error": "Worker did not complete"}

    async def execute():
        harness = make_harness(runner, args, logger)
        started = time.perf_counter()
        try:
            await harness.start([args.worker_case])
            case_result = await harness.run_case(args.worker_case)
            result.clear()
            result.update(case_result.details, passed=case_result.passed)
            result["models"] = harness.model_choices
        except Exception as exc:
            result.update(error=redact(f"{type(exc).__name__}: {exc}"))
            logger.exception("Trial failed")
        finally:
            try:
                await harness.stop()
            finally:
                result["trial_seconds"] = time.perf_counter() - started
                write_json(args.out / "result.json", result)

    # SPADE owns the event loop and (when requested) an in-memory XMPP server.
    try:
        runner.spade.run(execute(), embedded_xmpp_server=embedded)
    finally:
        write_json(args.out / "result.json", result)
    return 0 if result["passed"] else 1


def stop_process(process):
    """Stop the worker AND its managed services, including after interruption."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    # A startup failure or signal can leave descendants after the worker exits.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


async def run_trial(command, log_path, timeout):
    # Run blocking pipe reads off the asyncio loop so the watchdog remains active.
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)
    try:
        with log_path.open("w") as log:
            async with asyncio.timeout(timeout):
                while line := await asyncio.to_thread(process.stdout.readline):
                    line = redact(line)
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                return await asyncio.to_thread(process.wait)
    finally:
        stop_process(process)
        process.stdout.close()


def write_summary(out, records, planned):
    rows = []
    for name in (*SCENARIOS, "ALL"):
        selected = [r for r in records if name == "ALL" or r["scenario"] == name]
        if not selected:
            continue
        row = {"scenario": name, "completed": len(selected),
               "passed": sum(bool(r["passed"]) for r in selected)}
        row["failed"] = row["completed"] - row["passed"]
        row["pass_rate"] = row["passed"] / row["completed"]
        for metric in METRICS:
            values = [r[metric] for r in selected if isinstance(r.get(metric), (int, float))]
            row[f"total_{metric}"] = sum(values)
            row[f"mean_{metric}"] = sum(values) / len(values) if values else None
        rows.append(row)
    write_json(out / "summary.json", {"planned_trials": planned, "completed_trials": len(records), "rows": rows})
    if rows:
        with (out / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# HASP lab308e benchmark", "", f"Completed {len(records)} / {planned} trials.", "",
             "| Scenario | Passed / completed | Mean seconds | LLM calls | Input tokens | Output tokens |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        seconds = row["mean_duration_seconds"]
        lines.append(f"| {row['scenario']} | {row['passed']} / {row['completed']} | "
                     + (f"{seconds:.2f}" if seconds is not None else "—")
                     + f" | {row['total_llm_calls']} | {row['total_input_tokens']} | {row['total_output_tokens']} |")
    lines.extend(["", "Durations include case setup, planning, execution, and assertions; agent/service startup is",
                  "recorded separately in each result's `trial_seconds`. Missing usage is excluded from means.",
                  "LLM calls/tokens cover successful responses observed by the existing E2E runner.",
                  "Every trial uses fresh agents and an isolated memory directory.", ""])
    (out / "summary.md").write_text("\n".join(lines))


async def benchmark(args):
    for key in ("HA_TOKEN", "OPENAI_API_KEY"):
        if not os.getenv(key, "").strip():
            raise ValueError(f"Set {key} in the environment or repository .env")
    if port_open("127.0.0.1", args.hasp_port):
        raise RuntimeError(f"HASP port {args.hasp_port} is occupied; choose another --hasp-port")
    prepare_environment(args)
    # Import to obtain the existing suite's authoritative expected entity IDs.
    runner = load_module("lab308e_e2e_preflight", ROOT / "tests/e2e-lab308e/run_cases.py")
    print("Loading lab308e into Home Assistant...", flush=True)
    home = await install_home(args, set(runner.LAB308E_HARD_RESET_STATES))
    out = args.out.expanduser().resolve() / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    out.mkdir(parents=True)
    cases = prepare_cases(out, ha_urls(args.ha_url)[0])
    write_json(out / "run.json", {"started": datetime.now().astimezone().isoformat(),
                                "runs": args.runs, "scenarios": list(SCENARIOS), "home": home,
                                "ha_url": ha_urls(args.ha_url)[0], "hasp_port": args.hasp_port,
                                "model_override": args.model, "memory": "isolated per trial"})
    records = []
    planned = args.runs * len(cases)
    print(f"Results: {out}", flush=True)
    write_summary(out, records, planned)
    for iteration in range(1, args.runs + 1):
        for case in cases:
            trial_dir = out / f"run_{iteration:03d}" / case.stem
            trial_dir.mkdir(parents=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--worker-case", str(case), "--out", str(trial_dir),
                       "--ha-url", args.ha_url, "--hasp-port", str(args.hasp_port),
                       "--xmpp-mode", args.xmpp_mode, "--response-timeout", str(args.response_timeout),
                       "--startup-timeout", str(args.startup_timeout)]
            if args.model:
                command += ["--model", args.model]
            if args.dump_prompts:
                command += ["--dump-prompts"]
            print(f"\n[{len(records) + 1}/{planned}] run {iteration}: {case.stem}", flush=True)
            failure = None
            try:
                code = await run_trial(command, trial_dir / "trial.log", args.trial_timeout)
            except TimeoutError:
                code, failure = 2, f"Trial exceeded {args.trial_timeout}s"
            except asyncio.CancelledError:
                write_json(trial_dir / "result.json", {"passed": False, "failure_stage": "interrupted"})
                raise
            result_path = trial_dir / "result.json"
            try:
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                result = {}
            if "passed" not in result or failure or (code != 0 and result.get("passed")):
                result.update(passed=False, failure_stage="infrastructure",
                              error=failure or f"Worker exited with code {code}; see trial.log")
            result.update(scenario=case.stem, iteration=iteration, worker_exit_code=code)
            write_json(result_path, result)
            records.append(result)
            write_summary(out, records, planned)
            print(f"{'PASS' if result['passed'] else 'FAIL'} {case.stem}", flush=True)
    print(f"\nStats: {out / 'summary.md'}", flush=True)
    return 0 if all(r["passed"] for r in records) else 1


def main(argv=None):
    # .env is optional; explicit environment values always win.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    args = parse_args(argv)
    if args.dry_run:
        base, _ = ha_urls(args.ha_url)
        print(f"Home Assistant: {base}\nHASP: http://127.0.0.1:{args.hasp_port}\n"
              f"{args.runs} repetitions × {len(SCENARIOS)} scenarios = {args.runs * len(SCENARIOS)} trials")
        print("\n".join(f"  {name}" for name in SCENARIOS))
        print(f"Results under: {args.out}\nA fresh agent stack and memory store are used for every trial.")
        return 0
    try:
        if args.worker_case:
            return worker_main(args)
        return asyncio.run(benchmark(args))
    except KeyboardInterrupt:
        print("\nInterrupted; completed results are preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(redact(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
