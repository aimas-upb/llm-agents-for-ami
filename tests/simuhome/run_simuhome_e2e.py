#!/usr/bin/env python3
"""
Run one SimuHome benchmark JSON end to end through Home Assistant, HASP, and
the AMI e2e case runner.

Current scope:
  - qt2 feasible SimuHome environmental-control cases.

The script:
  1. Converts the SimuHome JSON to Home Assistant Virtual Devices YAML.
  2. Imports that YAML into Home Assistant.
  3. Starts HASP for the imported workspace.
  4. Starts the SimuHome sidecar simulator.
  5. Generates a temporary e2e case and runs tests/e2e-lab308e/run_cases.py.
  6. Saves logs/results under tests/simuhome/results/.
  7. Stops subprocesses and removes the imported Virtual Devices workspace.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMUHOME_DIR = PROJECT_ROOT / "tests" / "simuhome"
E2E_RUNNER = PROJECT_ROOT / "tests" / "e2e-lab308e" / "run_cases.py"
HA_INTEGRATION_DIR = PROJECT_ROOT / "ami_agents" / "environment" / "integration" / "HomeAssistant"
RESULTS_ROOT = SIMUHOME_DIR / "results"


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "item"


def _dotted_token(value: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", value.strip())
    cleaned = [part for part in parts if part]
    return "".join(part[:1].upper() + part[1:] for part in cleaned) or "Item"


def _scenario_token(source_path: Path) -> str:
    return _dotted_token(source_path.stem)


def _device_token(device_id: str, room_id: str) -> str:
    prefix = f"{room_id}_"
    core = device_id[len(prefix) :] if device_id.startswith(prefix) else device_id
    return _dotted_token(core)


def _room_state_value(state_name: str, raw_value: Any) -> float:
    if not isinstance(raw_value, (int, float)):
        raise ValueError(f"Room state {state_name!r} is not numeric: {raw_value!r}")
    if state_name in {"temperature", "humidity"}:
        return round(float(raw_value) / 100.0, 3)
    return round(float(raw_value), 3)


def _threshold_delta(state_name: str, overrides: Dict[str, float]) -> float:
    if state_name in overrides:
        return overrides[state_name]
    return {
        "temperature": 0.2,
        "humidity": 1.0,
        "illuminance": 50.0,
        # The sidecar's purifier dynamics reach at most ~0.5 below the pm10
        # baseline at full fan speed (removal 0.05/s vs 0.1/s relaxation),
        # so a delta of 1.0 is physically unsatisfiable.
        "pm10": 0.3,
    }.get(state_name, 1.0)


def _observable_property_for_state(state_name: str) -> Optional[str]:
    return {
        "temperature": "thermal_comfort",
        "humidity": "humidity",
        "illuminance": "luminosity",
        "pm10": "air_quality",
    }.get(state_name)


def _range_unit_for_state(state_name: str) -> str:
    return {
        "temperature": "°C",
        "humidity": "%",
        "illuminance": "lx",
        "pm10": "ug/m3",
    }.get(state_name, "")


def _domain_for_device_type(device_type: str) -> Optional[str]:
    if device_type in {"on_off_light", "dimmable_light"}:
        return "light"
    if device_type in {"fan", "air_purifier", "humidifier", "dehumidifier"}:
        return "fan"
    if device_type in {"air_conditioner", "heat_pump"}:
        return "climate"
    if device_type == "window_covering_controller":
        return "cover"
    # Kept in step with the converter and the sidecar: cabinets are `number`
    # (settable target temperature), start/stop appliances are `switch`.
    if device_type in {"freezer", "refrigerator"}:
        return "number"
    if device_type in {"dishwasher", "laundry_washer", "laundry_dryer", "tv", "rvc"}:
        return "switch"
    return None


def _primary_entity_id(scenario_path: Path, room_id: str, device_id: str, device_type: str) -> Optional[str]:
    domain = _domain_for_device_type(device_type)
    if not domain:
        return None
    name = f"{_scenario_token(scenario_path)}.{_dotted_token(room_id)}.{_device_token(device_id, room_id)}"
    return f"{domain}.{_slug(name)}"


def _room_state_entity_id(scenario_path: Path, room_id: str, state_name: str) -> str:
    name = f"{_scenario_token(scenario_path)}.{_dotted_token(room_id)}.{_dotted_token(state_name)}"
    return f"sensor.{_slug(name)}"


def _env_with_pythonpath(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    memory_path = str(PROJECT_ROOT / "ami_agents" / "shared" / "memory")
    parts = [memory_path, str(PROJECT_ROOT)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra:
        env.update(extra)
    return env


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _run_command(
    cmd: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if log_path:
        log_path.write_text(proc.stdout, encoding="utf-8")
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}\n{proc.stdout}"
        )
    return proc


def _run_command_streaming(
    cmd: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    check: bool = True,
    watch: Optional[Dict[str, subprocess.Popen[str]]] = None,
) -> int:
    log_fh = log_path.open("w", encoding="utf-8") if log_path else None
    proc: Optional[subprocess.Popen[str]] = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        while True:
            for name, watched in (watch or {}).items():
                if watched.poll() is not None:
                    raise RuntimeError(
                        f"{name} exited unexpectedly with code {watched.returncode} "
                        f"while the e2e runner was active; aborting run"
                    )
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            print(line, end="", flush=True)
            if log_fh:
                log_fh.write(line)
                log_fh.flush()
        returncode = proc.wait()
        if check and returncode != 0:
            raise RuntimeError(f"Command failed with exit code {returncode}: {' '.join(cmd)}")
        return returncode
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)
        if log_fh:
            log_fh.close()


def _start_process(
    cmd: list[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
) -> subprocess.Popen[str]:
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    proc._simuhome_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _ensure_running(name: str, proc: subprocess.Popen[str], *, log_path: Path) -> None:
    if proc.poll() is not None:
        raise RuntimeError(
            f"{name} exited unexpectedly with code {proc.returncode}; see {log_path}"
        )


def _stop_process(proc: Optional[subprocess.Popen[str]], *, timeout: float = 10.0) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
    log_fh = getattr(proc, "_simuhome_log_fh", None)
    if log_fh is not None:
        log_fh.close()


def _wait_http(url: str, *, timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Optional[str] = None
    while time.monotonic() < deadline:
        try:
            req = Request(url, headers={"Accept": "text/plain"})
            with urlopen(req, timeout=3.0) as response:
                if response.status < 500:
                    return
                last_error = f"HTTP {response.status}"
        except URLError as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {label} at {url}: {last_error}")


def _load_episode(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"SimuHome JSON must contain an object: {path}")
    return data


def _validate_supported_episode(episode: Dict[str, Any], path: Path) -> None:
    meta = episode.get("meta") if isinstance(episode.get("meta"), dict) else {}
    if meta.get("query_type") != "qt2":
        raise ValueError(
            f"{path.name} is {meta.get('query_type')!r}; this e2e orchestrator currently supports qt2 only"
        )
    if meta.get("case") != "feasible":
        raise ValueError(
            f"{path.name} is case={meta.get('case')!r}; this e2e orchestrator currently supports feasible qt2 only"
        )


def _artifact_slug_for_room_state(scenario_path: Path, room_id: str, state_name: str) -> str:
    entity_name = f"{_scenario_token(scenario_path)}.{_dotted_token(room_id)}.{_dotted_token(state_name)}"
    return _slug(entity_name)


def _initial_room_state(episode: Dict[str, Any], room_id: str, state_name: str) -> float:
    rooms = (((episode.get("initial_home_config") or {}).get("rooms")) or {})
    room = rooms.get(room_id)
    if not isinstance(room, dict):
        raise ValueError(f"Room {room_id!r} not found in initial_home_config")
    state = room.get("state")
    if not isinstance(state, dict) or state_name not in state:
        raise ValueError(f"Room state {room_id}.{state_name} not found in initial_home_config")
    return _room_state_value(state_name, state[state_name])


def _build_qt2_case(
    *,
    episode: Dict[str, Any],
    scenario_path: Path,
    workspace_id: str,
    settle_seconds: float,
    assertion_timeout: float,
    assertion_poll_interval: float,
    deltas: Dict[str, float],
) -> Dict[str, Any]:
    goals = ((episode.get("eval") or {}).get("goals")) or []
    assertions: list[Dict[str, Any]] = []
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        if goal.get("feasibility") is False:
            continue
        room_id = str(goal.get("room_id") or "").strip()
        state_name = str(goal.get("room_state") or "").strip()
        direction = str(goal.get("direction") or "").strip()
        if not room_id or not state_name or direction not in {"increase", "decrease"}:
            continue
        initial = _initial_room_state(episode, room_id, state_name)
        delta = _threshold_delta(state_name, deltas)
        artifact_slug = _artifact_slug_for_room_state(scenario_path, room_id, state_name)
        assertion: Dict[str, Any] = {
            "type": "property_gte" if direction == "increase" else "property_lte",
            "path": f"/workspaces/{workspace_id}/artifacts/{artifact_slug}/properties/state",
            "timeout_seconds": assertion_timeout,
            "poll_interval_seconds": assertion_poll_interval,
        }
        if direction == "increase":
            assertion["gte"] = round(initial + delta, 3)
        else:
            assertion["lte"] = round(initial - delta, 3)
        assertions.append(assertion)

    if not assertions:
        raise ValueError("No transformable feasible qt2 goals found in SimuHome JSON")

    return {
        "name": f"simuhome_{scenario_path.stem}",
        "hard_reset": False,
        "initial_state": [{"type": "sleep", "seconds": 2}],
        "query": episode["query"],
        "auto_confirm": True,
        "dynamic_confirmation": True,
        "confirmation_text": "yes",
        "max_confirmation_turns": 3,
        "settle_seconds": settle_seconds,
        "pass_criteria": [
            {"type": "all_of", "assertions": assertions},
            {"type": "final_reply_contains", "value": "success"},
        ],
        "simuhome": {
            "source_json": str(scenario_path),
            "workspace_id": workspace_id,
            "meta": episode.get("meta"),
        },
    }


def _goal_threshold(initial: float, state_name: str, direction: str, deltas: Dict[str, float]) -> Dict[str, float]:
    delta = _threshold_delta(state_name, deltas)
    if direction == "increase":
        return {"min": round(initial + delta, 3)}
    return {"max": round(initial - delta, 3)}


def _scenario_tdsosa_hints(
    *,
    episode: Dict[str, Any],
    scenario_path: Path,
    deltas: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    goals = ((episode.get("eval") or {}).get("goals")) or []
    rooms = (((episode.get("initial_home_config") or {}).get("rooms")) or {})
    env_overrides: Dict[str, str] = {}
    property_ranges: Dict[str, Dict[str, Any]] = {}

    for goal in goals:
        if not isinstance(goal, dict) or goal.get("feasibility") is False:
            continue
        room_id = str(goal.get("room_id") or "").strip()
        state_name = str(goal.get("room_state") or "").strip()
        direction = str(goal.get("direction") or "").strip()
        observable = _observable_property_for_state(state_name)
        if not room_id or not state_name or direction not in {"increase", "decrease"} or not observable:
            continue

        initial = _initial_room_state(episode, room_id, state_name)
        range_hint = _goal_threshold(initial, state_name, direction, deltas)
        unit = _range_unit_for_state(state_name)
        if unit:
            range_hint["unit"] = unit
        property_ranges[observable] = range_hint

        room_state_entity = _room_state_entity_id(scenario_path, room_id, state_name)
        env_overrides[f"entity:{room_state_entity}:state"] = observable

        room_cfg = rooms.get(room_id) if isinstance(rooms, dict) else None
        device_list = room_cfg.get("devices") if isinstance(room_cfg, dict) else None
        if not isinstance(device_list, list):
            continue
        for device in device_list:
            if not isinstance(device, dict):
                continue
            device_id = str(device.get("device_id") or "").strip()
            device_type = str(device.get("device_type") or "").strip()
            entity_id = _primary_entity_id(scenario_path, room_id, device_id, device_type)
            if not entity_id:
                continue

            # Direction suffixes encode device semantics that service-name
            # heuristics get wrong (e.g. purifier ON lowers pm10).
            if state_name == "temperature" and device_type in {"air_conditioner", "heat_pump"}:
                env_overrides[f"entity:{entity_id}:action:climate.set_temperature"] = observable
                env_overrides[f"entity:{entity_id}:action:climate.set_hvac_mode"] = observable
            elif state_name == "temperature" and device_type == "fan":
                env_overrides[f"entity:{entity_id}:action:fan.turn_on"] = f"{observable}:decrease"
                env_overrides[f"entity:{entity_id}:action:fan.set_percentage"] = f"{observable}:decrease"
                env_overrides[f"entity:{entity_id}:action:fan.turn_off"] = f"{observable}:increase"
            elif state_name == "humidity" and device_type in {"humidifier", "dehumidifier"}:
                humidity_direction = "increase" if device_type == "humidifier" else "decrease"
                inverse_direction = "decrease" if humidity_direction == "increase" else "increase"
                env_overrides[f"entity:{entity_id}:action:fan.turn_on"] = f"{observable}:{humidity_direction}"
                env_overrides[f"entity:{entity_id}:action:fan.set_percentage"] = f"{observable}:{humidity_direction}"
                env_overrides[f"entity:{entity_id}:action:fan.turn_off"] = f"{observable}:{inverse_direction}"
            elif state_name == "illuminance" and device_type in {
                "on_off_light",
                "dimmable_light",
                "window_covering_controller",
            }:
                domain = _domain_for_device_type(device_type)
                if domain == "light":
                    env_overrides[f"entity:{entity_id}:action:light.turn_on"] = observable
                    env_overrides[f"entity:{entity_id}:action:light.turn_off"] = observable
                elif domain == "cover":
                    env_overrides[f"entity:{entity_id}:action:cover.open_cover"] = observable
                    env_overrides[f"entity:{entity_id}:action:cover.close_cover"] = observable
                    env_overrides[f"entity:{entity_id}:action:cover.set_cover_position"] = observable
            elif state_name == "pm10" and device_type == "air_purifier":
                env_overrides[f"entity:{entity_id}:action:fan.turn_on"] = f"{observable}:decrease"
                env_overrides[f"entity:{entity_id}:action:fan.set_percentage"] = f"{observable}:decrease"
                env_overrides[f"entity:{entity_id}:action:fan.turn_off"] = f"{observable}:increase"

    return {
        "TD_SOSA_ENV_VAR_OVERRIDES": env_overrides,
        "TD_SOSA_PROPERTY_RANGES": property_ranges,
    }


def _post_json(url: str, payload: Dict[str, Any], *, timeout_s: float = 20.0) -> Dict[str, Any]:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            return json.loads(body or "{}")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"error": f"http_{exc.code}", "detail": body}
    except Exception as exc:
        return {"error": type(exc).__name__, "detail": str(exc)}


def _save_semantic_queries(
    *,
    result_dir: Path,
    hasp_url: str,
    workspace_id: str,
    observable_properties: Iterable[str],
    label: str,
) -> Path:
    responses: Dict[str, Any] = {}
    for observable_property in sorted(set(observable_properties)):
        responses[observable_property] = _post_json(
            f"{hasp_url}/_graph/query/actions-affecting-observable-property",
            {"workspace_id": workspace_id, "observable_property": observable_property},
        )
    out_path = result_dir / f"{label}_semantic_queries.json"
    out_path.write_text(json.dumps(responses, indent=2, ensure_ascii=True), encoding="utf-8")
    return out_path


def _parse_json_from_output(output: str) -> Dict[str, Any]:
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Command output did not contain a JSON object:\n{output}")
    return json.loads(output[start : end + 1])


def _write_manifest(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _parse_delta_overrides(items: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --delta state=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def run(args: argparse.Namespace) -> int:
    _require_env("HA_URL")
    _require_env("HA_TOKEN")
    _require_env("OPENAI_API_KEY")

    scenario_path = args.scenario.expanduser().resolve()
    episode = _load_episode(scenario_path)
    _validate_supported_episode(episode, scenario_path)
    delta_overrides = _parse_delta_overrides(args.delta)

    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{scenario_path.stem}"
    result_dir = (args.results_dir or RESULTS_ROOT / run_id).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    work_dir = result_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    yaml_dir = work_dir / "yaml"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    scenario_copy = yaml_dir / scenario_path.name
    shutil.copy2(scenario_path, scenario_copy)

    area_name = args.area_name or f"SimuHome {scenario_path.stem}"
    hasp_proc: Optional[subprocess.Popen[str]] = None
    sim_proc: Optional[subprocess.Popen[str]] = None
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "scenario_json": str(scenario_path),
        "result_dir": str(result_dir),
        "area_name": area_name,
        "started_at": dt.datetime.now().astimezone().isoformat(),
    }

    try:
        print(f"[simuhome-e2e] Converting {scenario_path} to Home Assistant YAML", flush=True)
        convert_log = result_dir / "convert.log"
        _run_command(
            [sys.executable, str(SIMUHOME_DIR / "initial_home_config_to_homeassistant_yaml.py"), str(yaml_dir)],
            log_path=convert_log,
        )
        yaml_path = scenario_copy.with_suffix(".yaml")
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Expected converter output not found: {yaml_path}")
        manifest["yaml_path"] = str(yaml_path)

        print(f"[simuhome-e2e] Importing YAML into Home Assistant area {area_name!r}", flush=True)
        add_log = result_dir / "add_virtual_devices.log"
        add_proc = _run_command(
            [
                sys.executable,
                str(SIMUHOME_DIR / "add_virtual_devices.py"),
                str(yaml_path),
                area_name,
                str(args.virtual_yaml_dir.expanduser().resolve()),
            ],
            env=_env_with_pythonpath(),
            log_path=add_log,
        )
        add_info = _parse_json_from_output(add_proc.stdout)
        workspace_id = str(add_info["area_id"])
        manifest["workspace_id"] = workspace_id
        manifest["add_virtual_devices"] = add_info

        if args.no_td_sosa:
            tdsosa_hints = {"TD_SOSA_ENV_VAR_OVERRIDES": {}, "TD_SOSA_PROPERTY_RANGES": {}}
            adapter_app = "ygg_ha_adapter:app"
            hasp_env = _env_with_pythonpath(
                {
                    "AREAS": workspace_id,
                    "BASE_WS_URI": f"http://127.0.0.1:{args.hasp_port}",
                }
            )
        else:
            tdsosa_hints = _scenario_tdsosa_hints(
                episode=episode,
                scenario_path=scenario_path,
                deltas=delta_overrides,
            )
            adapter_app = "hasp:app"
            tdsosa_env_path = result_dir / "tdsosa_env_var_overrides.json"
            tdsosa_ranges_path = result_dir / "tdsosa_property_ranges.json"
            tdsosa_env_path.write_text(
                json.dumps(tdsosa_hints["TD_SOSA_ENV_VAR_OVERRIDES"], indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            tdsosa_ranges_path.write_text(
                json.dumps(tdsosa_hints["TD_SOSA_PROPERTY_RANGES"], indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            manifest["tdsosa_env_var_overrides_path"] = str(tdsosa_env_path)
            manifest["tdsosa_property_ranges_path"] = str(tdsosa_ranges_path)
            manifest["tdsosa_env_var_overrides_count"] = len(tdsosa_hints["TD_SOSA_ENV_VAR_OVERRIDES"])
            manifest["tdsosa_property_ranges_count"] = len(tdsosa_hints["TD_SOSA_PROPERTY_RANGES"])
        manifest["adapter_app"] = adapter_app

        print(f"[simuhome-e2e] Starting {adapter_app} on {args.hasp_port} for workspace {workspace_id}", flush=True)
        hasp_url = f"http://127.0.0.1:{args.hasp_port}"
        if not args.no_td_sosa:
            hasp_env = _env_with_pythonpath(
                {
                    "AREAS": workspace_id,
                    "BASE_WS_URI": hasp_url,
                    "TD_SOSA_ENV_VAR_OVERRIDES": json.dumps(
                        tdsosa_hints["TD_SOSA_ENV_VAR_OVERRIDES"],
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                    "TD_SOSA_PROPERTY_RANGES": json.dumps(
                        tdsosa_hints["TD_SOSA_PROPERTY_RANGES"],
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                }
            )
        hasp_proc = _start_process(
            [
                sys.executable,
                "-m",
                "uvicorn",
                adapter_app,
                "--host",
                "127.0.0.1",
                "--port",
                str(args.hasp_port),
                "--log-level",
                args.hasp_log_level,
            ],
            cwd=HA_INTEGRATION_DIR,
            env=hasp_env,
            log_path=result_dir / "hasp.log",
        )
        manifest["hasp_url"] = hasp_url
        _wait_http(f"{hasp_url}/workspaces/{workspace_id}/artifacts", timeout_s=args.startup_timeout, label="HASP")
        _ensure_running("HASP", hasp_proc, log_path=result_dir / "hasp.log")
        query_props = set(tdsosa_hints["TD_SOSA_PROPERTY_RANGES"])
        if query_props:
            pre_query_path = _save_semantic_queries(
                result_dir=result_dir,
                hasp_url=hasp_url,
                workspace_id=workspace_id,
                observable_properties=query_props,
                label="pre_runner",
            )
            manifest["pre_runner_semantic_queries_path"] = str(pre_query_path)

        print("[simuhome-e2e] Starting SimuHome sidecar simulator", flush=True)
        sim_env_values = {
            "SIMUHOME_SIM_SPEED": str(args.sim_speed),
            "SIMUHOME_HA_SYNC_SECONDS": str(args.sim_ha_sync_seconds),
        }
        if args.sim_tick_seconds is not None:
            sim_env_values["SIMUHOME_TICK_SECONDS"] = str(args.sim_tick_seconds)
        sim_env = _env_with_pythonpath(sim_env_values)
        sim_cmd = [
            sys.executable,
            str(SIMUHOME_DIR / "simulate_simuhome_scenario.py"),
            str(scenario_path),
            "--speed",
            str(args.sim_speed),
            "--ha-sync-seconds",
            str(args.sim_ha_sync_seconds),
        ]
        if args.sim_tick_seconds is not None:
            sim_cmd.extend(["--tick-seconds", str(args.sim_tick_seconds)])
        sim_proc = _start_process(
            sim_cmd,
            cwd=PROJECT_ROOT,
            env=sim_env,
            log_path=result_dir / "sidecar.log",
        )
        time.sleep(args.sidecar_warmup_seconds)
        _ensure_running("SimuHome sidecar", sim_proc, log_path=result_dir / "sidecar.log")
        _ensure_running("HASP", hasp_proc, log_path=result_dir / "hasp.log")

        print("[simuhome-e2e] Generating temporary e2e case", flush=True)
        case = _build_qt2_case(
            episode=episode,
            scenario_path=scenario_path,
            workspace_id=workspace_id,
            settle_seconds=args.settle_seconds,
            assertion_timeout=args.assertion_timeout,
            assertion_poll_interval=args.assertion_poll_interval,
            deltas=delta_overrides,
        )
        case_path = work_dir / f"{scenario_path.stem}.case.json"
        case_path.write_text(json.dumps(case, indent=2, ensure_ascii=True), encoding="utf-8")
        manifest["case_path"] = str(case_path)
        manifest["case"] = case

        print(f"[simuhome-e2e] Running AMI e2e case via {E2E_RUNNER}", flush=True)
        runner_env = _env_with_pythonpath({"YGGDRASIL_URL": hasp_url})
        results_csv = args.results_csv.expanduser().resolve()
        manifest["results_csv"] = str(results_csv)
        runner_cmd = [
            sys.executable,
            str(E2E_RUNNER),
            "--case",
            str(case_path),
            "--write-results",
            "--results-csv",
            str(results_csv),
            "--response-timeout",
            str(args.response_timeout),
            "--discovery-timeout",
            str(args.discovery_timeout),
            "--settle-seconds",
            str(args.settle_seconds),
            "--log-level",
            args.runner_log_level,
        ]
        if args.no_td_sosa:
            runner_cmd.append("--no-td-sosa")
        if args.clear_signifiers:
            runner_cmd.append("--clear-signifiers")
        if args.clear_signifiers_per_case:
            runner_cmd.append("--clear-signifiers-per-case")
        if args.dump_prompts:
            runner_cmd.append("--dump-prompts")
        if args.prompt_dump_dir:
            prompt_dump_dir = args.prompt_dump_dir.expanduser().resolve()
            manifest["prompt_dump_dir"] = str(prompt_dump_dir)
            runner_cmd.extend(["--prompt-dump-dir", str(prompt_dump_dir)])
        if args.skip_prewarm:
            runner_cmd.append("--skip-prewarm")

        runner_returncode = _run_command_streaming(
            runner_cmd,
            env=runner_env,
            log_path=result_dir / "run_cases.log",
            check=False,
            watch={"HASP": hasp_proc, "SimuHome sidecar": sim_proc},
        )
        manifest["runner_returncode"] = runner_returncode
        if query_props:
            post_query_path = _save_semantic_queries(
                result_dir=result_dir,
                hasp_url=hasp_url,
                workspace_id=workspace_id,
                observable_properties=query_props,
                label="post_runner",
            )
            manifest["post_runner_semantic_queries_path"] = str(post_query_path)
        manifest["finished_at"] = dt.datetime.now().astimezone().isoformat()
        _write_manifest(result_dir / "manifest.json", manifest)
        print(f"Results written to {result_dir}")
        return runner_returncode
    finally:
        print("[simuhome-e2e] Cleaning up subprocesses and Home Assistant workspace", flush=True)
        _stop_process(sim_proc)
        _stop_process(hasp_proc)
        if not args.keep_workspace:
            remove_log = result_dir / "remove_virtual_devices.log"
            try:
                _run_command(
                    [sys.executable, str(SIMUHOME_DIR / "remove_virtual_devices.py"), area_name],
                    env=_env_with_pythonpath(),
                    log_path=remove_log,
                    check=False,
                )
            except Exception as exc:
                (result_dir / "cleanup_error.txt").write_text(str(exc), encoding="utf-8")
        if args.delete_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        manifest["cleanup_finished_at"] = dt.datetime.now().astimezone().isoformat()
        _write_manifest(result_dir / "manifest.json", manifest)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SimuHome JSON through HA, HASP, sidecar, and e2e runner.")
    parser.add_argument("scenario", type=Path, help="Path to a SimuHome benchmark JSON")
    parser.add_argument(
        "--virtual-yaml-dir",
        type=Path,
        required=True,
        help="Host directory mapped to Home Assistant /config/custom_components/virtual",
    )
    parser.add_argument("--area-name", help="Home Assistant area name to create/use")
    parser.add_argument("--results-dir", type=Path, help="Output directory; default tests/simuhome/results/<run_id>")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=SIMUHOME_DIR / "results.csv",
        help="Cumulative per-case results CSV (same schema as run_cases.py); default tests/simuhome/results.csv",
    )
    parser.add_argument("--hasp-port", type=int, default=8080)
    parser.add_argument("--hasp-log-level", default="info")
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--sim-tick-seconds", type=float, default=None)
    parser.add_argument("--sim-speed", type=float, default=1.0)
    parser.add_argument("--sim-ha-sync-seconds", type=float, default=1.0)
    parser.add_argument("--sidecar-warmup-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=15.0)
    parser.add_argument("--assertion-timeout", type=float, default=90.0)
    parser.add_argument("--assertion-poll-interval", type=float, default=2.0)
    parser.add_argument("--response-timeout", type=float, default=1620.0)
    parser.add_argument("--discovery-timeout", type=float, default=900.0)
    parser.add_argument("--runner-log-level", default="INFO")
    parser.add_argument("--delta", action="append", default=[], help="Override assertion delta, e.g. temperature=0.1")
    parser.add_argument("--clear-signifiers", action="store_true")
    parser.add_argument(
        "--no-td-sosa",
        action="store_true",
        help="Start ygg_ha_adapter.py instead of hasp.py (no TD-SOSA semantics) and skip semantic queries.",
    )
    parser.add_argument("--clear-signifiers-per-case", action="store_true")
    parser.add_argument("--dump-prompts", action="store_true")
    parser.add_argument(
        "--prompt-dump-dir",
        type=Path,
        help="Directory for --dump-prompts output; default <e2e-lab308e dir>/prompt_dumps.",
    )
    parser.add_argument("--skip-prewarm", action="store_true")
    parser.add_argument("--keep-workspace", action="store_true", help="Do not remove the HA Virtual Devices workspace")
    parser.add_argument("--delete-work-dir", action="store_true", help="Delete generated YAML/case working directory")
    return parser.parse_args(argv)


def main() -> int:
    return run(parse_args(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
