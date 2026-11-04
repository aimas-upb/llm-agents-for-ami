"""Find and load the runs a report covers.

A *root* is one attempt of the grid: `<root>/<config>/` holds the episode records
(`<episode>.json`), their scores (`<episode>.eval.json`), `run.json`, `gpu.txt` and
`stats/`. Several roots are several attempts of the same configs (the sbatch runs
ATTEMPT=1, 2, 3 under one OUT_ROOT). A *pooled* directory (`pool_attempts.py`) holds
every attempt's records under `<config>/<episode>.a<attempt>.json`; it is read back
into attempts by that suffix. A *reference* directory is a records directory (or a
root of them) whose configs are shown beside the local ones for comparison.

    configs = discover(roots=[Path("harness/1"), Path("harness/2")],
                       pooled=None, references=[(Path("ref/gpt6"), "gpt-6 reference")])
    configs[0].name, configs[0].attempts[0].label, configs[0].models

Every number per episode comes from `stats.episode_metrics`, every group row from
`stats.aggregate`, so the report agrees with `stats/summary.md` to the digit.
"""

from __future__ import annotations

import json
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# The repo root, so `python tests/simuhome/bench/viz/build_report.py` works as well as
# `python -m tests.simuhome.bench.viz.build_report`.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def _import_stats():
    """`tests.simuhome.bench.stats`. Its `meter` module imports the openai client only
    to wrap its two request classes; the report never calls a model, so when the
    package is not installed a stand-in with those two names is enough."""
    try:
        import openai  # noqa: F401
    except ImportError:
        for name in ("openai", "openai.resources", "openai.resources.chat",
                     "openai.resources.chat.completions", "openai.resources.responses"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["openai.resources.chat.completions"].AsyncCompletions = type(
            "AsyncCompletions", (), {})
        sys.modules["openai.resources.responses"].AsyncResponses = type(
            "AsyncResponses", (), {})
    from tests.simuhome.bench import stats
    return stats


stats = _import_stats()
PHASES = stats.PHASES
TYPES = stats.TYPES
CASES = stats.CASES

# `<episode>.a<attempt>.json` in a pooled directory.
_POOLED = re.compile(r"^(?P<stem>.+)\.a(?P<attempt>[^.]+)$")
# A line of gpu.txt: "job 290455  node ucsc-precis-h200-wn12  partition h200  gpu NVIDIA H200".
_GPU_FIELDS = ("job", "node", "partition", "gpu")
# The phases of the model table, in pipeline order (`run.json` -> variant.steps).
MODEL_STEPS = ("segmentation", "structuring", "planning.explicit", "planning.general",
               "profiling", "guarding", "response", "judge")


@dataclass
class Attempt:
    """One run of a config: its records directory and its rows."""
    label: str
    directory: Optional[Path]
    pairs: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Config:
    """One configuration (a variant name) over every attempt found."""
    name: str
    attempts: List[Attempt] = field(default_factory=list)
    reference: bool = False
    label: Optional[str] = None
    run: Dict[str, Any] = field(default_factory=dict)      # run.json of the first attempt
    gpu: Dict[str, Optional[str]] = field(default_factory=dict)
    models: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.label or self.name

    @property
    def rows(self) -> List[Dict[str, Any]]:
        """Every attempt's rows together: the pooled view."""
        return [row for attempt in self.attempts for row in attempt.rows]


# --------------------------------------------------------------------------
# Finding records
# --------------------------------------------------------------------------

def is_records_dir(directory: Path) -> bool:
    """A directory with at least one episode record in it."""
    if not directory.is_dir():
        return False
    for path in directory.glob("*.json"):
        if path.name.endswith(".eval.json") or path.name in stats.NOT_RECORDS:
            continue
        return True
    return False


def config_dirs(root: Path) -> List[Path]:
    """The config directories under a root: its subdirectories that hold records. A
    root that is itself a records directory counts as one config (its own name)."""
    if is_records_dir(root):
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and is_records_dir(p))


def load_pooled(directory: Path) -> Dict[str, List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]]:
    """A pooled directory's records by attempt, from the `.a<attempt>` suffix of the
    file names (records without one go under attempt "pooled")."""
    out: Dict[str, List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".eval.json") or path.name in stats.NOT_RECORDS:
            continue
        record = _read_json(path)
        if not record or "episode" not in record:
            continue
        match = _POOLED.match(path.stem)
        attempt = match.group("attempt") if match else "pooled"
        scored = path.with_name(path.stem + ".eval.json")
        evaluation = _read_json(scored) if scored.exists() else None
        out.setdefault(attempt, []).append((record, evaluation))
    return out


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Config metadata: run.json, gpu.txt, the models per phase
# --------------------------------------------------------------------------

def read_gpu(directory: Path) -> Dict[str, Optional[str]]:
    """`gpu.txt`: job id, node, partition and GPU model, each None when absent. The
    file is one line of "key value" pairs; a value runs to the next known key."""
    out: Dict[str, Optional[str]] = {k: None for k in _GPU_FIELDS}
    path = directory / "gpu.txt"
    if not path.exists():
        return out
    text = " ".join(path.read_text(encoding="utf-8", errors="replace").split())
    pattern = re.compile(r"\b(" + "|".join(_GPU_FIELDS) + r")\s+(.*?)(?=\s+\b(?:"
                         + "|".join(_GPU_FIELDS) + r")\b|$)")
    for key, value in pattern.findall(text):
        out[key] = value.strip() or None
    return out


def provider_of(step: Dict[str, Any]) -> str:
    """How a step reaches its model: the provider named, else the endpoint ("local
    http://host:port/v1"), else OpenAI when it names an API key, else unknown."""
    if step.get("provider"):
        return str(step["provider"]) + (f" {step['base_url']}" if step.get("base_url") else "")
    if step.get("base_url"):
        return f"local {step['base_url']}"
    if step.get("api_key_env"):
        return "openai"
    return "default"


def model_table(variant: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The model configuration rows: phase, model, provider / endpoint, API, and the
    sampling settings (reasoning effort or temperature, max tokens)."""
    steps = (variant or {}).get("steps") or {}
    names = [s for s in MODEL_STEPS if s in steps] + sorted(s for s in steps if s not in MODEL_STEPS)
    rows = []
    for name in names:
        step = steps.get(name) or {}
        settings = []
        if step.get("reasoning_effort"):
            settings.append(f"effort {step['reasoning_effort']}")
        if step.get("temperature") is not None:
            settings.append(f"T={step['temperature']}")
        if step.get("max_tokens"):
            settings.append(f"max {step['max_tokens']} tok")
        rows.append({"phase": name, "model": step.get("model") or "(default)",
                     "provider": provider_of(step), "api": step.get("api") or "",
                     "settings": ", ".join(settings)})
    return rows


def _meta(config: Config, directory: Path, pairs) -> None:
    """Fill a config's run.json, gpu.txt and model table from one attempt's directory
    (the first that has them); the variant falls back to the first record's."""
    run = _read_json(directory / "run.json") if (directory / "run.json").exists() else None
    if run and not config.run:
        config.run = run
    if not any(config.gpu.values()):
        config.gpu = read_gpu(directory)
    if not config.models:
        variant = (run or {}).get("variant") or (pairs[0][0].get("variant") if pairs else None)
        config.models = model_table(variant or {})


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def _attempt_label(root: Path, index: int) -> str:
    """The attempt's name: the root's directory name ("1", "attempt2"); its position
    when the root is itself a records directory (its name is the config's)."""
    return str(index + 1) if is_records_dir(root) or not root.name else root.name


def _add(configs: Dict[str, Config], name: str, label: str, directory: Optional[Path],
         pairs, reference: bool = False, title: Optional[str] = None) -> None:
    config = configs.setdefault(name, Config(name=name, reference=reference, label=title))
    rows = []
    for record, evaluation in pairs:
        row = stats.episode_metrics(record, evaluation)
        row["config"] = name
        row["attempt"] = label
        rows.append(row)
    config.attempts.append(Attempt(label=label, directory=directory, pairs=list(pairs), rows=rows))
    if directory is not None:
        _meta(config, directory, pairs)


def discover(roots: Iterable[Path], pooled: Optional[Path] = None,
             references: Iterable[Tuple[Path, Optional[str]]] = ()) -> List[Config]:
    """Every config found, local ones first (in the order of the roots, then by name),
    then the references. An episode present in several roots is one row per attempt."""
    configs: Dict[str, Config] = {}
    order: List[str] = []
    # 1. The roots: one attempt each.
    for index, root in enumerate(roots):
        root = Path(root)
        if not root.exists():
            print(f"warning: root {root} does not exist", file=sys.stderr)
            continue
        label = _attempt_label(root, index)
        for directory in config_dirs(root):
            name = directory.name
            pairs = stats.load(directory)
            if name not in configs:
                order.append(name)
            _add(configs, name, label, directory, pairs)
    # 2. A pooled directory: its attempts, less the ones a root already gave.
    if pooled is not None and Path(pooled).exists():
        for directory in config_dirs(Path(pooled)):
            name = directory.name
            for attempt, pairs in load_pooled(directory).items():
                known = {a.label for a in configs.get(name, Config(name)).attempts}
                if attempt in known:
                    continue
                if name not in configs:
                    order.append(name)
                _add(configs, name, attempt, directory, pairs)
    # 3. References: shown beside the locals, never pooled with them.
    for path, title in references:
        path = Path(path)
        if not path.exists():
            print(f"warning: reference {path} does not exist", file=sys.stderr)
            continue
        dirs = config_dirs(path)
        for directory in dirs:
            name = directory.name if len(dirs) > 1 or directory != path else (title or path.name)
            key = f"ref:{name}"
            if key not in configs:
                order.append(key)
            _add(configs, key, "ref", directory, stats.load(directory), reference=True,
                 title=title if len(dirs) == 1 else f"{title or 'ref'}: {name}")
    return [configs[name] for name in order]


# --------------------------------------------------------------------------
# Record compaction (what the episode viewer embeds)
# --------------------------------------------------------------------------

def _size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False))


def _checked_devices(evaluation: Optional[Dict[str, Any]]) -> set:
    """The devices the evaluation checks, whose attributes the compact final state keeps."""
    devices = set()
    for check in ((evaluation or {}).get("breakdown") or {}).get("checks") or []:
        if check.get("device"):
            devices.add(check["device"])
    for target in (evaluation or {}).get("targets_eval") or []:
        if target.get("device_id"):
            devices.add(target["device_id"])
    return devices


def _compact_state(state: Optional[Dict[str, Any]], devices: set) -> Optional[Dict[str, Any]]:
    """A home state reduced to the room readings and the checked devices' attributes."""
    if not state:
        return None
    rooms = {}
    for room, body in (state.get("rooms") or {}).items():
        kept = {"state": body.get("state")}
        picked = [d for d in body.get("devices") or [] if d.get("device_id") in devices]
        if picked:
            kept["devices"] = [{"device_id": d.get("device_id"), "device_type": d.get("device_type"),
                                "attributes": d.get("attributes")} for d in picked]
        rooms[room] = kept
    return {"current_tick": state.get("current_tick"), "current_time": state.get("current_time"),
            "base_time": state.get("base_time"), "rooms": rooms}


def compact_record(record: Dict[str, Any], evaluation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The record as the viewer embeds it: everything the trajectory needs, with the
    snapshots reduced to room readings, the final state to the checked devices, and
    the variant left out (the config page has it). `_stripped` says what was cut."""
    devices = _checked_devices(evaluation)
    out = {k: v for k, v in record.items()
           if k not in ("snapshots", "final_home_state", "variant", "ambient_sensors")}
    stripped = {}
    snapshots = record.get("snapshots") or []
    if snapshots:
        stripped["snapshots"] = {"bytes": _size(snapshots), "count": len(snapshots)}
        out["snapshots"] = [{"tick": s.get("tick"),
                             **(_compact_state(s.get("state"), devices) or {})} for s in snapshots]
    if record.get("final_home_state"):
        stripped["final_home_state"] = {"bytes": _size(record["final_home_state"])}
        out["final_home_state"] = _compact_state(record["final_home_state"], devices)
    if record.get("ambient_sensors"):
        out["ambient_sensors_count"] = len(record["ambient_sensors"])
    out["_stripped"] = stripped
    out["_eval"] = evaluation
    return out
