# Configuration

AmI HMAS is configured through the YAML files in [`ami_agents/config/`](../ami_agents/config/) and
environment variables. Every YAML value can read an environment variable with the
`"${VAR:-default}"` form, and a `.env` file at the repo root is loaded by the runners.

## Configuration files

| File | Purpose |
|---|---|
| [`agents.yaml`](../ami_agents/config/agents.yaml) | Agent JIDs, LLM per step, planning settings, memory, timeouts, BT execution limits |
| [`environment.yaml`](../ami_agents/config/environment.yaml) | The environments the agents can run in (and the default one), community API |
| [`environments/<id>.yaml`](../ami_agents/config/environments/) | One environment's definition: discovery, the clock, integration engine |
| [`planning_guidelines.yaml`](../ami_agents/config/planning_guidelines.yaml) | Device conditions per device class / manufacturer (see [Planning](#planning-agentsyaml--interaction_solverplanning)) |
| [`services.yaml`](../ami_agents/config/services.yaml) | External services as virtual Things. It holds disabled examples only. |

---

## Environment variables

### Agents

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — (required) | OpenAI (or compatible) API key |
| `OPENAI_REASONING_EFFORT` | — | UA fallback reasoning effort for a step whose config gives none |
| `YGGDRASIL_URL` | the default environment's URL | The HMAS platform the agents discover: HASP (`:8080`) or SHTD (`:8097`); overrides the default environment's URL |
| `CLOCK_SOURCE`, `CLOCK_URL` | `system`, — | Where plans read the time (see [Clock](#clock-environmentsidyaml--clock)) |
| `UA_CONFIRMATION_TIMEOUT` | `5.0` | Seconds an action plan waits for yes/no before it is taken as confirmed |
| `EXPLICIT_PLANNING_MODEL`, `EXPLICIT_PLANNING_REASONING_EFFORT`, `EXPLICIT_PLANNING_ATTEMPTS` | `gpt-6-luna`, `low`, `2` | The ISA's small model |
| `PLANNING_GUIDELINES_PATH` | repo file | Another device-conditions file |
| `DEPENDENCY_POLL_SECONDS`, `DEPENDENCY_DEFAULT_TOLERANCE`, `DEPENDENCY_STATED_TIME_TOLERANCE` | `30`, `0.5`, `60` | Dependency plans (see below) |
| `BT_MAX_TICKS_DEFAULT`, `BT_MAX_TICKS_USER_ASSISTANT`, `BT_MIN_TICK_YIELD`, `BT_MAINTENANCE_INTERVAL`, `BT_MAINTENANCE_TRIGGER_POLL`, `BT_DEPENDENCY_TICK_PERIOD` | `10`, `100`, `0.01`, `180`, `10`, `1.0` | Plan execution limits and cadence |
| `ACTUATION_CTX_MAX_ARTIFACTS`, `ACTUATION_CTX_MAX_ACTIONS` | `15`, `50` | Above both, the goal context is scoped to the rooms / device families the goal names |
| `LLM_TIMEOUT_DEFAULT`, `LLM_TIMEOUT_REASONING`, `HTTP_CLIENT_TIMEOUT`, `RPC_TIMEOUT`, `ENV_MESSAGE_TIMEOUT`, `SIGNIFIER_MATCH_TIMEOUT` | see `agents.yaml` | Timeouts |
| `COMMUNITY_API_URL`, `COMMUNITY_API_TIMEOUT` | `http://localhost:8085`, `10` | Community signifier store |
| `SIGNIFIER_LIMIT`, `SHACL_VALIDATION_ENABLED`, `SHACL_CACHE_ENABLED` | `10000`, `true`, `false` | Memory engine |
| `SPADE_USER_ASSISTANT_PASSWORD`, `SPADE_ENV_EXPLORER_PASSWORD`, `SPADE_INTERACTION_SOLVER_PASSWORD` | — | XMPP passwords in `agents.yaml` |
| `ENV_DISCOVERY_METHOD`, `CURRENT_ENVIRONMENT` | `mdns`, `lab308` | Discovery method; the entry of `environments.entries` to start in |
| `AMI_USER_JID` | — | XMPP address of the user's client, for the UA's end-of-plan notices (`main.py`); unset, they are dropped |
| `VERBOSE_LOGGING`, `NO_COLOR` / `AMI_NO_COLOR` | — | Logging detail; uncoloured console |

### Console runners

`tests/manual_interactive_cli.py` and `tests/e2e-lab308e/run_cases.py` also read:

| Variable | Default | Description |
|---|---|---|
| `SPADE_SERVER`, `SPADE_PASSWORD` | `localhost`, `password` | XMPP host and the password for all agent accounts |
| `OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_TIMEOUT` (interactive CLI only) | runner default | Fallback provider settings. The per-step models in `agents.yaml` take precedence. |

### Environments

| Variable | Used by | Description |
|---|---|---|
| `HA_URL`, `HA_TOKEN`, `AREAS`, `BASE_WS_URI` | HASP | Home Assistant WebSocket URL and token, areas to map, public base URI |
| `SEMANTIC_CONFIG`, `TD_SOSA_SETTLING_TIMES` | HASP | Semantic mapping file; per-device settling times (JSON) |
| `SIMULATOR_API_BASE_URL`, `SHTD_HOME`, `SHTD_BASE_URL` | SHTD | Same as `--sim`, `--home`, and the public base URL |
| `SHTD_AMBIENT_SENSORS` | SHTD | `0` turns the virtual ambient sensors off |

---

## Environment discovery (`environments/<id>.yaml`)

The agents discover one HMAS platform: the one of the default environment (see
[Environments](#environments-and-community-store-environmentyaml)), loaded as the `environment`
section. Point `YGGDRASIL_URL` at HASP or SHTD, or set the integration engine in the definition file:

```yaml
# environments/lab308.yaml
integration_engine:
  environment_type: 'yggdrasil'
  yggdrasil:
    url: "${YGGDRASIL_URL:-http://localhost:8080/}"
```

The other discovery methods (`direct` TD directory, `well_known` URI, `mdns`) are configured under
`environment.direct`, `environment.well_known` and `environment.mdns`, selected by
`environment.discovery_method`.

### Clock (`environments/<id>.yaml > clock`)

Plans that wait on time (dependency plans) read the clock through `shared/utils/clock.py`:

```yaml
clock:
  source: "${CLOCK_SOURCE:-system}"   # or "http"
  http:
    url: "${CLOCK_URL:-}"             # e.g. http://localhost:8097/_shtd/time
    field: "now"
    format: "%Y-%m-%d %H:%M:%S"
```

With `CLOCK_SOURCE=http CLOCK_URL=http://localhost:8097/_shtd/time`, the agents run on SimuHome's
virtual time without knowing it.

---

## LLM configuration (`agents.yaml`)

Shared defaults live under `default_llm`: the provider, its API key (`null` means
`OPENAI_API_KEY`), the default model / temperature / max tokens, and the retry timeout. Each
step picks its own model; anything a step omits falls back to `default_llm`. Model names
starting with `o` or `gpt-5` are treated as reasoning models: they take `reasoning_effort` instead of
`temperature`, `max_completion_tokens` instead of `max_tokens`, and a longer timeout.

| Step | Key | Notes |
|---|---|---|
| UA segmentation | `user_assistant.llm.atomic_segmentation` | also the UA default client: the plan summary uses it |
| UA query and goal structuring | `user_assistant.llm.intent_parsing` | |
| UA capability answer | `user_assistant.llm.capabilities_analysis` | |
| UA state answer | `user_assistant.llm.state_answer` | |
| UA plan summary | `user_assistant.llm.plan_summarization` | present but not read today; see above |
| ISA general planner (ambiguous goals, text fallback) | `interaction_solver.planning.llm_planning` | `max_planning_attempts` |
| ISA small model (explicit fallback, modify recipes, profiling, condition guards) | `interaction_solver.planning.explicit_planning` | env-overridable |

```yaml
user_assistant:
  llm:
    atomic_segmentation: {model: "gpt-6-luna", reasoning_effort: "low"}
    intent_parsing:      {model: "gpt-6-luna", reasoning_effort: "low"}
interaction_solver:
  planning:
    llm_planning:
      model: "gpt-6-luna"
      reasoning_effort: "low"
      max_planning_attempts: 3
    explicit_planning:
      model: "${EXPLICIT_PLANNING_MODEL:-gpt-6-luna}"
      reasoning_effort: "${EXPLICIT_PLANNING_REASONING_EFFORT:-low}"
      max_planning_attempts: "${EXPLICIT_PLANNING_ATTEMPTS:-2}"
```

The SimuHome bench sets models per step in its own variant files, without touching
`agents.yaml` (see [evaluation-simuhome.md](evaluation-simuhome.md#models-per-step)).

---

## Planning (`agents.yaml > interaction_solver.planning`)

Dependency structures (conditions, timing, maintenance) are planned by templates, with these
settings under `planning.dependency`:

| Key | Default | Meaning |
|---|---|---|
| `poll_seconds` | 30 | how often a regulated maintenance goal reads its value and acts again |
| `default_tolerance` | 0.5 | allowed deviation from a numeric target when the device states none |
| `stated_time_tolerance_seconds` | 60 | a start / end the user gave both as a clock time and relative to an anchor ("at 5:48 PM, which is 23 minutes before washer 1 finishes"): how far apart the two may be before the user is asked which one they mean |

Device conditions live in their own file,
[`ami_agents/config/planning_guidelines.yaml`](../ami_agents/config/planning_guidelines.yaml),
under `conditions:`. Each one is the device state that must hold before a command,
per device class or manufacturer, taken from the Matter specifications ("the
operational state must not be Running before the spin speed changes"). An entry has an
`id`, `applies_to` (`classes`, matched with their subclasses, and `manufacturers`, an
exact match), `commands` (the action command classes it constrains), optional
`acts_upon` (only actions writing a property of these classes) and its `text`.
`planning.guidelines.path` (`PLANNING_GUIDELINES_PATH`) points to another file; empty
uses the default one.

- `power_on_first` is applied deterministically to every plan
  (`interaction_solver/utils/power_on_guard.py`). In front of a mode, level or
  operational-state command on a device the entry lists, the plan checks the device's
  on/off state and switches it on first, unless the plan already switches that device
  on before the command, in execution order (dependency plans included). A device
  without a single on/off command is left unguarded, with a note.
- The cycle commands (start, pause, resume, stop: every action of class
  `homeont:SetOperationalStateCommand`, on any device whose operational state it acts
  upon, for example laundry washers, dryers and dishwashers) are not in the file. Matter
  allows them only from certain states (Operational State, Tables 3 and 4), and no action
  can establish those states without inventing a cycle. So `utils/cycle_guard.py` adds a
  check-only guard in code, before the power-on guard: a pause is sent only while
  Running, a start or resume only when not in Error, a stop only when not already
  Stopped. A command already in its target state is skipped as done; otherwise the step
  fails without sending.
- Every other condition goes through the condition-guard check, on simple and dependency
  plans alike (`utils/planning_conditions.py`, `utils/condition_check.py`,
  `utils/condition_guard.py`). Each step on a device with a condition is decided from
  the guard memory first, then by one small-model call (the profiling model). The
  answer is a check (property, `==` / `!=`, value) and the action that establishes it,
  or "no guard". A guard that pauses, stops or switches off is followed by the matching
  resume, start or switch-on; if none is found, the step is left unguarded. A guard the
  plan already satisfies, earlier in execution order, is removed again. The model's
  device-level decisions are remembered in the guard store after a successful run.

Other keys: `context_gathering.timeout` (also the wait for environment readiness) and
`community` (`enabled`, `api_url`, `timeout`).

---

## Experience memory (`agents.yaml > env_explorer.experience_engine`)

Signifiers are stored and matched by an **exact key** (`shared/utils/goal_signifiers.py`), with no
similarity threshold. What is recorded is described in
[workflow.md § Experience memory](workflow.md#4-experience-memory).

```yaml
env_explorer:
  experience_engine:
    signifier_limit: "${SIGNIFIER_LIMIT:-10000}"
    shacl_validation_enabled: "${SHACL_VALIDATION_ENABLED:-true}"   # kept; not used for matching
    enable_shacl_cache: "${SHACL_CACHE_ENABLED:-false}"
    # storage_dir: optional; default ami_agents/shared/memory/storage
```

Records are JSON files under `<storage_dir>/signifiers/` and `<storage_dir>/guards/`. To start from
an empty memory, delete those two directories.

---

## Environments and community store (`environment.yaml`)

`environment.yaml` lists the environments and marks the one the agents start in; each entry's
full definition is its own file under `environments/`:

```yaml
environments:
  default: "${CURRENT_ENVIRONMENT:-lab308}"
  entries:
    lab308: "environments/lab308.yaml"        # Yggdrasil :8080, HA :8123
    homebench: "environments/homebench.yaml"  # Yggdrasil :8090, HA :8124

community:
  api_url: "${COMMUNITY_API_URL:-http://localhost:8085}"
  timeout: "${COMMUNITY_TIMEOUT:-10.0}"
```

`ConfigLoader.load_environment_config()` loads the default's file as the `environment` section; a
default that is not listed is an error. To add a home, copy a definition file and list it. Several
homes can share signifiers through one community store; start it as described in
[installation.md](installation.md#community-signifier-store-optional).
