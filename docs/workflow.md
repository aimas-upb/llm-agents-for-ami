# Request Workflow

This page follows one user request through the three agents: what each step does, which file does
it, and which messages pass between agents. Paths are relative to `ami_agents/`.

```
user ──► UserAssistant ──GOAL_REQUEST──► InteractionSolver ──ENV_*/SIGNIFIER_*/GUARD_*──► EnvExplorer
              │  ▲                              │                                            │
              │  └────────── PLAN_CREATED ◄─────┘                                            │
              └── ENV_STATE / ENV_CAPABILITIES requests ────────────────────────────────────►┘
```

---

## 0. Startup

- **EnvExplorer** crawls the HMAS platform at `YGGDRASIL_URL`, which is HASP or SHTD
  (`behaviors/initial_discovery_behaviour.py`).
  - It builds the TD graph and subscribes to changes (`event_processing_behaviour.py`).
  - It opens the experience memory (`experience_engine_init_behaviour.py`).
  - Then it sends `ENV_DISCOVERY_COMPLETE`.
- **InteractionSolver** waits for that message (`behaviours/environment_ready.py`) before it plans.

## 1. UserAssistant: one FSM per utterance

`UserMessageBehaviour` (`agents/user_assistant/behaviours/user_message.py`) only receives. It
opens a request in the `RequestRegistry` (`registry.py`) and starts one `UserRequestBehaviour`
(`behaviours/user_request.py`) per utterance. SPADE interleaves concurrent requests.

```
SEGMENTING ─► EXTRACTING ─► AWAITING_PLAN ─► SUMMARIZING ─► AWAITING_CONFIRMATION ─► SUBMITTING ─► DONE
                  └─► DONE (queries are answered inline)
```

### SEGMENTING

`pipeline.segment_into_atomic_intents` makes one call with `ATOMIC_SEGMENTATION_SYSTEM_PROMPT`
(config `user_assistant.llm.atomic_segmentation`). It splits the utterance into atomic intents,
each typed `GOAL_REQUEST`, `ENV_STATE_REQUEST` or `ENV_CAPABILITIES_REQUEST`. Goals carry
qualifiers:

| Qualifier group | Values | Meaning |
|---|---|---|
| specificity (exactly one) | `explicit`, `incomplete`, `ambiguous` | names device and action / leaves a slot open / states a wish or a discomfort |
| kind (exactly one) | `achievement`, `maintenance` | reach a state once / keep it |
| structure (any) | `logical_dependency`, `temporal_dependency`, `sequential_dependency` | conditioned on another goal or a state / on time / several instructions to one device |

Segmentation is purely linguistic. It sees no environment.

### EXTRACTING: every intent structured concurrently

| Intent | Behaviour | What happens |
|---|---|---|
| `ENV_STATE_REQUEST` | `behaviours/env_state_structuring.py` | Parsed with `ENV_STATE_REQUEST_PARSER_PROMPT`, resolved by EnvExplorer, answered in code (`utils/state_answers.py`) or with `STATE_ANSWER_PROMPT` (`behaviours/state_answer.py`). |
| `ENV_CAPABILITIES_REQUEST` | `behaviours/env_capability_structuring.py` | Parsed with `ENV_CAPABILITIES_REQUEST_PARSER_PROMPT` (performative `query` or `query_if`), answered in code (`utils/capability_answers.py`) or with `CAPABILITY_ANSWER_PROMPT` (`behaviours/capability_answer.py`). |
| `GOAL_REQUEST` | `behaviours/goal_structuring.py` | See below. |

A `GOAL_REQUEST` is structured in four steps:

1. **Route** (`utils/goal_routing.py`). A goal is *simple* if it is an achievement with no
   dependency qualifier, and *dependency* otherwise. The route decides the prompt and how much
   context is fetched:
   - actions only, for simple explicit or incomplete goals;
   - actions + environment variables, for simple ambiguous goals;
   - actions + device properties + environment variables, for dependency goals.
2. **Context.** An `ENV_CAPABILITIES_REQUEST` with `detail_level: goal_context` goes to
   EnvExplorer. It is scoped to the rooms or device families the goal names once the home is
   larger than `env_explorer.actuation_context` allows.
3. **Structure.** One call with `GOAL_STRUCTURING_SIMPLE_PROMPT` or
   `GOAL_STRUCTURING_DEPENDENCY_PROMPT` (config `intent_parsing`) produces a `GoalStructure`
   (`shared/models/goal_structure.py`): goals, predicates, roles, timing.
4. **Resolve** (`utils/goal_resolution.py`). An incomplete goal the home makes unambiguous is
   promoted to explicit, and a missing room is filled in when only one fits. A goal that only the
   user can complete ("turn it off") is marked for clarification. It never reaches the ISA, and
   the UA asks the question.

### AWAITING_PLAN

The structured goals go to the ISA as one RPC `GOAL_REQUEST`, and the reply is a `PLAN_CREATED`.

### SUMMARIZING and AWAITING_CONFIRMATION

Clarification and unplanned entries are split off and reported. An action plan is summarised with
`PLAN_SUMMARY_SYSTEM_PROMPT` and waits for yes/no. After
`user_assistant.planning.confirmation_timeout` (default 5 s) it is taken as confirmed. A plan with
no action nodes runs without asking.

### SUBMITTING

A `PLAN_EXECUTE_REQUEST` goes to the UA itself, so the plan outlives the request that made it.

## 2. InteractionSolver: one path per goal kind

`GoalRequestBehaviour` (`agents/interaction_solver/behaviours/goal_request.py`) routes every
structure through the selectors in `ROUTES`, in this order:

| Path | Selector | Behaviour | How it plans |
|---|---|---|---|
| explicit | `deterministic_goal` (`utils/explicit_goal_plan.py`) | `explicit_goal_planning.py` | No LLM. `set` gives a one-action tree; `modify` gives read → compute → set (`utils/modify_goal_plan.py`). What cannot be derived this way goes to `explicit_goal_llm_planning.py`: the small model, against that one device, using `MODIFY_RECIPE_PROMPT` where a recipe is needed. |
| incomplete | `incomplete_goal` (`utils/incomplete_goal_plan.py`) | `incomplete_goal_planning.py` | A remembered plan, if any, is replayed. Otherwise candidate devices → choose: each chosen device is planned as an explicit goal, or the entry is impossible, or a clarification. |
| ambiguous | `ambiguous_goal` (`utils/ambiguous_goal_plan.py`) | `ambiguous_goal_planning.py` | Context per implied variable (`utils/specific_capability_context.py`), then the general planner (`BT_PLANNING_SYSTEM_PROMPT`). A remembered plan is passed as signifier hints. |
| dependency | `dependency_structure` (`utils/dependency_context.py`) | `dependency_goal_planning.py` | Steps listed below. |
| (none) | — | `planning_workflow.py` | Fallback for a structure no selector accepts: plans from the request text with `env_context_query.py` + `bt_plan_generation.py`. |

### Dependency plans

1. **Achievement steps.** Each goal is planned on the explicit / incomplete / ambiguous path. If
   any step is impossible or needs clarification, the whole structure carries that outcome.
2. **Predicates.** Each predicate is resolved to the property it reads (`utils/dependency_context.py`).
3. **Profiling.** A deterministic checklist (`utils/goal_profile.py`). Open questions go to one
   small-model call with `GOAL_PROFILE_PROMPT`; unanswered questions take documented fallbacks.
4. **Times.** Clock times and relative times must agree, and runs must fit their anchors
   (`utils/time_consistency.py`). Otherwise the entry is a clarification.
5. **Condition guards** (see below).
6. **Compile.** Templates turn the structure into one tree plus a runtime that keeps its keys
   current (`utils/dependency_compiler.py`).

### Guards (every plan)

The rules live in `config/planning_guidelines.yaml` (`conditions:`): device states, per device
class or manufacturer, taken from the Matter specifications. Guards are applied in execution
order:

1. **Cycle guard** (`utils/cycle_guard.py`): start / pause / resume / stop are sent only from the
   operational states Matter allows them in.
2. **Power-on guard** (`behaviours/power_on_guard.py`, `utils/power_on_guard.py`): a mode, level or
   operational-state command on a device listed under `power_on_first` gets an on/off check and a
   switch-on, unless the plan already switches it on before.
3. **Condition guards** (`behaviours/condition_guard.py`, `utils/condition_guard.py`,
   `utils/condition_check.py`): each step on a conditioned device is decided by the guard memory
   first, then by one call with `CONDITION_GUARD_PROMPT`. Guards the plan already satisfies are
   dropped.

The plan entries return to the UA in one `PLAN_CREATED`.

## 3. UserAssistant: execution

`PlanManagementBehaviour` (`behaviours/plan_management.py`) handles these messages:

| Message | Effect |
|---|---|
| `PLAN_EXECUTE_REQUEST` | Starts the plan's executor |
| `PLAN_STATUS_REQUEST` | Reports the plan's state |
| `PLAN_CANCEL_REQUEST` | Cancels the plan between ticks |
| `PLAN_EXPLAIN_REQUEST` | Explains the plan |
| `PLAN_TRIGGER_REQUEST` | Wakes a maintenance plan |

Executors (`behaviours/plan_execution.py`) tick once per `run()`, so the UA stays responsive:

| Executor | Lifecycle | Config (`bt_execution.*`) |
|---|---|---|
| `AchievementPlanBehaviour` | ticks until SUCCESS / FAILURE or `max_ticks` | `max_ticks.user_assistant`, `min_tick_yield` |
| `MaintenancePlanBehaviour` | ticks in bursts, stands down between them, ends only on cancel | `maintenance_interval`, `maintenance_trigger_poll` |
| `DependencyPlanBehaviour` | waits on its conditions and the clock, with no tick budget | `dependency_tick_period` |

**The clock** (`shared/utils/clock.py`, `environment.clock`) is the system time by default. With
`CLOCK_SOURCE=http CLOCK_URL=…` it reads a simulator's virtual time (SHTD serves it at
`/_shtd/time`), and the agents do not notice the difference.

## 4. Experience memory

Goals are remembered under an exact key from `shared/utils/goal_signifiers.py:signifier_key`
(kinds `explicit`, `incomplete`, `incomplete_text`, `ambiguous`). The UA uses it to record and the
ISA uses it to recall, so the two cannot drift.

The rules for what is recorded:
- **Successful plans only.**
- **explicit / incomplete**: recorded only when an LLM made the plan (`llm: true`). The whole tree
  is stored and replayed on an exact hit.
- **ambiguous**: always recorded. A context snapshot (taken before the first tick) and the
  affordances used are stored, and on an exact hit they are given to the planner as hints.
- **Never** a plan that was itself replayed from memory (`source: "memory"`).

```
UA   PlanExecutionBehaviour.on_start -> snapshot (ambiguous entries)
UA   on success -> SIGNIFIER_RECORD_EXECUTION_REQUEST {signifier} -> EnvExplorer
ISA  before an LLM call -> SIGNIFIER_MATCH_REQUEST {kind, key} -> EnvExplorer,
     then the community store on a miss
```

Condition-guard decisions follow the same pattern with `GUARD_RECORD_REQUEST` /
`GUARD_MATCH_REQUEST`. Storage is one JSON file per record, in `<storage_dir>/signifiers/` and
`<storage_dir>/guards/` (`shared/memory/src/storage/`).

## 5. EnvExplorer request types

| Message | Behaviour (`agents/env_explorer/behaviors/`) |
|---|---|
| `ENV_CAPABILITIES_REQUEST` (`detail_level`: `summary`, `goal_context`, `actuation_only`) | `environment_capabilities_behaviour.py` |
| `ENV_CAPABILITY_QUERY_REQUEST` | `capability_query_behaviour.py` |
| `ENV_STATE_REQUEST` | `environment_state_behaviour.py` (via `property_resolution_behaviour.py` + `value_retrieval_behaviour.py`) |
| `ENV_RESOLVE_REQUEST` | `property_resolution_behaviour.py`: SPARQL over TDs + HomeOnt + SAREF |
| `ENV_SNAPSHOT_REQUEST` | `environment_snapshot_behaviour.py` |
| `SIGNIFIER_MATCH_REQUEST`, `SIGNIFIER_RECORD_EXECUTION_REQUEST`, `SIGNIFIER_LIST_REQUEST` | `signifier_match_behaviour.py`, `signifier_record_behaviour.py`, `signifier_list_behaviour.py` |
| `GUARD_MATCH_REQUEST`, `GUARD_RECORD_REQUEST` | `guard_match_behaviour.py`, `guard_record_behaviour.py` |

All message types are in `shared/models/messages.py` (`MessageType`). Replies reuse the request's
correlation ID through `shared/utils/spade_rpc.py`.
