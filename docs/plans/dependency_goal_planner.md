# Plan: a dependency goal planner — achievement steps, profiling, template compiler, clock

## Context

Dependency structures (goals with predicates, timing, maintenance) are planned today by
the general workflow from the request's **text**, with the whole home as context: the
roles, timing and predicates the UA worked out are discarded. The user's design replaces
that with:

1. the **achievement steps** planned by the existing per-goal planners (`achieve(G)`);
2. a **profiling** call (small model, `goal_profile_prompt.md`) answering a deterministic
   **checklist** (`build_checklist.py`), with documented **fallbacks**
   (`resolve_profile.py`);
3. a **template compiler** (`bt_templates.md`) turning structure + profile into one tree;
4. a **clock** that is the system time by default and the simulator's virtual time when
   configured, transparent to the agents (part of the environment model).

All three attached designs are used as given; this plan places them in the codebase and
names what has to be written around them.

## 1. Routing
`behaviours/goal_request.py`: a fourth selector `dependency_structure(structure)`
(`structure == "dependency"`, or a simple maintenance goal) → `DependencyGoalPlanningBehaviour`.
The general workflow then only receives plain-text intents.

## 2. Achievement steps first — `achieve(G)`
Each goal is planned on its own as a simple achievement structure, by the existing
behaviours (explicit / incomplete / ambiguous; maintenance goals planned for their action).
Their outcome already names the devices they act on, so a goal is resolved **once**:
- a goal that is impossible / needs clarification / errors → the whole entry carries that
  outcome, and nothing further runs (no profiling call);
- the devices each subtree uses feed the profiling context (§3).

Predicates have no planner; they are resolved deterministically
(`utils/dependency_context.py`):
- device predicate → `find_artifacts(artifact_name, artifact_class, location_class)`
  (several with a quantifier);
- environment predicate (explicit / incomplete) → the readings of the variable in that room
  (the ambiguous context's `_READING_QUERY`);
- ambiguous predicate → the readings of its implied variables, with `condition_direction`.
A predicate with no resolvable property, or no determined value (qualitative "loud") →
unresolvable: the entry is an error naming it (no partial plans).

`specific_goal_context = {"G1": ctx, "P1": ctx, …}`: per id, the same affordance dicts as
for achievement goals (`specific_capability_context`), grouped per artifact with `name`,
most-specific `class`, `location_class` — what a PropertyRef needs; property dicts carry
their tdsosa kind (e.g. `tdsosa:ActuatablePropertyAffordance`).

## 3. Profiling
- `utils/goal_profile_checklist.py`: `build_checklist` — the attached code, on the
  `GoalStructure` model; `InconsistentStructure`.
- **`InconsistentStructure`** means the structure can never be carried out as given. It is
  raised (a) by `build_checklist` when a goal starts `after_goal` on a maintenance goal with
  no `ends`, scope or termination predicate — it would wait forever; checked before any
  model call; and (b) by `resolve_profile` when the model reports
  `"question": "inconsistency"` (prompt rule 5), with no fallback. The behaviour catches it
  and returns one tree-less entry `{"impossible": true, "error": "inconsistent_structure",
  "explanation": <the reason>}`, so the user is told why instead of getting a plan that
  hangs.
- `prompts.py`: `GOAL_PROFILE_PROMPT` — the attached prompt verbatim; `{goal_structure}`,
  `{specific_goal_context}`, `{checklist}` by plain replacement.
- one small-model call (`explicit_llm`) only when the checklist is non-empty.
- `utils/goal_profile.py`: `resolve_profile` — the attached code and fallback table; plus:
  a PropertyRef not present in that id's context counts as unresolved (fallback);
  `ProfileOutputError` → every item falls back.

## 4. Compiler — `utils/dependency_compiler.py`, applying `bt_templates.md` §2–7
Predicates (§2), goal bodies with filter / native / regulated / unobserved maintenance
(§3), end wrappers (§4), guards (§5), start wrappers incl. trigger-before and recurrence
(§6), placement of after / with / completes_with followers and the assembly into heads and
groups (§7) — exactly as specified. The worked example (§8) is a test.

`completes_with` is a timing field of its own, not a value of `starts` ("so A and B finish
together"):
- **No `starts`:** the plan chooses the start. It waits until the ref's time left is no
  more than the goal's run length (`completion_timing`).
- **A `starts` the user gave:** the goal is placed by that start. `completes_with` is only
  checked before planning, and adds nothing to the tree.

Either way, the planner reads the run length and the event's time left once
(`time_consistency.check_completion`). If either is unknown, or the two don't fit, it asks
the user instead of guessing a start. A time with no time in it ("soon") is also a question
(`VagueTime`). A trigger predicate that no goal waits for is a compile error: the structure
lost a link, which is our failure, not the user's.

The templates are written as py_trees built-ins; the tree still has to cross from the ISA
to the UA as JSON IR, so each built-in gets an IR type in `bt_planning/nodes/registry.py`
compiling to that built-in (no new behaviour classes, except the clock-aware ones in §5):

| IR type | py_trees |
|---|---|
| `check` | `behaviours.CheckBlackboardVariableValue` / `…Values` (`and_` / `or_`) |
| `wait_for` | `behaviours.WaitForBlackboardVariableValue` |
| `compare` | `behaviours.CompareBlackboardVariables` |
| `await` | `decorators.Condition(status=SUCCESS)` |
| `repeat` | `decorators.Repeat(num_success=-1)` |
| `inverter` | `decorators.Inverter` |
| `ignore_failure` | `decorators.FailureIsSuccess` (exists) |
| `running` | `behaviours.Running` |
| `set_variable` | `behaviours.SetBlackboardVariable` (static values only) |
| `timer`, `wait_until` | clock-aware, §5 |

Comparison operators travel as their names (`eq ne gt ge lt le`, `and_ or_`) and are mapped
back to `operator` functions when compiling. Blackboard keys follow the templates
(`/artifacts/<title>/<property>[/<field>]`, `/goals/<G>/…`, `/clock/now`), with titles'
dots replaced (py_trees reads a `.` as attribute access).

### Runtime inputs travel with the plan, not in the tree
The templates' pre-tick handlers and derived keys are not tree nodes, and a callable
cannot be serialised. So the plan entry carries a `runtime` section, and the UA installs
the handlers when it runs the plan:
```
"runtime": {
  "reads":   [{"key": "/artifacts/washer_1/operationalState", "url": "…/properties/operationalState", "field": null}],
  "derived": [{"key": "/goals/G2/due", "op": "minus_seconds", "inputs": ["/artifacts/…/endTime"], "args": {"seconds": 60}}],
  "clock":   true
}
```
- `reads`: refreshed before each tick. Done without blocking the agent: each read is
  submitted to the BT nodes' thread pool and its key updated when it lands; a key holds
  the last value read.
- `derived`: named ops (like compute ops, registered in `bt_planning/runtime_ops.py`):
  `minus_seconds` (`/goals/G/due`), `remaining_from_timestamp`, `remaining_from_duration`
  (`/goals/R/remaining`), `abs_deviation` (`/goals/G/deviation`); units converted from the
  property's QUDT unit, read from the TD at compile time and stored in `args`.
- `clock`: write `/clock/now` from the configured clock (§5).
- `PlanExecutionBehaviour` (UA) installs these as `add_pre_tick_handler` on its tree; an
  entry without `runtime` runs as today.

**When runtime is needed — example.** "23 minutes before washer 1 finishes, set
refrigerator 2 to 6 °C": P1 trigger, `offset_direction: "before"`; `time_to_event` names
the washer's remaining-time property (`reads_as: "duration"`). Tree:
`sequence[wait_for /artifacts/washer_1/countdownTime <= 1380, achieve(G1)]`. `wait_for`
only looks at the blackboard key, so the property must be read before every tick →
`runtime.reads`. With an end-time property instead (`reads_as: "timestamp"`), the tree is
`sequence[await(compare /clock/now >= /goals/G1/due), achieve(G1)]`, needing the read of
`endTime` (`reads`), `/clock/now` (`clock`) and `/goals/G1/due = endTime − 1380 s`
recomputed every tick (`derived`). These are not nodes because a reader in one branch goes
stale once a `memory=True` sequence has passed it, a forever-running reader would keep the
root from finishing, and a function cannot travel as JSON. Trees of actions,
`condition` and `wait_condition` (which read their own property) need no runtime.

### Times and offsets
`utils/time_text.py` parses `offset_text` / `ends.text` to seconds ("N seconds / minutes /
hours", combinations) and `time_text` to a time of day ("H:MM", "H:MM AM/PM"). A clock time
is resolved against `/clock/now` when the plan runs (today, or tomorrow if passed), never
at compile time — the ISA does not read the clock. Unparseable words → error entry naming
them.

### Plan lifecycle
A structure that never ends on its own (maintenance with no end, recurrence with no stop)
is a long-running plan: the entry carries `execution_mode: "maintenance"`, and
`PlanManagementBehaviour._classify_type` reads it from the entry (today only from the
envelope). Its root keeps RUNNING; cancelling interrupts it.

## 5. The clock — system time by default, simulated time by configuration

**One abstraction, configured with the environment.** `ami_agents/shared/utils/clock.py`:
```
class Clock:                       now() -> datetime (timezone-aware)
SystemClock                        datetime.now()                         (default)
HttpClock(url, field, fmt)         GET url -> json[field] parsed with fmt  (virtual time)
make_clock(environment_config)     from environment.yaml > environment.clock
```
```yaml
# environment.yaml
environment:
  clock:
    source: "${CLOCK_SOURCE:-system}"          # system | http
    http:
      url: "${CLOCK_URL:-}"                    # e.g. http://localhost:8097/_shtd/time
      field: "now"
      format: "%Y-%m-%d %H:%M:%S"
```
Every time-related piece takes its time from this clock, and nothing else:
- the `/clock/now` pre-tick handler;
- `timer` (replaces `timers.Timer`, which reads wall time): records its start from the
  clock on first tick, SUCCESS when `now - start >= duration`;
- `wait_until`: SUCCESS when the clock reaches the time of day (resolved on first tick);
- the derived-key ops (`remaining_*`, `minus_seconds`) through `/clock/now`.

The existing settling wait (`settle` / action `settling_time_seconds`, which waits for an
action's effect to show) stays in real time: the times come only from HASP TDs
(`tdsosa:settlingTime`), SimuHome TDs state none, and the achievement path is unchanged.

**SimuHome: the clock is part of the environment model.** SHTD gets a route `GET
/_shtd/time` that proxies the simulator's `GET /api/time` (`{"now": "YYYY-MM-DD HH:MM:SS"}`),
so the agents only know SHTD's address, not the simulator's. Running against SimuHome is
then `CLOCK_SOURCE=http CLOCK_URL=http://localhost:8097/_shtd/time`. The HASP / real
deployment keeps the default.

The clock is read once per tick (the pre-tick handler), cached for the tick; the clock-aware
nodes read `/clock/now` from the blackboard rather than calling the clock themselves, so one
tick sees one time. A clock that cannot be read leaves `/clock/now` unchanged and logs once:
time-based nodes stay RUNNING rather than firing on a stale or wrong time.

Not needed: SimuHome's `schedule_workflow` / `get_workflow_status` / `cancel_workflow`.
The tree itself waits in virtual time (`wait_until`, `timer`) and acts when due; the plan's
status and cancellation are the UA's existing plan-management messages.

## 6. Behaviour — `behaviours/dependency_goal_planning.py`
`DependencyGoalPlanningBehaviour(structures)`: per structure → achievement steps (§2) →
predicate resolution + context (§2) → checklist → profiling (§3) → compile (§4) → one plan
entry `{tree, runtime, explanation, source: "dependency", profile, fallbacks,
execution_mode?, llm}`. The fallback log goes into the explanation and the demo log.
Comments per the commenting rule.

## Tests (unit, no LLM)
- checklist rules; the maintenance-ref inconsistency.
- resolve_profile: answers, unresolved / missing → fallbacks + log, inconsistency raises,
  unknown / duplicate ids, PropertyRef not in context → fallback.
- predicate resolution and per-id context.
- compiler: every template of `bt_templates.md` §2–7 on hand-written structures; the worked
  example (§8); all trees validate.
- IR types compile to the right py_trees classes; operators round-trip.
- clock: `SystemClock`, `HttpClock` (fake HTTP), `make_clock` from config; `timer` and
  `wait_until` advance with a fake clock and not with wall time; a clock failure keeps them
  RUNNING.
- runtime: reads refresh keys without blocking; derived ops; `/clock/now`.
- SHTD `/_shtd/time` with a fake simulator (TestClient).
- behaviour with fake planners and a fake profile call; routing.

## Verification
```bash
conda run -n ami-agents pytest tests/unit ami_agents/shared/memory/tests ami_agents/environment/integration/HomeAssistant/tests -q
```
Offline: compile the prompt's examples and qt4 structures on the seed TDs with a fake
profile and a fake clock. Against the running simulator (yours, on 14000, SHTD on 8097):
an after_delay / at_time plan with `CLOCK_SOURCE=http`, checking it fires in virtual time —
handed over as a command. One live profiling call handed over as a command.

### A test for the user to run: `tests/simuhome/run_dependency_plan.py` (new)
Runs one dependency plan end to end against the running SimuHome stack (simulator on 14000,
SHTD on 8097 — started by the user), in virtual time, and reports whether it fired when due.

```bash
# no LLM call with --fake-profile; one gpt-5-nano call per dependency question without it
CLOCK_SOURCE=http CLOCK_URL=http://localhost:8097/_shtd/time \
conda run -n ami-agents python tests/simuhome/run_dependency_plan.py \
    --episode qt1_feasible_seed_1.json --case delay      # or: at_time, after_goal, trigger
```

What it does, per `--case` (each a hand-built `GoalStructure` on the episode's devices, so no
structuring call is needed; `--phrase "…"` structures a phrase instead, 2 more LLM calls):
- `delay`: "in 2 minutes turn on the living room on/off light 1";
- `at_time`: "at <virtual now + 3 min> turn it on";
- `after_goal`: "turn on light 1, then 1 minute later turn off light 2";
- `trigger`: "when light 1 turns on, turn on light 2" (the script turns light 1 on mid-run).

Steps: build the TD graph from SHTD's base URL (as `modify_live.py` did) → plan with the
real `DependencyGoalPlanningBehaviour` logic in-process (profiling faked with
`--fake-profile`, else the small model) → print the tree and `runtime` → execute it with
`IRExecutor` plus the runtime handlers and an `HttpClock` → poll until done or `--timeout`
(virtual minutes). It prints a timeline of virtual time vs. each action's POST, and checks:
each action fired at or after its due virtual time, not before, and within one tick
period after; the property read back after the action has the expected value. Exit code 0
on pass. It actuates the simulated home and leaves its servers running (they are yours).

## Out of scope
Memory for dependency structures; splitting `BT_PLANNING_SYSTEM_PROMPT`; the config-merge
bug.
