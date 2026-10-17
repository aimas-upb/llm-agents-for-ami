# Plan: a SimuHome reasoning harness — our pipeline, in-process, measured per step

## Context

We want to measure our agents' **reasoning** on SimuHome's six question types (qt1, qt2,
qt3, qt4-1/2/3; feasible and infeasible, 50 episodes each), not their messaging. The
harness runs the same steps the agents run — segmentation → structuring → planning →
execution → response — in one process, with no SPADE/XMPP. Each step is timed and its LLM use
counted, so we can compute time statistics per step. For qt4 the planning time is split
into the achievement steps and the dependency planning. The harness reports execution
success rates per question type, feasible and infeasible separately. Each step's LLM is
configurable (OpenAI models, or local ones through Ollama), so model choices can be run
as a sweep.

Constraints from you:
- **No change to the UA's segmentation or structuring.** Anything benchmark-specific
  (location, current time) is a test-only option.
- **Deterministic first.** Feasible episodes and the execution check on infeasible ones
  are deterministic. The only LLM judge is SimuHome's, on the UA's reply to infeasible
  requests (gpt-4o-mini).
- The lockstep clock exists in the tests only.

Everything new lives in `tests/simuhome/bench/`. Production changes are two small ISA
ones (§9).

## 1. Layout — `tests/simuhome/bench/`

| File | Holds |
|---|---|
| `models.py` | per-step model specs and client factory (OpenAI / Ollama) |
| `meter.py` | per-step timing, LLM calls, tokens |
| `stubs.py` | in-process EnvExplorer, UA and ISA stand-ins; memory |
| `pipeline.py` | the four steps for one episode |
| `execution.py` | tick loop (immediate plans) and lockstep (dependency plans) |
| `toollog.py` | request-time EnvExplorer queries → SimuHome tool invocations |
| `run.py` | CLI: run episodes for one model variant → records |
| `evaluate.py` | CLI: score records, deterministic checks + judge (separate process, see §7) |
| `sweep.py` | CLI: run + evaluate a grid of model variants |
| `report.py` | CLI: success rates and time statistics across variants |

## 2. Models per step — `models.py`

A variant is a YAML file with one model spec per step:
```yaml
name: nano-plan_mini-struct
segmentation:  {model: gpt-5-mini, reasoning_effort: low}
structuring:   {model: gpt-5-mini, reasoning_effort: medium}   # goals and state queries
planning:
  explicit:    {model: gpt-5-nano, reasoning_effort: low}      # explicit LLM, modify recipe, profiling
  general:     {model: gpt-5-mini, reasoning_effort: medium}   # ambiguous BT planner
response:      {model: gpt-5-mini, reasoning_effort: low}      # plan summary, state answers
judge:         {model: gpt-4o-mini, temperature: 0.5}          # fixed across a sweep
# local:       {model: qwen3:8b, base_url: http://localhost:11434/v1, temperature: 0}
```
- Spec fields: `model`, `base_url` (absent = OpenAI), `api_key_env`, `temperature`,
  `reasoning_effort`, `max_tokens`.
- Clients are `AsyncOpenAI(base_url=…)`, as in `eval_atomic_segmentation.py:483-515`.
  Ollama serves the OpenAI API at `/v1`.
- Reasoning-model handling follows the agents' rule (name starts with `o` / `gpt-5` →
  `reasoning_effort`, `max_completion_tokens`; else `temperature`, `max_tokens`).

How each step gets its client, without touching production code:
- **Segmentation:** reads `agent.llm_client` / `llm_model` / `build_llm_kwargs` → set on the
  UA stub.
- **Structuring:** builds its client inside the behaviour with
  `build_behaviour_llm_client(config, "intent_parsing")`. That builder hard-codes the
  OpenAI URL, so the harness replaces the name in the behaviour modules'
  namespace (`goal_structuring`, `env_state_structuring`) with its own factory. This is a
  test-only patch.
- **Planning:** the ISA stub carries `explicit_llm` and `llm_client` / `model` / … built
  from the variant.

## 3. Measurement — `meter.py`

- Step timer: a context manager per step and episode records wall time.
- LLM calls: `AsyncCompletions.create` is wrapped (pattern of `LLMCallCounter`,
  `tests/e2e-lab308e/run_cases.py:144,359`, copied, not imported). A context variable
  tags each call with the current step. Each call records latency, input/output tokens
  and the model. Ollama returns `usage` too.
- Steps recorded per episode:

| Step | Recorded |
|---|---|
| `segmentation` | |
| `structuring` | per intent, and total |
| `planning` | per structure, with its route (explicit / incomplete / ambiguous / dependency / clarify) and source (deterministic, LLM, memory) |
| `planning.achievement` | dependency structures only |
| `planning.dependency` | predicates + checklist + profiling + compile, split further |
| `execution` | real wall time, virtual time span, ticks / lockstep steps |
| `discovery` | crawling the home's TDs; informational only, never in statistics (§5) |
| `state_read` | ENV_STATE queries (qt1), the read that answers them |
| `response` | the UA's reply text |

## 4. Stand-ins — `stubs.py`

**EnvExplorer.** It is built from SHTD over HTTP: GET the home, room and artifact TDs, as
the real explorer crawls them. The same SimpleNamespace shape as
`structure_goal.py:explorer_stub`, so
- goal context and candidate resolution reuse `EnvironmentCapabilitiesBehaviour` and
  `resolve_capability_payload` (as `structure_goal.py:wire_explorer`);
- ENV_STATE queries reuse the EnvExplorer state behaviour's handler;
- the ISA's graph is `format_capabilities_detailed_rdf` + `close_types` (as
  `run_dependency_plan.py:build_graph`).

**UA stub.** `config` (as `structure_goal.py:ua_config`), the segmentation client.

**ISA stub.**
- Attributes: `config`, `explicit_llm`, `explicit_bt_planner`, `bt_planner`
  (`AsyncBTPlanner`), `llm_client` / `model` / `temperature` / `reasoning_effort` /
  `max_completion_tokens`, `community_client=None`.
- `_query_env_explorer`: answers ENV_CAPABILITIES (Turtle) and SIGNIFIER_MATCH.
- `add_behaviour(b)`: sets `b.agent`, creates `b._is_done`, schedules `b.run()` and clears
  `_is_done` when it ends, so the real nested behaviours' `join()` works.

**Memory** (`--memory off|run`, default `off`).
- `run` opens one `MemoryStore` in the run's output directory (via `ensure_engine_on_path`).
- Recall goes through the real `signifier_memory.recall` (SIGNIFIER_MATCH answered from
  the store).
- After a successful execution, `build_signifier` + `store.record` — the UA's own rules
  (`ExecutionContext.should_record`, snapshot before the first tick for ambiguous goals).
- With `--workers > 1` each worker has its own store, since episode order and memory
  interact.

## 5. One episode — `pipeline.py`

1. **Load.** Reset the worker's simulator with `initial_home_config`. Build the
   EnvExplorer from SHTD (the crawl and the ISA's graph).
   - This is **not timed**: a real deployment does discovery offline, before any request.
     It is logged as `discovery` (seconds) for information and is excluded from every
     step statistic and every total.
   - It does not feed the tool log either (§7).
   - What the EnvExplorer does per request stays inside the step that asks for it, as in
     the agents: building the goal context and resolving candidates (structuring), the
     ISA's capability graph query (planning), reading state (`state_read`).
2. **Input.** The query as written. `--context prepend` (test-only, default off)
   prefixes "(It is <base_time>. I am in the <user_location>.)" — nothing in the UA
   changes.
3. **Segmentation.** `pipeline.segment_into_atomic_intents`.
4. **Structuring.**
   - GOAL_REQUEST → `GoalStructuringBehaviour` (`run()` awaited directly).
   - ENV_STATE_REQUEST → `EnvStateStructuringBehaviour`.
   - UA-side clarifications (`structure_clarification`) stop that goal here, outcome
     `clarify`, as `user_request.py` does.
5. **Answer state queries.** The EnvExplorer state handler reads the values; recorded
   with what was read. The answer text comes in the response step.
6. **Planning.**
   - Routing: the same selectors and behaviours as `GoalRequestBehaviour`, through the
     extracted routing function (§9).
   - Each entry is recorded with its outcome: tree, `impossible`,
     `requires_clarification`, or `error`.
7. **Execution.**
   - Before executing, the simulator is reset to `initial_home_config` again. Planning
     never actuates, so the plan then starts at tick 0, as SimuHome's "now" assumes. The
     time spent in LLM calls does not consume virtual time.
   - Plans with trees run per §6.
8. **Response** (timed as `response`). The text the UA would send:
   - clarifications: the UA's `About "<intent>": <question>` text (no LLM);
   - plans, including impossible ones: `pipeline.summarize_plan` on the plan JSON;
   - state queries: `StateAnswerBehaviour`.
   Its model is the variant's `response` spec (the UA uses its segmentation client for
   summaries and `state_answer` for answers; the variant sets both).
9. **Snapshot.** `GET /api/home/state` (after 0.5 s, as SimuHome's runner) →
   `final_home_state`.

**Episode outcome:**

| Outcome | When |
|---|---|
| `executed` | some plan ran |
| `impossible` / `clarify` | no tree, and every entry says why |
| `nothing` | segmentation produced nothing |
| `error` | an exception, with the step it happened in |

## 6. Execution — `execution.py`

**Immediate and maintenance plans.** The UA's loop: compile with `compile_node`, `tick_once`
every `bt_execution.min_tick_yield`, up to `bt_execution.max_ticks`.

**Dependency plans — lockstep (tests only):**
```
reset clock origin; PlanRuntime(runtime, clock=HttpClock(shtd/_shtd/time), synchronous=True)
repeat:
    runtime.before_tick(); tree.tick_once()
    wait until no ActionAffordanceNode has _io_in_flight        (async_support.py:119)
    state = fast_forward_to(current_tick + Δ)                   (Δ = --step-seconds, default 10 s)
    keep (tick, state) when it is an observation point
until the tree is done and the horizon is passed, or the horizon is reached
```
- Δ is the precision: an action fires at most Δ late, never early. 10 s is inside the
  ±30 s tolerance.
- The free-running simulator adds the real time each step takes (milliseconds) — logged,
  harmless.
- **Horizon:** the episode's last `at_tick + tolerance` plus one minute. Infeasible qt4
  episodes run to their horizon too, so a wrongly scheduled action is caught.
- **Observation points:** the steps are cut so that they land on every eval goal's
  `at_tick − tol`, `at_tick`, `at_tick + tol`. The goal ticks only decide where states
  are recorded, never what the plan does.
- Settle and timeout waits in nodes stay in real time (SimuHome TDs state none).

**Recorded for every plan:**
- each action fired: URL, payload, HTTP status, real time, virtual tick;
- the tree's final status;
- the state snapshots.

## 7. Evaluation — `evaluate.py`, deterministic only

It runs as a separate process over the run's records, for two reasons:
- the memory engine and SimuHome's evaluators both import as top-level `src`;
- scores can be recomputed without re-running.

It needs a simulator only for qt2 feasible (one worker, started and stopped by itself).

**Tool invocations** (`toollog.py`): only what the per-request EnvExplorer queries hit.
Discovery is offline and never counts, and neither does the ISA's whole-home graph fetch,
which is the discovery result handed over.

| Request-time query | SimuHome tool |
|---|---|
| env-capability query (goal context, candidate resolution, capability questions): each room its answer returns devices from | `get_room_devices {room_id}` |
| env-state query: each room whose environment it reads | `get_room_states {room_id}` |
| env-state query or plan runtime read of a device property | `get_attribute {device_id, …}` |
| action POST during execution | `execute_command {device_id, …}` |

The stand-in EnvExplorer handlers record the rooms and devices in each answer they give.
That record is the tool log, not the SHTD traffic, since discovery made that traffic.

**Criteria.** Infeasible episodes get two scores — a deterministic execution check and
SimuHome's own LLM judge — reported separately and combined (both pass).

| Type | Feasible | Infeasible: execution check | Infeasible: judge |
|---|---|---|---|
| qt1 (state questions) | tools check, and every goal's value **as we read it** equals `current_value` (room state: unit-normalised, small tolerance; device attribute: equal) | no value read or reported for the device that does not exist; the room-state goals in the same query read correctly | `qt1.infeasible` on the UA's state answer |
| qt2 (implicit comfort) | SimuHome's `qt2.feasible` (counterfactual baseline on a simulator) | nothing executed on the infeasible part (below) | `qt2.infeasible` on the UA's reply |
| qt3 (explicit commands) | SimuHome's `qt3.feasible` (asserts on `final_home_state`) | nothing executed on the infeasible part | `qt3.infeasible` on the UA's reply |
| qt4-1/2/3 (timed) | SimuHome's `qt4.feasible`, given a replay client over the lockstep snapshots (its `fast_forward_to(t)` returns the latest recorded state at or before `t`) | nothing executed on the infeasible part, up to the horizon | `qt4.infeasible_1/2/3` on the UA's reply |

**The infeasible part** of an episode is the goals that cannot be carried out; the feasible
goals in the same request may run.

| Episode | Infeasible part | Failure |
|---|---|---|
| qt3, qt2 `nonexistence` | goals on a device or affordance missing from `initial_home_config` | any action on a device standing in for it, in that room |
| qt2 `contextual_saturation` | the room state and direction it asks for | an action in that room on a device acting on that state, in that direction |
| qt4-x | the conflicting timed goals' target devices | any action on those devices |

The infeasible part must also be answered `impossible` / `clarify`, not left out silently.

**Judge.** SimuHome's evaluator modules, unchanged:
- **Panel:** three `gpt-4o-mini` instances (seeds 42 / 43 / 44, temperature 0.5). Its
  `run_judge_panel` requires exactly three. The judge model is configurable and fixed
  across a sweep.
- **What it judges:** the text our UA would reply with. That's the `response` step (§5),
  timed and metered like the others.
- **What it gets:** our pipeline as SimuHome `steps`, one per step: `thought` is that
  step's explanation, `observation` is what the EnvExplorer queries returned (the judge
  prompts check the answer against those observations). The final step is
  `finish {answer: <UA reply>}`.
- **Cost:** 3 calls per judged episode, about 900 for the 300 infeasible episodes.

**Scores:** 1 / 0, with −1 for infrastructure errors (simulator down, LLM transport
errors), which are excluded from the denominator as in SimuHome's `aggregate_results`. A
model's malformed output is a 0.

**Comparability.** The feasible qt2/3/4 rates and the judge scores are SimuHome's own
criteria, so they can be compared with its published numbers. The execution check and our
qt1 value check are ours.

## 8. Running and reporting

**`run.py`**
```
--variant variants/x.yaml  --episodes qt3_feasible,qt4-2_*  --seeds 1-10 | --limit N
--workers N  --memory off|run  --context none|prepend  --step-seconds 10  --out DIR
```
- Each worker is a process with its own simulator and SHTD on free ports
  (`SimuHomeWorker`, `ami_agents/environment/integration/SimuHome/worker.py`), started
  and stopped by the run. The py_trees blackboard is per process, so trees never share it.
- Output: `DIR/<variant>/<episode>.json`, one record per episode (input, intents,
  structures, plan entries, outcome, actions, tool log, final state, snapshots,
  per-step meter), plus `run.json` (variant, git commit, prompt hashes, settings).

**Single episodes, for debugging** — `run.py --episode qt4-2_feasible_seed_1 [--debug]`:
- runs one episode in the foreground, in-process (one worker);
- `--debug` prints each step as it finishes, with what goes in and what comes out:
  - the intents and the GoalStructures;
  - the plan entries, with the tree and runtime;
  - the profiling checklist and answers;
  - the action timeline in virtual time;
  - the evaluation verdict, scored immediately with the same criteria (§7, run in a
    subprocess because of the `src` clash), and why it passed or failed.
- `--stop-after segmentation|structuring|planning` ends early (no execution, no
  actuation).
- `--shtd URL --sim URL`: use servers you already run (e.g. 8097 / 14000) instead of
  starting a worker, so you can watch the home in the inspector.
- The record is written like any other, so a debug run can be compared with a sweep's.

**`sweep.py`**
- A grid YAML: a base variant + lists of alternatives per step (product, or an explicit
  list) → one `run` + `evaluate` per variant.
- **Step cache:** segmentation and structuring results are reused across variants whose
  specs up to that step are identical (keyed by episode + those specs). Their meter
  entries are copied and marked `cached`, so a planning-only sweep doesn't pay
  segmentation and structuring again.

**`report.py`**
- Success rate per variant × question type × feasible / infeasible (n, correct,
  infra-excluded, rate).
- Per-step time statistics (mean, median, p95, max of wall time; LLM calls; tokens in/out)
  per variant and question type; qt4 with `planning.achievement` and
  `planning.dependency` separately.
- Execution: real seconds vs. virtual seconds.
- Outcome breakdown (executed / impossible / clarify / error by step).
- Output: CSV + `report.md` per sweep.

## 9. Production changes (ISA only, no behaviour change)

1. **Routing** in `goal_request.py` is extracted into a pure function: structures →
   buckets (explicit / incomplete / ambiguous / dependency / workflow). The behaviour and
   the harness then call the same routing.
2. **Timings.** `DependencyGoalPlanningBehaviour` records the durations of its phases
   (achievement steps, predicates + checklist, profiling, compile) on `self.timings`.
   The harness reads them; the entry is unchanged.

The UA (segmentation, structuring, prompts, clients) is not touched.

## Tests (unit, no LLM, no simulator)

- the tool-log mapping;
- each criterion on hand-made records, including the replay client with SimuHome's
  `qt4.feasible`;
- the lockstep loop with a fake simulator: it waits for in-flight actions and lands on
  observation points;
- the meter's tagging per step;
- the step cache keys;
- the ISA stub's `add_behaviour` / `join`;
- the routing extraction (existing routing tests stay green).

## Verification

- **Unit tests:** `conda run -n ami-agents pytest tests/unit -q`
- **Offline, no LLM:** a fake-model variant (canned responses) on one episode per type
  checks the plumbing end to end.
- **Smoke, by me:** one paid run on one episode, about 5 LLM calls:
  `run.py --limit 1 --episodes qt3_feasible`.
- **Handed over to you:** full runs and sweeps, with cost estimates. One variant over
  600 episodes is about 3–6 LLM calls per episode, so 2–4k calls; Ollama variants cost
  nothing.
