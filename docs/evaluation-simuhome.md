# Evaluating on SimuHome (and other benchmarks)

The evaluation suite in [`tests/simuhome/bench/`](../tests/simuhome/bench/) measures the agents'
**reasoning** on a benchmark rather than their messaging. Each episode goes through the steps
the agents run:

> segmentation → structuring → planning → execution → response

This happens in one process with no SPADE / XMPP, and it calls the agents' own behaviours
through thin stand-ins (`stubs.py`). Each step is timed and its LLM calls and tokens are
counted. Every episode is then scored.

Every command below runs from the repo root, prefixed with `conda run -n ami-agents`. Each one
says what it costs.

---

## The benchmark

[SimuHome](https://github.com/holi-lab/SimuHome) simulates a home of Matter devices with room
dynamics (temperature, humidity, illuminance, air quality). Its benchmark has 600 episodes:
six request types × feasible / infeasible × 50 seeds. Episodes are named like
`qt4-2_feasible_seed_1`.

| Type | Request | Example |
|---|---|---|
| `qt1` | state question | "how is the humidity in here right now?" |
| `qt2` | implicit: a discomfort, no device named | "the study is really dim, my eyes ache" |
| `qt3` | explicit command | "turn on the living room AC 1, cooling mode, fan at 60 percent" |
| `qt4-1`, `qt4-2`, `qt4-3` | timed / conditioned goals | "23 minutes before washer 1 finishes, set refrigerator 2 to 6 °C" |

An infeasible episode contains a part that cannot be done: a missing device, a state already
reached, or conflicting timed goals. The agents must not act on that part, and must say so.

### Scoring (`evaluate.py`)

| | Feasible | Infeasible |
|---|---|---|
| qt1 | our value check: every value read equals the episode's ground truth | our execution check, plus SimuHome's judge on the reply |
| qt2 | SimuHome's evaluator (counterfactual: did the variable move further than it would have alone?) | our execution check, plus SimuHome's judge |
| qt3 | SimuHome's evaluator (assertions on the final home state) | our execution check, plus SimuHome's judge |
| qt4-x | SimuHome's evaluator (over the recorded timeline) | our execution check, plus SimuHome's judge |

- **Execution check:** nothing was run on the infeasible part, and that part was answered as
  impossible or with a question.
- **Judge:** SimuHome's panel of three LLM calls (default `gpt-4o-mini`) on the UA's reply.
- **Scores:** 1 pass, 0 fail, −1 infrastructure error. −1 is left out of the rates, as in
  SimuHome.
- **Comparability:** the feasible qt2–qt4 rates and the judge scores use SimuHome's own criteria,
  so they compare with its published numbers. The qt1 value check and the execution check are ours.
- **Tool calls:** SimuHome's evaluators also require that certain tools were "invoked"
  (`get_room_devices`, `get_room_states`, …). `toollog.py` maps each per-request EnvExplorer query
  to the tool call it stands for. Discovery happens offline in a deployment, so it never counts.

---

## Prerequisites

1. **SimuHome next to this repo**, at `../SimuHome`, with its dependencies installed in
   `ami-agents`: see [installation.md § Step 5](installation.md#step-5--simuhome-and-shtd-matter-homes-benchmark).
   The episodes are read from `../SimuHome/data/benchmark`; `--benchmark DIR` points elsewhere.
2. **An LLM endpoint.**
   - OpenAI: `OPENAI_API_KEY`, in `.env` at the repo root or exported.
   - Local models: Ollama at `http://localhost:11434/v1`, via `provider: local` (see
     [Models per step](#models-per-step)).
3. **No servers to start.** Each worker starts its own simulator and SHTD on free ports and stops
   them at the end.

---

## 1. Smoke test: one episode

```bash
# ~5 LLM calls (+3 judge calls on an infeasible episode)
conda run -n ami-agents python -m tests.simuhome.bench.run --episode qt4-2_feasible_seed_1 --debug
```

`--episode` runs in the foreground, prints every step (intents, structures, plan trees, the action
timeline in virtual time), and scores the episode immediately. Useful flags:
- `--stop-after segmentation|structuring|planning` stops early, with no actuation;
- `--no-evaluate` skips the score.

## 2. The evaluation harness (recommended)

One config file chooses the model for each phase and the episodes to run. The harness then:
- runs them with scheduled execution (`run_scheduled --skip-existing`);
- scores them (`evaluate --skip-scored`);
- computes the statistics.

A stopped run resumes without paying twice.

```bash
# 1. see the commands and the cost estimate; nothing runs
conda run -n ami-agents python -m tests.simuhome.bench.harness \
    --config tests/simuhome/bench/variants/harness.example.yaml --dry-run

# 2. a small real run first: 2 episodes per type and case = 24 episodes,
#    ~3–6 LLM calls each + 3 judge calls per infeasible episode
conda run -n ami-agents python -m tests.simuhome.bench.harness \
    --config tests/simuhome/bench/variants/harness.example.yaml --limit 2 --workers 2

# 3. the full benchmark: 600 episodes, ~3,000 calls + ~900 judge calls
conda run -n ami-agents python -m tests.simuhome.bench.harness \
    --config tests/simuhome/bench/variants/harness.example.yaml

# 4. recompute the statistics from the records on disk (no LLM calls)
conda run -n ami-agents python -m tests.simuhome.bench.harness \
    --config tests/simuhome/bench/variants/harness.example.yaml --stats-only
```

Overrides: `--episodes qt3_*,qt4-1_*`, `--seeds 1-10,15`, `--limit N` (per type and case),
`--workers N`, `--no-judge` (infeasible episodes get the execution check only, with no LLM
calls), and `--out DIR` (default `tests/simuhome/results/harness`).

The statistics land in `<out>/<name>/stats/summary.md`, with `summary.csv`, `summary.json` and
`episodes.csv` beside it. They cover cost per phase (time, LLM calls, tokens), goals verified and
executed, and the SimuHome score. They are averaged:
- per type × feasible / infeasible;
- per type;
- per case;
- overall.

### Metrics (`stats.py`)

The statistics are computed once per episode (`episodes.csv`), then averaged over groups of
episodes (`summary.*`). `summary.md` starts with a legend of its columns.

**Terms**
- **Eval goal**: one target that SimuHome's answer key checks, for example "washer 1 running in
  Heavy mode at minute 40". A request usually has 1–3.
- **Check**: one expected device value inside an eval goal. `evaluate.py` labels each one *ok*,
  *missing* (never set), *wrong value* or *wrong timing*.
- **Plan entry**: what the ISA returned for one structured goal: a plan, an "impossible" verdict,
  or a question. A dependency plan (qt4-3) can cover several eval goals in one entry.
- **Infrastructure failure**: score −1, meaning the API, the simulator or the judge failed rather
  than the agents. These episodes are left out of every rate and every cost mean. An interrupted
  run costs almost nothing, so keeping it would make the averages look cheaper.

**Cost per phase.** The phases are segmentation, structuring, planning, profiling, guarding and
execution. A seventh bucket, *other*, holds the state reads and the reply to the user.

| Column (`episodes.csv`) | Meaning |
|---|---|
| `<phase>_s` | Wall seconds in the phase. Profiling and guarding run inside planning and are subtracted from it, so the phases add up to `total_s`. |
| `<phase>_calls` | LLM calls the phase made. |
| `<phase>_tok_in`, `<phase>_tok_out` | Prompt and completion tokens. Output includes a reasoning model's reasoning tokens. |
| `total_s`, `total_calls`, `total_tok_in`, `total_tok_out` | Sums over the phases: the end-to-end cost of handling the request. |
| `wall_s` | The episode's full wall time, including discovery and the simulator reset. |

The cost is read from the record's `meter.stages`. Discovery is not counted, because a deployment
does it offline. Steps replayed from a sweep's step cache are not counted either. A record written
before the meter kept stages has no cost columns.

**Goals achieved.** Two complementary views:

| Column | Meaning | Applies to |
|---|---|---|
| `goals_total` | Eval goals in the request. | feasible |
| `goals_verified` | Eval goals SimuHome confirms were fully reached (every check *ok*). This is the ground truth for how many of the things asked for actually happened. | feasible |
| `goals_verified_frac` | `goals_verified / goals_total`. This is partial credit where SimuHome's score is all-or-nothing. | feasible |
| `multi_goal` | The request has more than one eval goal. | feasible |
| `goal_intents` | Goals the UA structured. | all |
| `plan_entries` | Plan entries the ISA returned. | all |
| `plans_run`, `plans_success` | Plans executed, and those that ended `SUCCESS` as our executor sees it. These measure whether execution worked, regardless of whether the result matched the answer key. | all |
| `goals_refused` | Plan entries answered "impossible" or with a question, plus goals stopped for a clarification at structuring. On an infeasible request, refusing the impossible part is the correct behaviour. | all |

The two views can disagree. For example, both plans can run to `SUCCESS` (`plans_success` 2) while
one goal set a wrong value (`goals_verified` 1).

**SimuHome result** (see [Scoring](#scoring-evaluatepy)):

| Column | Meaning | Applies to |
|---|---|---|
| `score` | 1 pass, 0 fail, −1 infrastructure failure. A feasible request passes only if every check passes. An infeasible one passes only if both the execution check and the judge pass. | all |
| `check_rate` | Share of the checks that passed. | feasible |
| `execution_check` | 1 when nothing ran on the impossible part and that part was answered as impossible or with a question. | infeasible |
| `judge` | The judge panel's verdict on the reply. The judge's own failures (−1) are left empty. | infeasible |
| `outcome`, `error_step` | How the episode ended (`executed`, `answered`, `impossible`, `clarify`, `failed`, `error`), and for `error`, the step that raised. | all |

**Groups and averages.** The rows of `summary.*`, in this order:
1. type × case, for example `qt3 / feasible` (12 rows);
2. type, both cases together, for example `qt3 / all`;
3. case, all types together: `all / feasible`, `all / infeasible`;
4. overall: `all / all`.

| Column (`summary.csv`) | `summary.md` | Meaning |
|---|---|---|
| `n`, `n_scored`, `n_infra`, `n_error`, `n_unscored` | n, scored, infra, errors | Episodes in the group: all of them, scored 1 or 0, infrastructure failures, raised in our code, not scored yet. |
| `success_rate` | success | Share of scored episodes that passed. |
| `check_rate` | checks | Mean check pass rate. |
| `goals_verified_rate` | goals ok | Pooled over the group: verified eval goals summed ÷ eval goals summed. |
| `goals_verified_mean`, `goals_per_request` | ok/request, goals/request | Mean verified goals, and mean eval goals, per request. |
| `multi_n`, `multi_goals_verified_rate`, `multi_goals_verified_mean` | multi n, multi goals ok, multi ok/request | The same, over the multi-goal requests only. |
| `plans_success_rate`, `plans_success_mean` | plans ok | Plans that ended `SUCCESS` ÷ plans run (pooled), and their mean per request. |
| `goal_intents_mean`, `goals_refused_mean` | refused/request | Mean goals structured, and refused, per request. |
| `execution_check_rate`, `judge_rate` | exec check, judge | Infeasible episodes in the group: share passing the execution check, and the judge. |
| `<phase>_s`, `<phase>_calls`, `<phase>_tok_in`, `<phase>_tok_out` | *phase* s, *phase* in/out, calls | Mean per episode, for every phase and `total`. `n_costed` is the number of episodes these means cover. |
| `total_s_median`, `total_s_p95` | median, p95 | Median and 95th percentile of `total_s`. The p95 is the slow tail: 95% of the episodes took less. |

`stats.py` also runs on its own, over any records directory, with no LLM calls:

```bash
conda run -n ami-agents python -m tests.simuhome.bench.stats --records tests/simuhome/results/harness/<name>
```

### Harness config

```yaml
name: harness-openai-nano            # records go to <out>/<name>/
providers:
  openai: {api_key_env: OPENAI_API_KEY}   # "openai" and "local" are built in
models:
  segmentation:  {provider: openai, model: gpt-5-nano, reasoning_effort: low}
  structuring:   {provider: openai, model: gpt-5-nano, reasoning_effort: low}
  planning:
    explicit:    {provider: openai, model: gpt-5-nano, reasoning_effort: low}
    general:     {provider: openai, model: gpt-5-nano, reasoning_effort: medium}
  profiling:     {provider: openai, model: gpt-5-nano, reasoning_effort: low}
  guarding:      {provider: openai, model: gpt-5-nano, reasoning_effort: low}
  response:      {provider: openai, model: gpt-5-nano, reasoning_effort: low}
  judge:         {provider: openai, model: gpt-4o-mini, temperature: 0.5}
run:
  episodes: "qt*"        # comma-separated patterns
  limit: null            # per (type, case); null = all 50
  workers: 4
  context: none          # prepend: "(It is <time>. I am in the <room>.)" before the query (test-only)
  memory: "off"          # run: the experience memory is on, per worker
evaluate:
  judge: true
```

`run:` also takes `seeds`, `step_seconds`, `dry_step_seconds`, `real_read_seconds` and
`exec_timeout`. For a fully local run, `variants/harness.local.example.yaml` puts every phase on
`qwen3:8b` through Ollama with one worker, and keeps the judge on OpenAI.

### Models per step

| Step | What it is in the agents |
|---|---|
| `segmentation` | UA `ATOMIC_SEGMENTATION_SYSTEM_PROMPT` |
| `structuring` | UA query and goal structuring |
| `planning.explicit` | ISA small model (explicit LLM fallback, modify recipes) |
| `planning.general` | ISA general BT planner (ambiguous goals) |
| `profiling` | dependency-plan profiling (defaults to `planning.explicit`) |
| `guarding` | condition-guard call (defaults to `planning.explicit`) |
| `response` | the UA's reply, which the judge reads |
| `judge` | SimuHome's judge panel |

How a step names its model:
- `provider: openai` or `provider: local`, or any endpoint defined under `providers:`;
- or a per-step `base_url`.

A step the file does not name uses the bench's default (`models.py`). Model names starting with
`o` or `gpt-5` are treated as reasoning models.

## 3. The individual commands

`harness` chains these; use them directly for finer control.

| Command | What it does | Cost |
|---|---|---|
| `python -m tests.simuhome.bench.run --variant V --episodes … [--limit N] [--workers N]` | Runs episodes with **tick** execution: the trees are ticked against the simulator, in lockstep (10 s virtual steps, `--step-seconds`) when a plan waits on time. One record per episode, in `results/bench/<variant>/`. | ~3–6 calls / episode |
| `python -m tests.simuhome.bench.run_scheduled …` | Same options, with **scheduled** execution: a dry run of the trees, submitted as SimuHome `schedule_workflow` calls (`scheduling.py`). Faster on long timed plans. Writes to `results/bench_scheduled/`. | as `run` |
| `python -m tests.simuhome.bench.evaluate --records DIR [--no-judge] [--skip-scored] [--explain]` | Scores the records, writing `<episode>.eval.json` beside each. | 3 judge calls / infeasible episode |
| `python -m tests.simuhome.bench.stats --records DIR` | The harness statistics for one variant, in `DIR/stats/`. | none |
| `python -m tests.simuhome.bench.report --root DIR [--variants a,b]` | Compares variants: `rates.csv`, `steps.csv`, `outcomes.csv`, `report.md`. | none |
| `python -m tests.simuhome.bench.sweep --grid G [--dry-run]` | Runs a grid of model variants (run + evaluate each), then `report`. Segmentation and structuring results are cached and shared across variants that agree on them. | per variant |

Other `run` flags:
- `--seeds`;
- `--memory off|run`;
- `--context none|prepend`;
- `--skip-existing`;
- `--cache DIR` (shared segmentation / structuring results);
- `--exec-timeout` (real seconds before an immediate plan counts as failed);
- `--out`, `--benchmark`.

A sweep grid varies steps over a base variant:

```yaml
name: planning-models
base: tests/simuhome/bench/variants/default.yaml
vary:
  planning.explicit:
    - {model: gpt-5-nano, reasoning_effort: low}
    - {model: "qwen3:8b", base_url: "http://localhost:11434/v1", temperature: 0}
  planning.general:
    - {model: gpt-5-mini, reasoning_effort: medium}
combine: product            # or one-at-a-time
run: {episodes: "qt2_*,qt3_*,qt4-1_*", limit: 10, workers: 2}
evaluate: {judge: true}
```

### Outputs

```
tests/simuhome/results/
├── bench/<variant>/            run.py: <episode>.json, <episode>.eval.json, run.json, memory_worker<N>/
├── bench_scheduled/<variant>/  run_scheduled.py, same files
├── harness/<name>/             harness.py, same files + stats/
└── sweeps/<grid>/              sweep.py: variants/, cache/, <variant>/, report.md + CSVs
```

An episode record holds:
- the input;
- intents, structures and plan entries;
- the outcome, actions and tool log;
- the final state and snapshots;
- the per-step meter (time, calls, tokens).

`run.json` holds the variant, the git commit, the prompt hashes and the settings.

## 4. Running against your own servers

Use this to watch a home in the inspector while an episode runs, or to debug SHTD. Start the
two servers as in [installation.md § 5c](installation.md#5c-run-one-home-by-hand): the simulator on
14000 with the inspector on 8098, and SHTD on 8097. Then:

```bash
conda run -n ami-agents python -m tests.simuhome.bench.run --episode qt2_feasible_seed_77 --debug \
    --sim http://127.0.0.1:14000/api --shtd http://localhost:8097
```

`--sim` and `--shtd` go together and force one worker. The episode must be the one SHTD was started
with (`--home`).

## 5. Component evaluations

Smaller tools that exercise one step:

| Script | What it measures | Cost |
|---|---|---|
| `python tests/simuhome/eval_atomic_segmentation.py --families qt1,qt2 --case both --out DIR` | Segmentation alone: intent count against the episodes' goals, per family. Supports `--limit`, `--repetitions`, `--concurrency`, local models via `--base-url` + `--model`, and `--reasoning-effort`. | 1 call / episode × repetitions |
| `python tests/simuhome/score_subset.py --baseline A/rows.jsonl --trial B/rows.jsonl` | Compares a segmentation re-run against its baseline: what it fixed, and what it broke. | none |
| `python tests/simuhome/structure_goal.py "turn on the kitchen light" --episode qt3_feasible_seed_1` | One phrase through segmentation and goal structuring over the episode's TDs, with no simulator. `--context-only` prints the context alone. | 1 + 1 per goal; `--context-only` none |
| `CLOCK_SOURCE=http CLOCK_URL=http://localhost:8097/_shtd/time python tests/simuhome/run_dependency_plan.py --episode … --case delay` | One dependency plan, live against the hand-started servers, in virtual time. Cases: `delay`, `at_time`, `after_goal`, `trigger`. | none with `--fake-profile`, else 1 small-model call |

## 6. Maintaining the TD mappings

SHTD types Matter devices with SAREF / SOSA / TD-SOSA using the tables in
[`tests/simuhome/td/mappings/`](../tests/simuhome/td/mappings/), generated from the benchmark and
the vendored Matter model:

```bash
conda run -n ami-agents python tests/simuhome/td/convert.py --phase-a   # regenerate (rows marked approved are kept)
conda run -n ami-agents python tests/simuhome/td/verify.py              # verification suite
```

Both scan the benchmark's `*.yaml` home configs. These are produced by
`tests/simuhome/initial_home_config_to_homeassistant_yaml.py <benchmark dir>`, and the SimuHome
checkout used here already has them.

---

## Adding another benchmark

The agents only ever see an HMAS platform of TDs. The bench drives their code in-process and
adapts the benchmark around it. A new benchmark needs four pieces, which SimuHome illustrates:

| Piece | What it does | SimuHome example |
|---|---|---|
| **Environment as TDs** | Serves the benchmark's home as an HMAS platform the EnvExplorer can crawl: workspaces per room, artifacts per device, HomeOnt / SAREF types, TD-SOSA effects, and a clock route if time matters. Put device facts where real homes get them (Matter / Home Assistant description builders), not in benchmark-specific code. | `ami_agents/environment/integration/SimuHome/shtd.py`, `td_builder.py` |
| **Worker** | Starts an isolated simulator + TD server pair per worker, loads / resets an episode, and clears subscriptions between episodes. | `SimuHome/worker.py` (`SimuHomeWorker`) |
| **Episode pipeline** | Loads an episode (home, query, ground truth), then runs the agents' steps through the stand-ins and executes the plans against the simulator. Benchmark-only options stay test-only, for example `--context prepend`. | `bench/pipeline.py`, `stubs.py`, `execution.py` / `scheduling.py` |
| **Scoring** | Maps the record to the benchmark's own evaluators where they exist; adds deterministic checks where they don't; maps per-request queries to whatever tool calls the evaluators expect. | `bench/evaluate.py`, `toollog.py` |

What carries over:
- **Unchanged:** the meter (`meter.py`) and the per-step model format (`models.py`).
- **With small changes:** the runners and reports (`run.py`, `harness.py`, `stats.py`, `report.py`,
  `sweep.py`). They assume SimuHome's layout: episode files named `<type>_<case>_seed_<n>.json`,
  the type list `qt1` … `qt4-3` in `stats.py`, and the `../SimuHome` location. Generalise those
  points, or keep the same naming, and the rest applies as it is.

HomeBench is the second worked example, and the bench's `--suite` switch is how a benchmark
plugs in: [evaluation-homebench.md](evaluation-homebench.md#adding-another-benchmark).

The design rationale is in
[docs/plans/simuhome_reasoning_harness.md](plans/simuhome_reasoning_harness.md).
