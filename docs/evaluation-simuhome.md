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
