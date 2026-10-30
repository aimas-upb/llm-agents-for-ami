# Evaluating on HomeBench

HomeBench runs through the same evaluation suite as SimuHome
([evaluation-simuhome.md](evaluation-simuhome.md)): the code in
[`tests/simuhome/bench/`](../tests/simuhome/bench/), with `--suite homebench`. Each episode goes
through the steps the agents run:

> segmentation → structuring → planning → execution → response

This happens in one process with no SPADE / XMPP. The agents' own behaviours are called through
the same stand-ins (`stubs.py`), each step is timed, its LLM calls and tokens are counted, and
every episode is scored. Read the SimuHome document first for the parts the two share (the
harness, the per-step model format, the individual commands); this one covers what is
HomeBench's own.

Every command below runs from the repo root, prefixed with `conda run -n ami-agents`. If `conda`
is not on your `PATH` (it is not on the machine this was written on), call it by path:
`C:\Users\Vlad\miniconda3\Scripts\conda.exe run -n ami-agents ...`, or run
`C:\Users\Vlad\miniconda3\envs\ami-agents\python.exe ...` directly. Each command says what it
costs. **Anything that makes LLM calls is for you to run**: the repo's rule is that nobody
starts a benchmark sweep on your behalf (CLAUDE.md § 5.1).

---

## The benchmark

[HomeBench](https://github.com/BITHLP/HomeBench) (ACL 2025 main; cloned next to this repo, see
[installation.md § Step 5-HB](installation.md#step-5-hb--homebench-benchmark)) is a set of
smart-home requests over simulated homes. The test set has 17,366 rows over 100 homes. Every
home has the same 12 rooms (`master_bedroom guest_bedroom living_room ding_room study_room
kitchen bathroom foyer corridor balcony garage store_room`; the dataset spells the dining room
`ding_room`), 34-51 devices of up to 15 types, and 107-165 methods. A row is one request and a
gold string, for example:

| Request | Gold |
|---|---|
| "Increase the volume of the media player by 49% in the garage." | `garage.media_player.set_volume(70)` |
| "Set the brightness of the light in the master bedroom to 90." (that home's light has no brightness) | `error_input` |
| "Set the humidifier in the store room to sleep mode and adjust the aromatherapy in the foyer to decrease the interval by 1 second." (no humidifier in that room) | `error_input,foyer.aromatherapy.set_interval(30),` |

`error_input` means "the request names a device, or an attribute of a device, that is not in
that room". Nothing in HomeBench has a clock or room physics: a command changes the attribute
it names and nothing else.

### Episodes

`tests/homebench/convert.py` turns each row into a SimuHome-shaped episode, so the bench reads
it as it reads SimuHome's. The converted data is generated, not committed
(`tests/homebench/data/`, git-ignored, about 22 MB). Episodes are named
`hb<N>_<case>_seed_<n>`:

| Part | Meaning |
|---|---|
| `hb<N>` | the request has N parts: `hb1` is a single command (12,952 rows), `hb2` … `hb10` are the multi-command requests (4,414 rows) |
| `feasible` | every gold part is a call |
| `infeasible` | every gold part is `error_input` |
| `mixed` | both kinds (`mix` rows of the dataset) |
| `seed_<n>` | 1-based position inside the `(type, case)` group after a **fixed random shuffle** (RNG seed `20261024`, recorded in `manifest.json`) |

There are 27 non-empty `(type, case)` groups (`hb1` 2, `hb2`-`hb8` 3 each, `hb9` 2, `hb10` 2).
The shuffle matters: `--limit 4` takes `seed_1..4`, which is a random sample over the homes and
is the same on every conversion, so results stay comparable across models. `meta.id` and
`meta.homebench_type` keep the dataset's own id and label, and the converter's gold replay
(every gold call applied to its home) runs at conversion time.

### The environment

The agents see HomeBench the way they see SimuHome: as an HMAS platform of Thing Descriptions,
served by two processes that a worker starts on free ports (the same shape as SimuHome's
simulator + SHTD):

| | SimuHome | HomeBench |
|---|---|---|
| state | the external simulator, `../SimuHome` | `HomeBench/simulator.py`: a small engine behind the same `/api` routes and envelope |
| TD server | `SimuHome/shtd.py` (SHTD) | `HomeBench/hbtd.py` (HBTD), same CLI and routes |
| worker | `SimuHome/worker.py` | `HomeBench/worker.py` (`HomeBenchWorker`), same API |

Details, the shape of a device TD and the error codes are in
[`ami_agents/environment/integration/HomeBench/README.md`](../ami_agents/environment/integration/HomeBench/README.md).
Three choices decide what the scores mean:

- **The engine is gold-consistent, not physically correct.** The gold ignores device
  preconditions (a `set_*` on a device that is off occurs 6,472 times, on a closed curtain 851),
  so the engine applies them. The one range correction is heating: 99 of 129 heaters declare a
  temperature range of 30-100 while every gold heating value is 16-30, so the engine uses 16-30.
  With that, applying every gold call of all 17,366 rows raises nothing and leaves no range.
- **One TD action per settable attribute**, typed with a command class, as SHTD does for
  Matter: `onOff` stands for `turn_on` / `turn_off`, `openClose` for `open` / `close`, and so on.
  The gold string of an executed action is a function of `(device, action, value)`.
- **The vacuum robot has no room in HomeBench.** It is `household_vacuum_robot` in a synthetic
  `household` workspace (56 homes, about 1.8% of the rows).

The ontology gained the classes HomeBench needs (`Curtain` and `Blinds` as separate subclasses
of the window covering, `Aromatherapy`, `GarageDoor`, `WaterHeater`, `Trash`) in
`ami_agents/shared/ontologies/homeont.ttl`, additions only. They are not specific to HomeBench:
the Lab308 and SimuHome agents read the same file.

---

## Scoring

An episode is scored from its record alone, so the statistics re-run without LLM calls. The
harness reports **three families of numbers, each labelled for what it is**. None of them
depends on the paper's Table 1.

| Family | What it is | Where |
|---|---|---|
| 1. Gold-token score (primary) | 1 pass, 0 fail, −1 infrastructure error (left out of the rates, as in SimuHome) | `score` in `<episode>.eval.json`, the `success` columns |
| 2. Official HomeBench metrics | EM, Precision, Recall, F1 per bucket, re-implemented from the public `code/eval.py` | `buckets.csv`, `official.csv`, a table in `summary.md` / `report.md` |
| 3. Generic per-part rates | gold tokens met over gold tokens; `error_input` tokens met by a refusal | `parts ok`, `impossible ok` columns. **Our definitions** |

### 1. The score: every gold token met, no judge

One rule for every case (`tests/homebench/scoring.py`), decided from the record, with **no
judge**. Our agents do not argue why a request cannot be done; they report it to the user,
which is what HomeBench's `error_input` stands for.

| Gold token | Met when |
|---|---|
| `room.device.op(arg)` | an accepted action whose gold token is exactly this one (`gold_log.py`), **or**, with no action needed, the part's assert held both before the run and after it (`matched: "held"`; about 9% of the doable parts of a run start at their gold value, and our plans skip a goal that already holds) |
| `error_input` | a goal answered **impossible or with a question**; k gold `error_input`s need at least k such answers (refusals are not tied to parts) |

The episode passes when **every** gold token is met (the same all-or-nothing as SimuHome's
request success). Anything else the agents did is **not counted against them**: we model the
general case (a light is switched on before its brightness is set, a request is split into
more goals than HomeBench has parts), while HomeBench assumes every device is on and one call
per part. A substitute (the blinds closed for a missing curtain) is such an extra action; the
episode still fails, because the `error_input` it should have been is missing. SimuHome's
required `get_room_devices` reads are not part of the rule.

**The extra reading.** As the SimuHome bench does for percent levels, HomeBench has **one**
named extra reading, which is **our criterion, not HomeBench's**: when the gold is
`set_brightness(0)` or `set_intensity(0)` and the clause of the request asks to turn the device
off, that device's `turn_off()` also counts. This covers about 62 single rows. `score_extra` is
the score with that reading; `score` is the strict one; each check records how it was met
(`matched`: `action`, `held` or `turn_off`). Every other disagreement with the gold is a
failure.

### 2. The official metrics, and their caveat

EM (multiset equality per row), micro-averaged Precision / Recall / F1, and the five buckets
(`normal_single` 6,187, `unexist_single` 6,765, `normal_multi` 245, `mix_multi` 4,072,
`error_multi` 97, plus `all`) follow `code/eval.py`. Two of its edge cases are kept on purpose
(`fan only` becomes `fanonly` and misses `fan_only`; tokens compare exactly), and two that only
crash it are changed (an empty bucket reports nothing, no prediction gives precision 0).

Print this caveat next to any number from this family (the tables do):

> The shipped `eval.py` is not runnable as it is (hard-coded paths, and a field `output2`
> that no dataset file has), so the official numbers cannot be reproduced bit for bit. Our
> agents segment a request into atomic intents and answer in prose, so the prediction is built
> from what happened: one gold token per accepted action plus one `error_input` per goal
> answered impossible or with a question. That approximates the official token stream; it is
> not that stream.

### 3. Generic rates

`parts ok` (gold calls met; `error_input` tokens met by a refusal, at most as many as there were
refusals) and `impossible ok` are strict, in the benchmark's reading. If the names of an older paper table
(full success, full + partial, impossible detection) are wanted, derive them from these and
label them as **our definitions**: they are not set beside the paper's Table 1, which came from
an older harness that is not in this repository (see [evaluation.md](evaluation.md)).

### What caps and inflates the numbers

Read these before comparing any rate, with SimuHome's or with another model's.

| Effect | Size | What it does |
|---|---|---|
| **Floor of the primary score** | an agent that does nothing passes 10.7% of feasible and 2.3% of mixed episodes; an agent that refuses everything passes 100% of infeasible ones | the gold ignores preconditions, so some gold calls are no-ops on the initial state. Never compare a feasible or infeasible rate without the other side of the table. The official EM has no such floor |
| **Relative requests in percent** | 5,395 of 17,366 rows ask "by N percent" | the gold adds N *points* to the initial value (2,042 of 2,182 single relative rows equal `current ± N` exactly). Our planner adds points on a property in `unit:PERCENT` and scales otherwise, so the 0-100 levels (brightness, volume, curtain degree, intensity) carry that unit. Flagged per episode (`percent_relative`) and reported apart in the table notes; **not relaxed** |
| **Vacuum operation calls** | 70 gold calls (`start`, `stop`, `pause`, `charge`) | the home offers them, as the vacuum's `state` action (`cleaning`, `stopped`, `paused`, `charging`), because the gold uses them |
| **Heating range** | 883 of 1,197 gold heating values are outside the declared range | corrected in the engine (16-30), so a correct answer is not marked wrong |
| **"Turn off" labelled as level 0** | about 62 single rows | the one extra reading above |
| **Unresolved infeasible parts** | 1,705 parts in 426 multi rows (8.4% of parts) | the request has fewer or merged clauses than the gold has parts, so room and device of an `error_input` cannot be recovered from the text. The score does not need them (refusals are counted, not placed); the count is in `hb_unresolved` |
| **Duplicates and noise** | 49 duplicate `(home, input)` rows; 3,478 multi gold strings hold more than one `error_input` | label noise; part of what a reader should not take for agent error |
| **Segmentation granularity** | any request | a model that merges two parts changes the `error_input` count: the official metrics depend on it, the primary score does not |

### The power-on guard (Q1)

The ISA's `power_on_first` guard is Matter's rule: before a level or mode command on an
`AirConditioner` or `DimmableLight` (a `ColorLight` is one) it inserts a `turn_on`.
HomeBench has no such precondition, and its gold has no `turn_on` there. Run with the guard, the
final state still matches (each goal asserts only the attribute it sets), but the action log
gains a call the gold lacks, which costs precision in the official metrics.

So **by default the HomeBench suite points the ISA at `tests/homebench/guidelines.yaml`, which
lists no conditions** (`run.guidelines: homebench`, or `--guidelines homebench`). The consequence
is stated plainly: the power-on guard and the condition-guard LLM call do nothing on HomeBench,
so the **`guarding` phase is not exercised there**. To run the same episodes "as deployed", with
the agents' own file, use `--guidelines default` (or `run.guidelines: default`) and read the
official metrics with the extra-`turn_on` caveat. The two settings are separate record
directories if the configs are named apart.

---

## Prerequisites

1. **HomeBench next to this repo**, at `../HomeBench`, **and the dataset converted**:
   [installation.md § Step 5-HB](installation.md#step-5-hb--homebench-benchmark). Scoring
   needs nothing of SimuHome's (no judge, no evaluator module).
2. **An LLM endpoint**, as for SimuHome: `OPENAI_API_KEY` for OpenAI models, or Ollama at `http://localhost:11434/v1` through `provider: local`. **The planner needs tool
   calling**, so a local model must support tools. On the machine this was written on, `gemma3:12b`
   does not (HTTP 400), `ministral-3:8b` does, and `qwen3:8b` (the model in
   `harness.local.example.yaml`) is not installed.
3. **No servers to start.** Each worker starts its own engine and HBTD on free ports and stops
   them at the end.

---

## 1. Smoke test: one episode

```bash
# 3 LLM calls on a local model (segmentation, structuring, the reply); nothing is scored
conda run -n ami-agents python -m tests.simuhome.bench.run --suite homebench \
    --episode hb1_feasible_seed_1 --variant tests/homebench/variants/homebench.quick.yaml \
    --no-evaluate --debug

# score it (no LLM call: HomeBench has no judge), then the statistics
conda run -n ami-agents python -m tests.simuhome.bench.evaluate --suite homebench \
    --records tests/homebench/results/bench/homebench.quick --explain
conda run -n ami-agents python -m tests.simuhome.bench.stats \
    --records tests/homebench/results/bench/homebench.quick
```

Drop `--no-evaluate` to score the episode at once (no LLM call either way).
`--stop-after segmentation|structuring|planning` stops early, with no actuation.

Measured on a local `ministral-3:8b`, one worker: this episode took 12.8 s wall (segmentation
6.1 s, structuring 3.7 s, planning 0.3 s, execution 0.1 s, the reply 1.2 s), 3 LLM calls, and
was scored 0: the request says "49%", the planner set 31 and the gold says 70 (the percent
convention above). That is a correct run of the harness and a wrong answer by the benchmark's
reading, not an infrastructure failure.

## 2. The evaluation harness

The harness is SimuHome's, with `suite: homebench` in the config. HomeBench has no timed
episodes, so it runs the **tick** execution (`run.py`), never `run_scheduled`. One config file
chooses the model for each phase and the episodes to run; the harness runs them, scores them
(`evaluate --skip-scored`) and computes the statistics. A stopped run resumes without paying
twice (`--skip-existing`, `--skip-scored`).

### Three configs, one mechanism

The three configs live in `tests/homebench/variants/`. They differ **only** in `run:`; the model
blocks are identical (the local `ministral-3:8b` on every step; no judge).

| Config | Selection | Episodes | LLM calls | Wall clock, one local 8B worker |
|---|---|---|---|---|
| `homebench.quick.yaml` | `hb1`-`hb4`, `limit: 4` per (type, case): 11 groups | 44 | ~160 | **~24 min estimated** |
| `homebench.spread.yaml` | all types, `limit: 1` per (type, case): 27 groups | 27 | ~172 | **~25 min estimated** |
| `homebench.full.yaml` | all, `limit: null` | 17,366 | ~56,557 | **~125 h estimated** |

The episode and call counts are exact, from `--dry-run`. **The times are estimates, not
measurements.** They assume about 8 s per LLM call plus 3-5 s of per-episode overhead. The
calls above count the steps only (one segmentation + one structuring per part); the one episode
measured also made a call for the reply. It took 12.8 s for 3 calls, which is
faster than that rule, but a multi-part request makes more and longer calls and none was
measured. **Calibrate with the first `--limit 1` run on your machine** and lower `limit` if a run
would pass 30 minutes.

Two limits of the subsets, to state with every table made from them: 44 episodes cannot resolve
a 5-point difference between models (a group of 4 steps in 25%), and `limit` is per group, so a
subset over-represents the rare groups (the `hb10` feasible group has 2 rows in the data). The
quick config is for comparing configurations and catching regressions; the spread config checks
that nothing breaks at large N. Going from quick to full is `--limit` removed (or the `full`
config), nothing else.

```bash
# 0 cost: what would run, with the call estimate
conda run -n ami-agents python -m tests.simuhome.bench.harness --suite homebench \
    --config tests/homebench/variants/homebench.quick.yaml --dry-run

# ~160 LLM calls on a local model, ~24 min estimated -- you run this
conda run -n ami-agents python -m tests.simuhome.bench.harness --suite homebench \
    --config tests/homebench/variants/homebench.quick.yaml --workers 1

# ~172 LLM calls, ~25 min estimated, to check large N -- you run this
conda run -n ami-agents python -m tests.simuhome.bench.harness --suite homebench \
    --config tests/homebench/variants/homebench.spread.yaml --workers 1

# ~56.5k LLM calls, about 125 h per local model -- you decide and run this, in installments
# if need be (records are resumable)
conda run -n ami-agents python -m tests.simuhome.bench.harness --suite homebench \
    --config tests/homebench/variants/homebench.full.yaml --workers 1

# recompute the statistics from the records on disk (no LLM calls; "No records in ..." until a run has written some)
conda run -n ami-agents python -m tests.simuhome.bench.harness --suite homebench \
    --config tests/homebench/variants/homebench.quick.yaml --stats-only
```

`--suite homebench` is optional when the config says `suite: homebench`; the commands above keep
it so they read the same as the single-episode ones. Overrides: `--episodes hb1_*,hb3_mixed`,
`--seeds 1-10,15`, `--limit N` (per type and case), `--workers N`, `--out DIR`
(default `tests/homebench/results/harness`), `--benchmark DIR` (default `tests/homebench/data`).
Use one worker with one local model: more workers only queue on it. A home's episodes are kept
on one worker, so its engine is reset to the same home in a row.

### Harness config

The same format as SimuHome's ([evaluation-simuhome.md](evaluation-simuhome.md#harness-config)),
with two differences. `suite: homebench` selects the benchmark, and `run.guidelines` chooses the
ISA's planning conditions:

```yaml
suite: homebench
providers:
  local: {base_url: "http://localhost:11434/v1"}     # Ollama; the planner needs tool calling
models:
  segmentation:  {provider: local, model: "ministral-3:8b", temperature: 0}
  # ... structuring, planning.explicit, planning.general, profiling, guarding, response ...
run:
  episodes: "hb1_*,hb2_*,hb3_*,hb4_*"
  limit: 4                 # per (type, case); null = every episode
  workers: 1
  context: none
  memory: "off"
  guidelines: homebench    # homebench: no device conditions | default: as deployed
```

There is no `evaluate:` section and no `judge` model: HomeBench is scored without a judge, so
nothing would read them.

The configs have no `name:`, so the records directory is the file's name
(`<out>/homebench.quick/`) and the three never share records. Variant files are read with
`yaml.safe_load`, so `${VAR:-default}` does not apply to them: model names are plain, as in the
SimuHome examples.

## 3. The individual commands

`harness` chains these. They are SimuHome's commands with `--suite homebench`; the table lists
what differs.

| Command | What it does | Cost |
|---|---|---|
| `python -m tests.simuhome.bench.run --suite homebench --variant V --episodes … [--limit N]` | Runs episodes (tick execution), one record per episode in `tests/homebench/results/bench/<variant>/`. `--execution scheduled` is refused for HomeBench. `--guidelines homebench\|default` | 1 + N step calls per episode (N parts), and a call for the reply |
| `python -m tests.simuhome.bench.evaluate --suite homebench --records DIR [--skip-scored] [--explain]` | Scores the records, writing `<episode>.eval.json` beside each. `--suite` defaults to each record's own | none |
| `python -m tests.simuhome.bench.stats --records DIR` | The statistics for one variant in `DIR/stats/`, with the official table | none |
| `python -m tests.simuhome.bench.report --root DIR [--variants a,b]` | Compares variants: `rates.csv`, `steps.csv`, `outcomes.csv`, `official.csv`, `report.md` | none |
| `python -m tests.simuhome.bench.sweep --grid G [--dry-run]` | A grid of model variants (run + evaluate each), then `report`. The grid says `suite: homebench` | per variant |

A HomeBench sweep grid is SimuHome's with `suite: homebench` and a HomeBench `base:`:

```yaml
name: hb-planning-models
suite: homebench
base: tests/homebench/variants/homebench.quick.yaml
vary:
  planning.explicit:
    - {model: "ministral-3:8b", base_url: "http://localhost:11434/v1", temperature: 0}
    - {model: gpt-4o-mini, temperature: 0}
combine: one-at-a-time
run: {episodes: "hb1_*,hb2_*", limit: 4, workers: 1}
```

The sweep runs the `base` variant as well as one variant per value: the grid above is 3
variants (`--dry-run` lists them; it makes no call). `run_scheduled.py` is not offered for
HomeBench.

### Outputs

```
tests/homebench/results/
├── bench/<variant>/     run.py: <episode>.json, <episode>.eval.json, run.json, stats/
├── harness/<name>/      harness.py, same files + stats/
└── sweeps/<grid>/       sweep.py: variants/, cache/, <variant>/, report.md + CSVs
```

An episode record has the same fields as SimuHome's (intents, structures, plan entries,
outcome, actions, tool log, final state, the per-step meter) plus `suite`. The verdict
(`<episode>.eval.json`) holds `score`, `score_extra`, `error_type`, the checks (one per gold
call, each with its `expected` token, its `class` and, when ok, how it was `matched`), the
refusal check (`execution_check`: `error_inputs` against `refusals`) for an episode with an
`error_input`, and a **`homebench` block**: `gold_tokens`, `predicted_tokens`, `em`, `bucket`,
per-part counts, `unresolved`, `percent_relative`. The block is absent for infrastructure failures and stopped runs.

`stats/` holds `summary.md`, `summary.csv`, `summary.json`, `episodes.csv` and, for HomeBench,
`buckets.csv`; `summary.md` has the per-type table, the official table per bucket, and the notes
(the official-metrics caveat, and the percent requests' success next to the others'). The
notes are printed only when they apply to the records.

## 4. Running against your own servers

To look at one home by hand, or debug HBTD, start the two servers on fixed ports:

```bash
conda run -n ami-agents python ami_agents/environment/integration/HomeBench/run_engine.py \
    --home 42 --sim-port 8099 --hbtd-port 8097
```

then point the console runner at it
(`YGGDRASIL_URL=http://127.0.0.1:8097 conda run -n ami-agents python tests/manual_interactive_cli.py`),
or run the bench against it, as for SimuHome:

```bash
conda run -n ami-agents python -m tests.simuhome.bench.run --suite homebench \
    --episode hb1_feasible_seed_1 --variant tests/homebench/variants/homebench.quick.yaml --debug \
    --sim http://127.0.0.1:8099/api --shtd http://localhost:8097
```

`--sim` and `--shtd` go together and force one worker. The episode must be one of the home the
servers were started with (`home42` above: `hb1_feasible_seed_1` is a `home42` row).
`run_engine.py --engine-only` starts the engine without HBTD.

## 5. Maintaining the data and the mapping tables

```bash
# regenerate the episodes, the homes and the mapping tables from ../HomeBench (0 cost, ~35 s)
conda run -n ami-agents python tests/homebench/convert.py --homebench ../HomeBench --out tests/homebench/data

# the mapping tables only (a few seconds)
conda run -n ami-agents python tests/homebench/convert.py --homebench ../HomeBench --mappings-only

# the data-layer, environment, bench and scoring tests (no LLM)
conda run -n ami-agents pytest tests/unit -q -k homebench

# crawl one home with the EnvExplorer's code path, over live engine + HBTD processes (~1.5 min, no LLM)
conda run -n ami-agents python ami_agents/environment/integration/HomeBench/verify_phase2.py
```

The tables are `tests/homebench/mappings/*.yaml` (same schema and loader style as SimuHome's).
`observable_properties`, `ambient_sensors` and `actuation_effects` are empty on purpose: HomeBench
has no physics, and no action carries a TD-SOSA effect, because that would feed the planner a
causal claim the benchmark never makes.

---

## Adding another benchmark

[evaluation-simuhome.md](evaluation-simuhome.md#adding-another-benchmark) lists the four pieces a
benchmark needs. There are now two examples, and the bench has a seam for the choice: a
**suite** (`tests/simuhome/bench/suites/`), one small object holding the episode names, the
episode loader, the worker class, the TD builder class, the evaluators, the cost estimate, the
planning-conditions file and the results directory.

| Piece | SimuHome | HomeBench |
|---|---|---|
| **Environment as TDs** | external simulator + `SimuHome/shtd.py`, `td_builder.py` (Matter devices, room physics, TD-SOSA effects) | `HomeBench/simulator.py` (a dictionary and a few validators) + `hbtd.py`; `HomeBenchTD(SimuHomeTD)` reuses every document builder and overrides the device TD and the attribute handling. No physics |
| **Worker** | `SimuHome/worker.py` | `HomeBench/worker.py`: the same API; one home repeats over many episodes, so `load` resets the engine to the episode's home and restarts HBTD each time |
| **Episode pipeline** | `bench/pipeline.py`, `stubs.py`, `execution.py` / `scheduling.py` | the same, through the suite: `Home(builder_cls=suite.td_cls)`, `timed` false, `scheduling` imported only when used |
| **Scoring** | SimuHome's evaluators + judge, `toollog.py` | `tests/homebench/scoring.py` (the gold-token rule, no judge), `gold_log.py` (the reverse of `toollog.py`), `homebench_metrics.py` (the official metrics) |

To add a third benchmark:
1. Write the data tooling that emits **SimuHome-shaped episodes** (`meta`, `query`,
   `eval.goals`, `initial_home_config`) named `<type>_<case>_seed_<n>`. If the files are not a
   plain JSON read, give the suite its own `load_episode` (HomeBench's is `convert.load_episode`).
2. Serve the home as TDs: subclass `SimuHomeTD` for the documents, write the state engine and
   the worker with `SimuHomeWorker`'s API.
3. Add `tests/simuhome/bench/suites/<name>.py` and register it in `suites/__init__.py`
   (`SUITE_NAMES`); the classes are named by import path and imported on first use. Reuse
   an existing evaluator wherever a rule is the same, and say in the docs which readings are
   ours.
4. Add the unit tests: the data layer against the real dataset (skipped when absent, as
   `tests/unit/test_homebench_data.py` does), the environment, and the scoring on hand-built
   records, including a gold-as-the-agent run that must score 1 everywhere.

---

## Decisions this document assumes (Q1-Q5)

The plan left five points open, each with a default, and the defaults are what was built and
what the commands above use.

| | Question | Default used |
|---|---|---|
| Q1 | power-on guard on HomeBench | a guidelines file with no conditions (`run.guidelines: homebench`); `default` runs "as deployed" |
| Q2 | the shared ontology | `homeont.ttl` edited, additions only; two existing tests changed (a pinned class count, and "blinds" now naming two classes) |
| Q3 | branch and commits | branch `feature/homebench-harness`, **nothing committed** |
| Q4 | the calibration model | `ministral-3:8b` (local); `qwen3:8b` not installed, `gemma3` cannot plan |
| Q5 | the judge in subset runs | superseded: HomeBench is scored without a judge (a refusal is the `error_input` token) |

## Not verified

- **Any subset run** (quick, spread, full) and therefore every wall-clock figure above.
- **The bench against hand-started servers** (§ 4: `run --sim ... --shtd ...`). `run_engine.py`
  was started and answered (`/_shtd/time`, the engine's health route, the platform document),
  but the bench command itself needs an LLM call and was not run.
- **The planner on every new class** (`Aromatherapy`, `GarageDoor`, `WaterHeater`, `Trash`,
  `Curtain` against `Blinds`) beyond the one media-player episode measured here.
- **The HomeBench licence and citation:** the repository has neither, and they were not looked
  up. Settle both before publishing numbers.
