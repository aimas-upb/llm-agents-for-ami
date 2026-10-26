# SimuHome bench: our agents' reasoning, step by step

Runs SimuHome episodes through the steps the agents run (segmentation, structuring,
planning, execution, response) in one process, without SPADE. Each step is timed and its
LLM calls are counted, and every episode is scored. Design:
[docs/plans/simuhome_reasoning_harness.md](../../../docs/plans/simuhome_reasoning_harness.md).
Full guide (setup, scoring, config formats, outputs, adding a benchmark):
[docs/evaluation-simuhome.md](../../../docs/evaluation-simuhome.md); for HomeBench,
[docs/evaluation-homebench.md](../../../docs/evaluation-homebench.md).

| Command | What it does | Cost |
|---|---|---|
| `python -m tests.simuhome.bench.run --episode qt4-2_feasible_seed_1` | one episode, every step printed, scored at once | ~5 LLM calls (+3 judge calls if infeasible) |
| `python -m tests.simuhome.bench.run --variant V --episodes qt3_*,qt4-1_* --limit 10 --workers 2` | episodes → `<out>/<variant>/<episode>.json` | ~3–6 calls per episode |
| `python -m tests.simuhome.bench.run_scheduled --episode qt4-3_feasible_seed_45` | the same, with execution as SimuHome scheduled workflows from a dry run of the trees (`scheduling.py`) instead of lockstep ticks; records → `results/bench_scheduled/` | as `run` |
| `python -m tests.simuhome.bench.evaluate --records <out>/<variant> [--no-judge]` | scores → `<episode>.eval.json` | 3 judge calls per infeasible episode |
| `python -m tests.simuhome.bench.report --root <out>` | `rates.csv`, `steps.csv`, `outcomes.csv`, `report.md` | none |
| `python -m tests.simuhome.bench.harness --config variants/harness.example.yaml [--dry-run \| --stats-only]` | the evaluation harness: `run_scheduled` + `evaluate` + statistics from one config (a model per phase, OpenAI or `provider: local`); averages per type × feasible/infeasible, per type, per case and overall → `<out>/<name>/stats/summary.md` | as `run_scheduled` + `evaluate`; `--stats-only` none |
| `python -m tests.simuhome.bench.stats --records <out>/<name>` | the harness statistics alone: cost per phase, goals verified / executed, SimuHome score | none |
| `python -m tests.simuhome.bench.sweep --grid variants/sweep.example.yaml [--dry-run]` | run + evaluate + report per variant | per variant as above |

Prefix each command with `conda run -n ami-agents`.

- **Suites** (`suites/`). `run`, `evaluate`, `harness` and `sweep` take `--suite simuhome`
  (default) or `--suite homebench`; a harness or sweep config can say `suite:`. A suite is
  the episodes, the worker, the TD builder, the scoring and the planning conditions of one
  benchmark; SimuHome's behaviour is the default and unchanged. HomeBench episodes are
  `hb<N>_<case>_seed_<n>` (N parts; feasible / infeasible / mixed) from
  `tests/homebench/convert.py`; its configs are `tests/homebench/variants/homebench.{quick,
  spread,full}.yaml`; it has no timed episodes, so it never uses `run_scheduled`.
  `--guidelines homebench|default` (`run.guidelines` in a config) chooses the ISA's planning
  conditions: HomeBench's file lists none (`tests/homebench/guidelines.yaml`), `default` is
  the agents' own, as deployed.
- **HomeBench scoring** (`tests/homebench/scoring.py`, `gold_log.py`, `homebench_metrics.py`).
  Feasible episodes use SimuHome's qt3 check unchanged. Infeasible ones: an execution check
  per part (no accepted action in the part's scope that no feasible part asks for, and the
  agents answered impossible or asked) plus a judge panel -- SimuHome's own for a missing
  device, HomeBench's prompt (`tests/homebench/judge_prompt.py`) for a missing attribute.
  Mixed episodes: both, each on its own parts, all must pass. `score` is the benchmark's own
  reading of each label; `score_extra` also accepts one extra reading (a gold
  `set_brightness(0)` / `set_intensity(0)` for a request that says turn off / switch off is
  met by the device being off: our criterion, not HomeBench's), and every check records which
  reading `matched`. Each verdict carries a `homebench` block (gold and predicted tokens, the
  official bucket, per-part counts) from which `stats.py` computes HomeBench's official EM /
  Precision / Recall / F1 per bucket (`buckets.csv`, a table in `summary.md`) with no LLM
  call; `gold_log.py` turns the accepted actions into the gold's `room.device.op(args)`
  strings. `--no-judge` leaves the execution check alone for the infeasible parts, which is
  weaker, and every table says so.
- **Variants** (`variants/*.yaml`) set one model per step, including local Ollama models
  via `base_url` or `provider: local`. A step a variant doesn't name uses the agents'
  default; profiling and guarding, unnamed, use the variant's `planning.explicit` model.
- **Servers.** Each worker starts its own simulator and SHTD and stops them at the end. Use
  `--sim/--shtd` to run against servers you already have.
- **Timed episodes (qt4)** run in lockstep: the bench advances the simulator clock in
  10-second virtual steps (`--step-seconds`), so a 40-minute plan finishes in seconds.
- **Discovery** is timed separately and kept out of every statistic, since it happens
  offline in a deployment.
- **Scoring.** Feasible qt2, qt3 and qt4 episodes are scored by SimuHome's own evaluators.
  qt1 uses our value check. Infeasible episodes get our execution check (nothing run on
  the impossible part) plus SimuHome's judge on the UA's reply.
