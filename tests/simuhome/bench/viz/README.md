# Results report and trajectory viewer

`build_report.py` turns one or more harness result roots into **one self-contained HTML
file**: inline CSS, JS and SVG, no CDN, so it opens offline and can be mailed. Standard
library only, no LLM calls. The statistics are computed by `tests/simuhome/bench/stats.py`,
so every number agrees with each run's `stats/summary.md`.

```bash
# one attempt
python -m tests.simuhome.bench.viz.build_report --roots ~/experiments/semantic/harness/1 \
    --out report.html --title "SimuHome, local models on fep"

# three attempts, pooled, with a reference config and the benchmark's answer keys
python -m tests.simuhome.bench.viz.build_report \
    --roots ~/experiments/semantic/harness/1 ~/experiments/semantic/harness/2 ~/experiments/semantic/harness/3 \
    --pooled ~/experiments/semantic/harness/pooled \
    --reference ~/experiments/reference/harness-gpt6-luna="gpt-6 luna (reference)" \
    --benchmark ../SimuHome/data/benchmark \
    --out report.html
```

| Option | Meaning |
|---|---|
| `--roots R1 [R2 ...]` | result roots, one per attempt; each holds `<config>/` directories of records (`<episode>.json`, `<episode>.eval.json`, `run.json`, `gpu.txt`, `stats/`). A root that is itself a records directory counts as one config. |
| `--pooled DIR` | a `pool_attempts.py` directory; its `<episode>.a<N>.json` links are read back as attempt `N` (attempts a root already gave are not duplicated). |
| `--reference DIR[=label]` | a records directory (or a root of them) shown beside the local configs, in gray, as the comparison row. Repeatable. |
| `--benchmark DIR` | SimuHome's `data/benchmark`; when given, each episode's answer key (`eval.goals`) is shown in the viewer. Without it the evaluation's own checks serve as the ground truth. |
| `--no-records` | leave the episode records out: a summary-only page a few hundred KB in size. |
| `--out`, `--title` | the file to write and the page title. |

Requirements: Python 3.9+. `stats.py` imports the bench's `meter.py`, which imports the
`openai` package only to wrap its client classes; when the package is missing the loader
stands in a stub, so the report builds in a bare interpreter too. Run from the repo root
(`python -m ...`) or by path (`python tests/simuhome/bench/viz/build_report.py ...`).

## What the page shows

**Overview** (one tab): a card per config (GPU, node, partition and job from `gpu.txt`;
workers, commit, start time and settings from `run.json`; attempts found; the model per
phase), then: success per request type × feasible / infeasible (bars per config); the
overall rates (success, checks, goals ok, plans ok, exec check, judge) and a table with
refused/request per case; latency per episode (median / mean / p95 dot plot, with the GPU in
the row label) and mean seconds per phase (stacked); tokens in / out per phase and LLM
calls per episode; the failure categories across configs (click a cell to open those
episodes); and, with several attempts, the per-attempt success for qt2 / qt4-x with the
range, standard deviation and pooled rate.

**Config pages** (one tab each): the config header and full model table; the three
`summary.md` tables (score and goals, seconds per phase, tokens per phase) over the pooled
attempts or any single one (a selector appears when there are several), the HomeBench
official bucket table when the records are HomeBench's; the **failure catalogue**: counts
per category × request type (click a cell to filter), per case, the detection rules, and
the per-case list in the style of `analysis.txt` (episode, category, one-line explanation),
each line opening the trajectory.

**Episodes** (one tab): every episode row, filterable by config, type, case, score,
category, outcome and free text; sortable columns. A row opens the **trajectory panel**:
query and ground truth (the benchmark's `eval.goals` when `--benchmark` is given, always
the evaluation's expected values with each check's verdict and the infeasible part);
intents; structures (goals, questions, errors); plan entries with their behaviour trees;
execution (plan statuses, the action timeline, scheduled workflows, tracked values, room
readings per snapshot); the tool log; the reply; the meter (seconds, LLM seconds, calls,
tokens per phase, and every LLM call); the evaluation verdict (execution check, judge
votes, required tool calls, SimuHome targets); warnings and anomalies gathered from the
record; the reasoning trace; and the compacted record as JSON. Every chart has a Table
button showing the same numbers; the page follows the OS theme, with a toggle.

Deep links work: `#config/<name>`, `#episodes?config=<name>&category=<category>`,
`#episode/<config>|<attempt>|<episode>`.

## Failure categories (`failures.py`)

One category per failed or errored episode, derived from the record and its `.eval.json`
without any model call; the first rule that fires wins, in this order:

| Category | Detected by |
|---|---|
| infrastructure error | score −1 (Infrastructure, Not Run, No Simulator, Judge Error, Evaluation Runtime Error), or the record's error holds an infrastructure marker |
| pipeline error | `outcome == "error"` (our code raised, in `error_step`) |
| nothing understood | no intents, structures or plans (`outcome == "nothing"`) |
| empty structuring | a structure with an error (no / unparsable JSON) or without goals, and nothing reached the planner |
| executed infeasible part | infeasible: execution check `Executed Infeasible Part` (actions on the impossible part) |
| not reported infeasible | infeasible: execution check `Not Reported Infeasible` |
| judge rejected reply | infeasible: execution check passed, judge panel 0 (`Context Faithfulness Failure`) |
| clarification asked | feasible: a structure's `clarification` / `question`, or a plan entry with outcome `clarify` |
| impossible verdict | feasible: a plan entry with outcome `impossible` |
| invalid plan tree | a plan entry with error `ambiguous_planning_failed` (the generated tree failed validation) |
| no plan path | a plan entry with another error: `unplannable_goal`, `unresolvable_predicate`, `dependency_compile_failed`, `unrouted` |
| execution timeout | a run plan with `timeout after Ns` or still `RUNNING` |
| action rejected | a run plan with an HTTP error, or an action with `success: false` |
| plan execution failed | a run plan ended `FAILURE` with no rejected action or timeout |
| wrong timing | a check classed `wrong_timing` |
| wrong value | a check classed `wrong_value` |
| wrong device | checks classed `missing` while actions ran on devices the answer key does not name |
| missing action | checks classed `missing` with no action attempted on them |
| required tools not invoked | `Required Actions Failure` |
| value mismatch | qt1 `Value Mismatch` (a wrong reading reported) |
| other | nothing above matched |

## Size

Records are embedded one per `<script type="application/json">` block and parsed only
when opened. Snapshots are reduced to the room readings per tick, the final home state to
the room readings plus the checked devices, and the variant (identical per config) is left
out; the viewer's "Warnings and anomalies" section says how much was removed. 209 episodes
over three configs build a 5.6 MB page in under two seconds; the full records were 38 MB.

## Files

- `build_report.py` — the CLI, the per-config aggregates, the page assembly.
- `loader.py` — discovery of roots / pooled / reference directories, `gpu.txt`, `run.json`, the model table, record compaction.
- `failures.py` — the failure categories and the one-line explanations.
- `static/report.css`, `static/report.js` — the page, inlined at build time.
