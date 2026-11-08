# Prompt variants for the structuring step: mechanism and experiments

Why: in the M3 run (ECAI-400, DeepSeek-R1-Distill-Qwen-7B structuring) 299 of the 308
failed episodes are lost at structuring, and the biggest single cause is notation, not
choice: 855 of the 963 structured goals write a class without its prefix (`Foyer`,
`DimmableLight`) and the planner accepts none of them (`errors.md`, section 4). This
document says how the way structuring builds its call is now configurable, and what each
experiment on it is, costs and can conclude.

## 1. The mechanism

A step has *axes*; an axis has named *options*, one of them the default (what the step
did before). A selection names one option per axis; a missing axis is its default.

```yaml
# agents.yaml (production), or a harness variant file (one run)
prompt_variants:
  structuring: {context: json, constraint: schema}
```

| file | role |
|---|---|
| `ami_agents/shared/utils/prompt_variants.py` | the registry: `declare_step`, `Axis.option`, `resolve`, `selection_from_config`, `normalize`, `apply_to_config`. Checks names, not signatures. |
| `ami_agents/agents/user_assistant/utils/goal_structuring_variants.py` | the `structuring` step's four axes and their options |
| `ami_agents/agents/user_assistant/utils/goal_context.py` | `GoalContext` (text + the structured view it is rendered from), `Vocabulary` (the names and classes of a context) |
| `ami_agents/agents/user_assistant/behaviours/goal_structuring.py` | `_structure` reads the selection and calls the four options in order |
| `tests/simuhome/bench/models.py`, `run.py` | a variant file's `prompt_variants:` is validated at load, recorded in `variant.as_dict()`, written into the agents' config by the worker, and part of the step cache key |

Properties that matter for experiments:

- All defaults is the old call byte for byte. Pinned by a hash test, and checked end to
  end: the unchanged M3 config, rerun on `hb1_feasible_seed_1085`, gives the same structure
  and the same token counts (3339 in / 758 out) as the stored M3 record. **Arm A is the
  finished M3 run; it is not rerun.**
- A misspelt step, axis or option raises with the valid names. In a variant file it fails
  `--dry-run`; in agents.yaml it stops the UA at startup. It never runs as the default.
- A variant file naming only defaults is the same variant as one naming nothing (same
  cache key); a variant with no `prompt_variants:` has the cache key it always had.

### Axes of `structuring`

| axis | default | option | what it does |
|---|---|---|---|
| `context` | `text` | `json` | the home context as indented JSON (EnvExplorer's structured view, nulls dropped) with a guide to its keys, instead of the indented text and its guide |
| `prompt` | `full` | `focused` | for an explicit/incomplete achievement goal: no ambiguous-goal rules or output shape, device conventions only if the home has a cycle device, two worked answers. Any other goal (dependency, ambiguous) gets the full prompt |
| `constraint` | `none` | `schema` | `response_format` = strict `json_schema`; slots naming something of the home are enums of this home's values. Simple structure only (see 5) |
| `repair` | `none` | `class_names` | after parsing, each class slot is rewritten as the home writes it, only when exactly one class of the home has that name (prefix, case, spacing ignored) |

### Adding a variant, or a step

A variant of an existing axis: one function in `goal_structuring_variants.py`,

```python
@PROMPT.option("terse")
def terse_prompt(structure, rendered, qualifiers) -> str: ...
```

and `terse` is valid in a config. The contract of an axis (what its options receive and
return) is its `doc` in the step's module. Add a test beside `test_prompt_variants.py`.
A new axis: `STEP.axis(name, default, doc)`, register its default option, and have the
step call it. A new step (say the segmenter): a module that calls
`declare_step("segmentation", agent="user_assistant")` and declares its axes, one line in
`STEP_MODULES`, and `selection_from_config(self.agent.config, "segmentation")` at the call
site. `tests/unit/test_prompt_variants.py::test_another_step_is_declared_the_same_way`
shows the whole thing for a second step.

## 2. Baseline: arm A = M3

| quantity | M3 |
|---|---|
| episodes, pass rate | 400, 22.8% (91 of 399 scored) |
| hb1 feasible / hb1 infeasible success | 7% (114 episodes) / 92% (86 episodes) |
| failing episodes lost at structuring | 299 of 308 (97%) |
| goals with a bare class field | 855 of 963; none of them got a plan. Of the 108 fully prefixed, 25 got one |
| failing parts by code | `struct_class_prefix` 38.6%, `struct_slot_confusion` 17.3%, `struct_wrong_value` 13.7%, `struct_wrong_room` 12.5%, `struct_invalid_device_name` 4.3% |
| time | 33.5 s per episode on average (structuring 20.8 s), about 3.7 h for 400; hb1 alone 1.1 h |
| structuring tokens | about 11.8k in and 1.2k out per episode (the reasoning model's thinking is in the output) |

Segmentation is Qwen3-8B in every arm and is deterministic at temperature 0, so the
episodes reach structuring with the same parts in every arm.

## 3. The experiments

One factor at a time against A, same 400 episodes, same models, same run settings: the
arm files differ from `homebench.ecai.M3.yaml` only by `prompt_variants:` (a test pins
that). No paid API call: every model is local.

| arm | file (`tests/homebench/variants/`) | change | question it answers |
|---|---|---|---|
| A | `homebench.ecai.M3.yaml` (done) | none | baseline |
| B | `homebench.ecai.M3.pv-schema.yaml` | `constraint: schema` | can the format be forced out of the model, and how much of the failure goes with it? |
| C | `homebench.ecai.M3.pv-json.yaml` | `context: json` | does the format of the context matter, all else equal? |
| D | `homebench.ecai.M3.pv-focused.yaml` | `prompt: focused` | does a shorter prompt with examples help a 7B model, and at what token cost? |
| E | `homebench.ecai.M3.pv-repair.yaml` | `repair: class_names` | how much of the 22.8% is notation alone? (no LLM change) |

### B: strict schema with the home's enums

What it sends: `response_format={"type": "json_schema", ...strict: true}`. `location_class`,
`artifact_class`, `artifact_name`, `affordance_class`, `affordance_name` and
`parameter_name` (plus `"NA"`) are enums of the values in the goal's context; every slot is
required, nullable where the prompt allows null. An ambiguous-labelled goal gets the
environment-variable shape. It reuses the idea of `condition_check.RESPONSE_FORMAT` (the
precedent in the repo; `profiling` uses only `{"type": "json_object"}`, not a schema).

- Expected to fix: prefix (38.6%), slot confusion such as `location_class: Curtain` or
  `Room` (17.3%), invented device names (4.3%): a value outside the home cannot be written.
- Cannot fix: a valid but wrong value, room or action (13.7%, 12.5%, `affordance` 6.2%), and
  device and room consistency (the enums are flat: `artifact_name` may be a device of
  another room). If B leaves wrong-room and wrong-affordance errors, the next step is a
  schema that pins a device with its room and its own actions (one more option; see the
  second wave).
- Risks:
  1. Grammar-constrained decoding and a reasoning model. On one episode the thinking was
     kept (767 output tokens against 758) and the classes came out prefixed. One episode
     proves nothing; the output token mean over the run is the check.
  2. The 92% of infeasible requests that are refused must not drop: the schema forces a
     home's name wherever a name is written, so a request for a device the home lacks
     could be forced onto a wrong device. Watch `impossible ok` and
     `struct_false_clarify_device`.
  3. Grammar compile time for the larger homes (52 devices). Watch structuring seconds.
- The strict schema is the same call on OpenAI endpoints (the Responses adapter translates
  `response_format`); only Ollama has been exercised.

### C: JSON context

Same facts under EnvExplorer's keys (`title`, `space_class`, `homeont_types`,
`command_types`, `input`, ...), with a guide rewritten for those keys. Prior: little effect
on correctness, more input tokens (3626 against 3339 on the one smoke episode, +8.6%).
It is an ablation that says whether the indented text format is part of the problem, and
it makes the format of the context a configured choice for later steps. Cheapest arm to
drop if B or D already win.

### D: focused prompt

For a goal labelled explicit or incomplete: the explicit/incomplete rules, without the
ambiguous-goal rules (1.3k chars), the ambiguous output shape (0.4k) and the cycle
convention (1.2k, kept when the context has `SetOperationalStateCommand`); plus two worked
answers (1.6k: a percentage on a dimmable light, an incomplete goal on an on/off light),
synthetic and not from ECAI-400. Net size: 10.5k chars against 11.9k (-11%); 3083 tokens in
against 3339 on the smoke episode (-7.7%). Not a dramatic cut: the explicit/incomplete
rules (about 6k) dominate the prompt and stay.
- Expected: small fewer-tokens gain; the examples show prefixed copies. Possible harm: the
  incomplete example biases toward incomplete goals (more clarifying questions on feasible
  requests: `struct_false_clarify_device`), and `device_reference: "description"` in both
  examples.
- A "much shorter prompt" (rules cut by half) is a different arm if D shows that length
  matters; it would be a new option, `prompt: terse`.

### E: class-name repair (control without an LLM change)

Deterministic. `Foyer` becomes `homeont:Foyer`, `Master bedroom` becomes
`homeont:MasterBedroom`, only if the home has exactly one class of that name for that slot.
A class in the wrong slot, a class the home lacks, or a name fitting two classes is left
as written. Each repair is logged and stored in the record (`structures[].repairs`), so
the share of results owed to it is countable. This is not tuning to the benchmark: the
repair uses the home's own context, which the planner already has, and a deployed system
may want it. What it measures is the ceiling of "notation only": the pass rate of E minus
A is the part of the 22.8% that the model had right and wrote differently. If E alone
reaches near B, the cheapest fix is E in production and B is not needed.

## 4. Protocol

1. Screening on hb1 (200 episodes: 114 feasible, 86 infeasible), per arm about 1.1 h, 400
   LLM calls for the steps, no cost. hb1 is the single-command bucket, where one structured
   goal decides the episode, and its infeasible half is the regression guard.

   ```bash
   conda run -n ami-agents python -m tests.simuhome.bench.harness \
       --config tests/homebench/variants/homebench.ecai.M3.pv-schema.yaml --episodes "hb1_*"
   ```

   Same for `pv-repair`, `pv-json`, `pv-focused`. One at a time: one local model serves one
   request at a time (`workers: 1`). Check with `--dry-run` first.
2. The full 400 for the arms worth it: the same command without `--episodes`. Records are
   resumed (`--skip-existing`), so the 200 already done are not run again; about 2.6 h more
   per arm. A full run of all four arms is about 15 h.
3. Per arm, the error report (no LLM call):

   ```bash
   conda run -n ami-agents python -m tests.homebench.error_report \
       tests/homebench/results/harness/homebench.ecai.M3.pv-schema
   ```

4. Compare with A on the **same episodes** (hb1 subset first), paired by episode id:
   - primary: success rate (all-or-nothing) and parts ok; hb1 feasible success;
   - guard: hb1 infeasible success (A: 92%) and `impossible ok`; a drop of more than a few
     episodes cancels an arm whatever its feasible gain;
   - mechanism: counts of `struct_class_prefix`, `struct_slot_confusion`,
     `struct_invalid_device_name`, `struct_wrong_room`, `struct_wrong_value` (parts, from
     `errors_parts.csv`; join on episode and part);
   - cost: structuring seconds and output tokens per episode (`stats/summary.csv`); for E,
     the number of repairs;
   - health: infrastructure failures and timeouts (an arm that times out more has not won).
   With 114 feasible hb1 episodes and a 7% baseline (8 passes), a gain of fewer than about
   10 episodes is within noise: use the paired counts (gained vs lost episodes, exact
   McNemar), not the rates alone. Temperature is 0, so reruns of one arm are
   near-deterministic and a repeat does not measure variance.
5. Decision: an arm moves forward if it gains on the primary, holds the guard, and its
   mechanism counts show the change it was built for. A gain with the wrong mechanism (say D
   improving values but not prefixes) is reported, not claimed.

### Second wave (after the screening, by result)

- Combinations of the winners (`B+D`, `B+C`, `D+E`), each one more arm file; the registry
  allows any, the files are copies with a different `prompt_variants:`. No full factorial:
  `2^4` arms is 60 h for answers the single-factor results make mostly predictable.
- A second structuring model: the same arm files with M1's model lines (Qwen3-8B, 72.7% in
  M1), to see whether the winning variant lifts M3 to M1's level and whether it lifts M1.
  If B or E brings M3 close to M1, most of the model gap was notation.
- The device-pinned schema (6), only if B leaves wrong-room and wrong-affordance errors.
- A production check: the winning variant set in `agents.yaml` (`prompt_variants`, or the
  `AMI_STRUCTURING_*` environment variables) runs the live agents with no code change.

## 5. Limits that are not choices

- `constraint: schema` and `prompt: focused` are written for the `simple` structure (one
  goal): 965 of the 976 goal structurings in M3 (98.9%). The `dependency` structure has goals
  and predicates under open keys (`G1`, `G2`, ...) that a strict schema cannot name; it gets
  no schema, and the full prompt. For a run that is mostly dependency requests, B and D
  would measure almost nothing.
- Options that read the structure raise when the context source sends text only (a stub, an
  EnvExplorer without the `context` field): the goal fails with "needs the structured goal
  context" rather than running as a different variant.
- A schema makes a copied value valid, not right. Value, room and action errors are the
  model's, and an experiment on the format cannot remove them.

## 6. What was checked, and what was not

Checked: 65 new unit tests (`tests/unit/test_prompt_variants.py`) and the existing unit
suite; the control reproduces the stored M3 record; one episode
(`hb1_feasible_seed_1085`, M3 failed it on class prefixes) run through each of the four
arms on the local models. All four write the classes prefixed; B, D and E then execute
the command. C fills `parameter_name` for a direct-value action, which the planner
refuses, so it ends "impossible" for another reason (`struct_parameter_name`). One
episode per arm is an existence check, not a result.

Not checked: any arm over more than one episode; the OpenAI endpoint with a schema; the
grammar compile time on the 52-device home; production (agents running under SPADE) with
a variant set in agents.yaml.

Not built: the device-pinned schema, `prompt: terse`, and a script that compares two run
folders paired by episode (step 4 is done by hand from `errors_episodes.csv` and
`errors_parts.csv` until it exists).
