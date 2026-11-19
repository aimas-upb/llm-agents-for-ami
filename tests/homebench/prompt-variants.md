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
| `context` | `text` | `json` | version 1 of the JSON prompt: the home context as indented JSON (EnvExplorer's structured view, nulls dropped) with a guide to its keys, instead of the indented text and its guide. The RULES are still the text format's words (`Artifact`, `name`, `command class`). Kept as it ran (arm C) |
| `context` | `text` | `json_v2` | version 2 of the JSON prompt: the same JSON, a guide that explains every key, and the rules reworded in the JSON's keys (`title`, `command_types`, `acts_upon`, ...) |
| `prompt` | `full` | `focused` | for an explicit/incomplete achievement goal: no ambiguous-goal rules or output shape, device conventions only if the home has a cycle device, two worked answers. Any other goal (dependency, ambiguous) gets the full prompt |
| `constraint` | `none` | `schema` | `response_format` = strict `json_schema`; slots naming something of the home are enums of this home's values. Simple structure only (see 5) |
| `decoding` | `free` | `iri_trie` | GENRE-style constrained decoding (De Cao et al. 2021): each IRI value of the reply is decoded under a trie of the values the home allows in that slot, narrowed by the slots the same object already holds (a device of the written room, an action of the written device); the rest of the reply, thinking included, is free. Needs an endpoint that continues a prefilled assistant message and returns `top_logprobs` (Ollama). Not with `constraint: schema`. See G |
| `decoding` | `free` | `genre` | GENRE itself: the structuring model runs in-process (Hugging Face, `tests/simuhome/bench/genre_server.py`), the logits of the whole vocabulary are masked at every step to the tokens that keep a value inside its trie, and the answer is decoded by beam search. Needs that server as the structuring provider. Not with `constraint: schema`. See H |
| `decoding` | `free` | `iri_trie_v2` | `iri_trie` plus abstention: an `artifact_class` that is a device family of the ontology the room does not have is not replaced by the room's nearest device; its goal becomes one about the missing device (`unmatched_device`, the IRI slots null), which the pipeline answers with "I can't find the X" (`utils/abstention.py`). Arm G2 |
| `decoding` | `free` | `genre_v2` | `genre` plus four rules in `genre_server.py`: no `null` for a room, device or action slot the home has values for; a JSON key stays one the free answer wrote; the same abstention, read from the free answer of the same thinking; a thinking cut at the token cap is closed and answered from, not an error. Arm H2 |
| `repair` | `none` | `class_names` | after parsing, each class slot is rewritten as the home writes it, only when exactly one class of the home has that name (prefix, case, spacing ignored) |

### Versions of the prompt

The option name of the `context` axis is the version of the prompt. The structuring prompt
is the rules, the guide to the context's format and the context; the rules name what they
point at, and were written for the text rendering (`Artifact`, `name`, `command class`,
`acts upon`). Sent a JSON context, a small model must translate those words into the keys it
sees (`artifacts`, `title`, `command_types`, `acts_upon`): that is version 1 (`json`). Version
2 (`json_v2`) has no translation left: its rules say `title`, `command_types`, `acts_upon`.

| `context` | context | guide | rules in |
|---|---|---|---|
| `text` (default) | indented text | `CONTEXT_GUIDE` | the text's words |
| `json` (v1) | JSON | `CONTEXT_GUIDE_JSON` | the text's words |
| `json_v2` | JSON | `CONTEXT_GUIDE_JSON_V2` | the JSON's keys |

How it is built, so that the versions cannot drift: the JSON rules are not a second copy. They
are the text rules with a table of phrases swapped (`JSON_REWORDINGS` in
`goal_structuring_prompts.py`); a phrase the rules no longer hold once makes the import fail, so
an edit of a rule is made in one place and the table is the only thing to follow. Tests pin
that the default prompts and the v1 guide are unchanged (by hash), that the v2 rules name
none of the text format's words, and that every key they name is explained by the v2 guide
and exists in EnvExplorer's structured view. The version reaches `focused` as well
(`context: json_v2, prompt: focused` has the focused prompt with the JSON rules).

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
| F | `homebench.ecai.M3.pv-json-v2.yaml` | `context: json_v2` | is C's result the format, or the rules left in the text's words? (F against C: the wording; F against A: both) |
| G | `homebench.ecai.M3.pv-trie.yaml` | `decoding: iri_trie` | if the failure is malformed or invented IRIs, does decoding them under a trie of the home's valid IRIs fix it, and what is left? |
| H | `homebench.ecai.M3.pv-genre.yaml` | `decoding: genre` | G emulates GENRE over an API that shows 20 tokens per step; with the mask on the whole vocabulary and beam search, how much of G's result was the emulation (top 20, `nearest`, greedy) and how much the constraint? |
| G2 | `homebench.ecai.M3.pv-trie-v2.yaml` | `decoding: iri_trie_v2` | G lost the infeasible requests by putting the room's nearest device in place of the missing one; if the device the free reply named is reported missing instead, how much of that comes back, and what does the evidence rule (first run on the 400) do to the feasible ones? |
| H2 | `homebench.ecai.M3.pv-genre-v2.yaml` | `decoding: genre_v2` | H lost goals to `null` exits and unreadable keys; with those closed and the same abstention, does the mask on the whole vocabulary beat G2? |

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

### F: JSON context, version 2 of the prompt

C sends the JSON under rules that say `Artifact`, `name`, `command class` and `acts upon`,
none of which the JSON has; a 7B model is left to translate, and a worse C than A could be
that, not the format. F sends the same JSON and facts with the guide completed (`written_by`,
`quantity_kind`, `unit`, `type`, `description`, `parameter_schema`, the `observed_by`
entries) and every rule reworded in the keys. Prior: F at least as good as C; F against A
says whether JSON is a better format once nothing is left to translate. Size: 15,407 chars sent against 14,920 for C on `hb1_feasible_seed_1085`
(+0.5k: the larger guide and the longer phrases); 3770 tokens in against 3626.

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

### G: constrained decoding with a trie of valid IRIs (GENRE)

The question from the 2026-11-01 discussion: if the structuring errors are wrong URLs
(malformed, missing, or names the home does not have), constrain the decoding itself, as
GENRE (facebookresearch/GENRE, "Autoregressive Entity Retrieval") does for entity names:
at each step only the tokens that keep the written prefix inside a prefix tree of the
valid names may be chosen. Code: `ami_agents/agents/user_assistant/utils/iri_trie.py`.

- The tries. One per slot that names something of the home (`location_class`,
  `artifact_class`, `artifact_name`, `affordance_class`, `affordance_name`,
  `parameter_name`, the `property_*` and `space_*` slots). The values are narrowed by
  what the same JSON object already holds, from the paths of the home (room -> device
  -> action -> parameter): after `homeont:LaundryRoom`, only that room's devices; after
  the device, only its actions. If no path fits, the slot's whole vocabulary (logged
  `conditioned: false`). This is what B's flat enums cannot do.
- On Ollama. The API has no logits processor, and a grammar cannot be applied to a
  continued message (`Failed to initialize samplers`). So: the reply is decoded freely
  first (the control call); each IRI value in it that is in its trie is kept, because
  constrained greedy decoding writes the same value (every prefix of it is allowed, so
  the greedy token was allowed at every step). At the first value outside its trie, the
  reply is cut at the opening quote and the value decoded again one token per call (the
  reply so far prefilled as the assistant message, `top_logprobs=20`, the most probable
  allowed token taken; characters the trie forces are written without a call); the value
  is closed with the model's own closing token (`",`, not a bare `"`: with a bare quote
  the model skips the comma it believes written); then the model continues freely, and
  the check goes on. If none of the 20 tokens is allowed, the model's free continuation
  of the value is snapped to the nearest value of the trie (`how: nearest`).
- Evidence from the later slots (since 2026-11-15). Valid or not is decided on the slots
  written before a value, as above; but the trie a value is re-decoded with is also
  narrowed by the slots the model wrote after it in the same object, when they fit the
  home. Without it, a guessed class made the correct action after it invalid:
  `hb1_feasible_seed_4351` wrote `affordance_class: homeont:TemperatureSetpoint`, then
  `affordance_name: temperature`; greedy took the `homeont:` branch, wrote
  `homeont:SetModeCommand`, and `temperature` became `mode`. With the evidence the class
  is `saref:SetAbsoluteLevelCommand` (the only class of `temperature`) and the action is
  kept. The G v1/v2 results (20 episodes, 2026-11-14) were made before this change.
- Recorded: `structures[].decoding` = calls, and per re-decoded value the slot, what the
  free decoding wrote, what the trie wrote, how, steps, conditioned, candidates, evidence;
  and `trace`, every decision in order (the free reply, values kept, each step's 20
  tokens with logprobs and the rank taken, forced characters, the closing token, each
  free continuation; `iri_trie.TRACE_EVENTS`).
- Seeing it: `python -m tests.homebench.decoding_report --config
  tests/homebench/variants/homebench.ecai.M3.pv-trie.yaml --episode <episode>` runs one
  episode up to structuring (the prompt report of `prompt_report.py` in the same
  folder) and writes `decoding.md`: a timeline of the decisions, per value the candidates
  as a prefix tree, the reply the model saw, each step's tokens (allowed or not, taken,
  how far the trie overruled the model), and the free reply against the decoded one.
  `--record <episode.json>` draws a record already on disk.
- Differences from GENRE: greedy, not beam search; a character trie, so any tokenization
  of a name is accepted; constrained only inside quoted IRI values, so `null` stays
  the model's choice (an infeasible request can still name no device).
- Cost: the meter counts the full prompt on every call (`tokens_in`), most of it served
  from Ollama's prompt cache; the generated tokens and the wall time are the cost.

### H: GENRE with a mask on the logits (`decoding: genre`)

G is GENRE emulated: top 20 tokens per step, greedy, `nearest` when none of the 20 is
allowed, one API call per token. H is the same constraint where GENRE applies it. G stays
as it ran (`homebench.ecai.M3.pv-trie.yaml`); H is a second arm, not a replacement.
Code: `utils/genre_decoding.py` (the rules, no torch) and `tests/simuhome/bench/genre_server.py`
(the model).

- The server. An OpenAI-compatible endpoint that runs `DeepSeek-R1-Distill-Qwen-7B` (the
  Hugging Face weights, bf16, not Ollama's Q4_K_M) on the job's GPU. The harness job starts
  it when `GENRE_MODEL_DIR` is set (`slurm_harness.sbatch`) and rewrites the variant's
  `genre` provider (`http://localhost:11600/v1`) to the job's port. Only structuring goes
  to it; segmentation, planning, profiling, guarding and the response stay on Ollama.
- A call. The thinking is decoded freely (greedy) up to `</think>`. The answer is then
  decoded twice from that thinking: freely (the control on this backend, recorded as
  `free_answer`) and by beam search (`num_beams`, default 4) under a `LogitsProcessor`.
  At each step the processor reads the text written so far; if it ends inside the quoted
  value of a slot that names something of the home, only the token ids that keep the
  value inside that slot's trie keep their score. The trie is G's, conditioned on the
  slots the same object already holds. Elsewhere (keys, free text, `null`) only the
  tokens that can continue a JSON object are allowed (see "The escape through `null`").
- Tokens, not characters. The trie stays on characters: a character trie over the text
  of every token id of the vocabulary (`TokenTrie`) is walked along what remains of each
  valid value, so any tokenization of a name is allowed, and a token may also finish the
  name and close the string (`"`, `",`, `"}`). A value that leaves its trie (a dead end)
  is not held to it (only to JSON), and counted (`dead_steps`).
- What it removes from G: the top 20 and `nearest`, the greedy choice of a namespace, one
  call per token, and the closing-token bug (`"` against `",`). What it does not add: an
  exit for "known term, absent from the home" (an infeasible request can still be
  substituted by a valid name); that is a separate change.
- The end of the sequence. Beam search scores whole sequences, so the mask also holds the
  end-of-sequence token until the reply is a complete JSON object (`genre_decoding.
  may_end`; a leading code fence is ignored). Without it, `length_penalty` 0 gave
  one-token answers (an immediate stop) and 7 of 7 smoke episodes failed to parse in 4.
  `length_penalty` is the server's `--length-penalty` (job: `GENRE_LENGTH_PENALTY`);
  default 1.0 (transformers'). Of 7 smoke episodes, 1.0 scored 6 and 0.0 scored 5 (0.0
  re-wrote a correct free answer of `hb2_feasible_seed_101`: `living_room_light` ->
  `LivingRoom`); too few to choose by, 1.0 was kept as the default.
- D58 (evidence from the later slots) is not applied: beam search compares whole
  sequences, so a class followed by the action only it allows should win by probability.
  Whether it does is a thing H measures (`hb1_feasible_seed_4351`).
- Recorded: `structures[].decoding` = the server's record: `beams`, `length_penalty`,
  `thinking_tokens`, `answer_tokens`, `masked_steps`, `dead_steps`, `ends_held` (rows where
  the end was refused, summed over steps), seconds per phase, the free answer and
  `changed` (the slots the constraint changed, free against decoded).
- Smoke test on fep (2026-11-15, one H200, 1 worker, 7 episodes: `hb1_feasible_seed_
  {244,4351}`, `hb1_infeasible_seed_{1121,1303,1424,4644}`, `hb2_feasible_seed_101`;
  records in `~/experiments/semantic/harness/genre-eos2-lp1.0`): the server loads in
  ~75 s (15 GB, bf16), the token trie of 151,665 tokens in 1 s; per episode ~40-55 s,
  structuring ~40 s of it (thinking ~15-20 s, free answer ~9 s, beam answer ~10 s).
  `hb1_feasible_seed_4351` passes: the free answer of these weights already wrote
  `saref:SetAbsoluteLevelCommand` + `temperature`, and the beam kept it. Masked steps
  120-135 per constrained answer, `dead_steps` 0. Notation fixed in the same way as G
  (`StoreRoom` -> `homeont:StoreRoom`). Not a result: 7 episodes.
- The escape through `null` (fixed). `hb1_infeasible_seed_4644`: the free answer names
  `StoreRoom`/`Humidifier` (no prefix); under the mask the first token of the value must
  start `homeont:`, which the model rates low, so the beam took `"location_class": null`
  (unmasked) and went on with a `// comment`, which is not JSON. `may_end` never opened,
  the answer ran to the 2048-token cap (125 s) and the structure was an error, with
  `length_penalty` 0 and 1. The fix is not about `null`: outside a constrained value the
  mask now also keeps the reply a JSON object (`genre_decoding.JsonPrefix`, a character
  automaton; `json_allowed` tests the best-scored tokens of each row, widening to the
  whole vocabulary only when too few pass). A comment, prose after the object or a second
  fence cannot be written; `null` stays a way out for a request the home cannot serve,
  which is what an infeasible episode needs (refusing it would force a substitution such
  as G's `Humidifier`). The end of the sequence is allowed exactly when `may_end`.
  Recorded: `json_steps` (rows so masked), `json_widened`. Not done: falling back to the
  free answer when the beam answer is not JSON; with this mask a non-JSON answer can only
  be one cut at the token cap.
- Control. Compare H with M3 only for episodes that reach the same structuring input;
  the free answer of the server is the control on the HF backend (weights and chat
  template differ from Ollama's). A run of `decoding: free` on this backend over the 400
  episodes would separate the backend from the constraint; it is not part of this arm.
- Run (on fep, from the repo root; a value with commas goes in the shell, not in
  `--export`):

  ```
  sbatch --export=ALL,REPO=$HOME/llm-agents-for-ami,\
  VARIANT=tests/homebench/variants/homebench.ecai.M3.pv-genre.yaml,\
  GENRE_MODEL_DIR=$HOME/hf_models/DeepSeek-R1-Distill-Qwen-7B,WORKERS=1 \
  tests/simuhome/bench/slurm_harness.sbatch
  ```

  Add `ATTEMPT=genre-ecai400` (in the shell, as in the sbatch header) to pick the results
  directory under `$HOME/experiments/semantic/harness/`.
- Safeguards, from the first ECAI-400 run (job 300191: the server was killed by the
  job's 96 GB limit after 34 episodes, and the other 366 episodes failed in 4 s with
  `Connection error`). Cause: `ThreadingHTTPServer` runs each request on a new thread,
  and every thread that runs the model leaves about 1 MB of host memory per generated
  token (0.4 GB for a 400-token request; flat when the same decode runs on one persistent
  thread). `genre_server.py` therefore runs every model call on one worker thread. The
  thinking stops on `ThinkEnd`, a stopping criterion on the decoded last tokens, not on
  transformers' `stop_strings` (same tokens, checked on the real model; the latter left
  40-70 MB per call in a controlled test). What is left is growth with the shapes the
  server has seen: a prompt length not seen before costs memory once (measured on the
  node: 80 distinct lengths, +0.13 GB per 20 requests, then 0 on a second pass over the
  same lengths; the same under `CUDA_MODULE_LOADING=EAGER`, so it is not lazy kernel
  loading), a length seen before costs nothing (9 MB per request on average in the run;
  the resident size also steps up by 0.6-0.9 GB for a prompt longer than any before). The
  prompts span 3k to 15k tokens, so this is bounded; the run's server was at 17 GB after
  1 h and the expectation is a plateau near 20-25 GB (inferred, not measured to the end).
  Around it,
  `slurm_harness.sbatch` has a watchdog (`GENRE_RESTARTS`, default 5): it restarts the
  server within seconds if its process is gone and stops the job after the last restart,
  so a dead server never fills the results with failed records; the episode whose
  request was in flight at the restart fails with `Connection error`. The job directory
  holds `mem.log` (once a minute: the cgroup's anonymous and file memory and the resident
  GB per process) and the server logs, per request, its resident GB after the thinking,
  the free answer and the beam.

### G2 and H2: what the fep runs of G and H called for

Both arms ran on the cluster (400 episodes, commit `bf3be605`; G job 299621, H job
300297), and `python -m tests.homebench.error_report` was run on the records.

| bucket (success) | M3 | G | H |
|---|---|---|---|
| overall (scored) | 22.6% | 34.3% | 32.0% |
| hb1 feasible | 8% | 51% | 65% |
| hb1 infeasible | 88% | 49% | 33% |
| feasible parts passed | 12.5% | 55.3% | 59.4% |
| parts lost to segmentation (`seg_part_lost`) | 2 | 9 | 62 |
| episodes failing on an LLM or endpoint error | 3 | 9 | 18 |

Both gain heavily on feasible requests and lose most of the infeasible ones. M3 passed
those largely because it refused almost everything (broken IRIs found no device), not
because it saw the device was missing.

**G: a replaced device.** Of the 39 `artifact_class` values the trie replaced in the
hb1 infeasible episodes, 36 were a real device kind of the ontology that the room does
not have (`AirConditioner` in a room with a heater, a `Curtain` in a room with none);
the free reply had read the request correctly and the trie overruled it. In the feasible
episodes it was 2 of 21. The failing parts are `infeas_substituted` (33),
`infeas_executed_other` (10) and `infeas_accepted_on_target` (12, some of them label
suspects, see the 657-row note in the dossier). On the hb1 structures of the G run
the rule (with the ontology as the list of device kinds) fires 40 times: 37 on infeasible
requests (21 of them pass today, 16 fail) and 3 on feasible ones (1 passes, 2 fail
anyway). A `DimmableLight` request in a room whose light is of another class is the
case the family rule keeps out.

**H: three causes, none of them the GENRE constraint.**
(1) `null` as a cheap exit. Beam search scores whole sequences, and a `null` costs a few
nats where a long IRI costs more, so slots came out null although the free answer of
the same thinking had named a value: 67 feasible goals lost their room (49 episodes
`struct_room_missing`, against 5 for G). An all-null goal is `unrouted` in the planner,
which is not a refusal either.
(2) Unreadable keys. In 15 episodes beam search wrote `"location-class"` for
`"location_class"`: the mask only recognises the underscore names, so nothing was
constrained and the reply gave "no goal could be structured" (all 15 of the failures
with that message).
(3) A thinking that ran to the token cap ended the request (3 infrastructure errors).

**G2 changes** (`decoding: iri_trie_v2`): abstention. When the first invalid value is an
`artifact_class` naming a device family of the ontology (`homeont.ttl`, classes under
SAREF's Actuator, Appliance, HVAC, Meter, Sensor) that the room the goal names does not
have, and no class of its family is there either, the value is not decoded again: the
object's IRI slots become null and `unmatched_device` the user's words (`air
conditioner`). Not done, on purpose: abstention on actions (`unmatched_action`). Where
`affordance_name` and `affordance_class` were both outside the device's actions, 3 of the
15 feasible hb1 structures pass after the trie replaced them, against at most 2
infeasible ones (`unexist_attribute`) that abstention would turn into a pass. It also is the first run on the 400 of the
evidence rule (`narrow_by_evidence`, 54 of the 56 `struct_wrong_affordance` episodes).

**H2 changes** (`decoding: genre_v2`, all in `genre_server.py` and `genre_decoding.py`):
`no_null` (no `null` token for a room, device or action slot the home has values for;
`parameter_name`, free text and `unmatched_*` stay nullable), `keys` (a JSON key stays a
prefix of a key the free answer wrote), `abstain` (the same rule as G2, read from the
free answer and written over the same goal of the beam answer, paired by position), and
`close_thinking` (a thinking cut at the cap is closed with `</think>` and answered from).
The record of a call says what each did (`filtered_steps`, `abstained`,
`thinking_closed`, `v2`).

**H2 speed.** The first fep run (job 300721) took 244 s per episode against 61 s for H
on the same 66 episodes (a beam of 60 s per call against 5 s): the `keys` and `no_null`
filters refuse nearly every token the model scores high, and testing candidates widened
to the whole vocabulary at almost every step. The allowed ids now come from the rule's
trie (`HomeRules.filter_ids`), the same tokens as `token_filter`; the number of beams is
unchanged. Measured on 7 of the 66 episodes (job 302725, 1 worker): a beam of 9.4 s per
call against 63.5 s before (H: 7.2 s on the same episodes), 77 s per episode against 215 s
(H: 67 s), `json_widened` 7.2 against 242 per call (H: 7.7). The full 400-episode run is
job 302815.

What would count as the result, from the numbers above and not measured yet: G2 can
return at most the 16 hb1 infeasible failures that the rule fires on (49% -> 68% there)
plus its share of the mixed episodes; H2 at most the 48 + 18 episodes lost to the three
causes, and its infeasible rate depends on the same rule. Whether H2 beats G2 is the
question; the free answer of the same thinking, the control H records on its own
backend (`structures[].decoding.free_answer`), stays in the record.

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

Arm F: new unit tests for the version and for the reports (`test_run_config.py`, `TestJsonPromptVersions`); one episode (`hb1_feasible_seed_1085`, `--stop-after structuring`) through the prompt report: the v2 prompt is built and sent, and its report names `context=json_v2`. The model's reply there leaves `parameter_name` null where the prompt says "NA", so the reading of one episode is: the rules are reworded, nothing is shown yet about the results.

Not checked: any arm over more than one episode; the OpenAI endpoint with a schema; the
grammar compile time on the 52-device home; production (agents running under SPADE) with
a variant set in agents.yaml.

Not built: the device-pinned schema, `prompt: terse`, and a script that compares two run
folders paired by episode (step 4 is done by hand from `errors_episodes.csv` and
`errors_parts.csv` until it exists).

## 7. Where a run's configuration is stated

A result must say what it ran with, without reading the prompt text. `run.py` writes into
`<records>/run.json` the model of every step, the non-default prompt choices
(`prompt_variants`) and the whole selection (`prompt_selection`: every axis, the defaults
written out, so a later change of a default does not change what an old run is read as having
used), and the run settings (guidelines, environment, memory, context, ...). Every report
made from a run opens with that, through one module, `tests/simuhome/bench/run_config.py`:

| file | where |
|---|---|
| `stats/summary.md` (`stats.py`) | `## Experiment configuration`, before the legend and tables |
| `report.md` (`report.py`) | `## Configuration: <variant>` per variant |
| `errors.md` (`error_report.py`) | `## Configuration of the run`, before section 1 |
| prompt report `README.md` | `Prompt variants: ...` line, then the section; also `abstract.md`, `index.md` and the head of every call page |
| `--dry-run` and the run's own output | every axis, the non-default marked |

A run.json from before `prompt_selection` is read with the defaults of the current code and
says so; a run folder without a run.json says that the configuration is not recorded.
