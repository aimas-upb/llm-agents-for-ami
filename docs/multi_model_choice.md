# Multi-model choice: the NLU/planner role split

This note records why the SimuHome generation-mode experiment (JSON IR vs.
builder-DSL code vs. free py_trees code) can run the two LLM-driven agents on
**different models**, how that is wired, and how to read results produced
under each protocol.

## The problem

Every SimuHome run passes through two LLM-driven agents in sequence:

1. **UserAssistant (NLU stage).** Reads the user's request, extracts and
   classifies a structured intent, handles clarification/confirmation, and
   forwards the intent to the InteractionSolver.
2. **InteractionSolver (planner stage).** Gathers the environment context and
   runs the plan generator. This is the *only* place where the three
   generation modes (`behavior_tree`, `python_code`, `py_trees_code`) differ.

The original harness gave both agents one model: the ablation runner exported
a single `OPENAI_MODEL`, and `run_cases.py` wrote it into the shared
`llm.providers.openai` block that both agents' client factories read.

The first SLM grid (3 models x 3 modes, 481 valid runs) showed why that is a
problem. Two of the three small models (codegemma-7B-instruct,
deepseek-coder-1.3B) failed at the UserAssistant stage in essentially every
run and never produced a plan. For those six grid cells the generation mode
was never exercised: the cells measure intent extraction, not IR-vs-code.
Even qwen2.5-coder-3B, which did reach planning, lost a share of runs upstream
of the planner.

## The decision

Hold the NLU stage fixed on a strong model and vary **only the planner model**
across the SLMs. The independent variables then are exactly the two things
the experiment is about, planner model and generation mode, and the upstream
stage is a controlled constant rather than a per-model failure source.

What this changes about the claim:

- **Single-model protocol** (both agents on the SLM): "can this small model
  drive the whole pipeline end to end, and does output structure help?"
- **Role-split protocol** (NLU fixed, planner varied): "given a correctly
  extracted intent, how does output structure affect small-model planning?"

The role split is the protocol for the main IR-vs-code table. The existing
single-model runs remain a valid *separate* condition ("end-to-end SLM") and
must not be pooled with role-split rows; see "Reading results" below.

Reference conditions (GPT-5-mini, GPT-4o as planner) must use the **same NLU
model** as the SLM cells so that every row in the table shares the upstream
stage. The NLU model is a cost consideration: every SLM run now spends OpenAI
calls on the NLU turns, so pick the cheapest model that extracts intents
reliably (GPT-5-mini is the working default).

Two harness details follow from the same principle:

- The **simulated user** (the harness's reply generator for confirmation and
  option-choice turns) runs on the NLU model, not the planner. It is test
  machinery, and a weak planner model must not be able to break the
  confirmation turn as a side effect.
- The planner keeps the **yaml planning temperature** (0.5) regardless of the
  NLU model. Without that, a gpt-5 NLU model's pinned temperature of 1.0 would
  leak into every SLM planner and confound the planner comparison.

## How it is wired

`run_cases.py` reads two families of environment variables:

| Variable | Agent | Notes |
|---|---|---|
| `OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_REASONING_EFFORT` | UserAssistant, and the InteractionSolver unless overridden | Unchanged single-model behaviour |
| `PLANNER_MODEL` | InteractionSolver only | Presence of this variable switches on the role split |
| `PLANNER_BASE_URL` | InteractionSolver only | Defaults from the tag shape, like `OPENAI_MODEL` does |
| `PLANNER_API_KEY` | InteractionSolver only | Defaults to `OPENAI_API_KEY` |
| `PLANNER_REASONING_EFFORT` | InteractionSolver only | Written even when empty, so a local planner never inherits the NLU model's effort setting |

The planner values are written into `planning.llm_planning` (and mirrored
into `interaction_solver.planning.llm_planning`, the `main.py` config shape).
The InteractionSolver's client factory
(`ami_agents/agents/interaction_solver/utils/llm_client.py`) reads that
section before the shared `llm.providers` block for model, base URL, API key,
reasoning effort and temperature. The UserAssistant's factory reads only the
shared block, so it is unaffected.

`tests/simuhome/run_ablation_modes.sh` exposes the split as one variable:

```bash
# Single-model protocol (default): each MODELS entry runs both agents.
MODELS="qwen2.5-coder:3b" ... run_ablation_modes.sh

# Role split: UserAssistant pinned to gpt-5-mini, MODELS entries are planners.
NLU_MODEL=gpt-5-mini OPENAI_API_KEY=sk-... MODELS="qwen2.5-coder:3b codegemma:7b-instruct deepseek-coder:1.3b" ... run_ablation_modes.sh
```

Backend resolution is per model tag (a `:` or `/` in the tag means Ollama via
`OLLAMA_BASE_URL`, otherwise OpenAI), applied independently to the NLU model
and to each planner model. Mixed backends are the normal case: NLU on OpenAI,
planner on the node-local Ollama.

The SLURM shard job (`tests/simuhome/slurm_experiment_shard.sbatch`) passes
`NLU_MODEL` straight through and fails before starting Ollama when the NLU
model is an OpenAI model but no API key is in the submitting environment:

```bash
OPENAI_API_KEY=sk-... sbatch --export=ALL,NLU_MODEL=gpt-5-mini,MODELS="qwen2.5-coder:3b",RUNS=10 \
  tests/simuhome/slurm_experiment_shard.sbatch
```

## Reading results

- Every results row carries `model_name` (the UserAssistant/NLU model) and
  `planner_model_name` (the InteractionSolver model). In single-model runs
  the two are equal.
- Role-split CSVs are named `results_<planner>_nlu-<nlu>_<mode>_<flavor>.csv`
  so they never collide with single-model CSVs of the same planner.
- Older CSVs written before `planner_model_name` existed are migrated on the
  next append: the column is filled from `model_name`, which is correct
  because every pre-split run was single-model.
- Group by `(model_name, planner_model_name, generation_mode)` when
  aggregating. Rows with `model_name != planner_model_name` belong to the
  role-split protocol; do not average them with single-model rows.
- The validity filter from the first grid still applies: exclude rows with
  `llm_calls == 0` (dead endpoint), and check that the NLU stage produced an
  intent before attributing a failure to the planner.
