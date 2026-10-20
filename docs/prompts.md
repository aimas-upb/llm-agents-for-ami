# LLM Prompt Reference

AmI HMAS calls an LLM only at fixed points inside deterministic behaviours. Everything an LLM
does not need to decide is decided in code: routing, explicit `set` / `modify` plans, dependency
templates, power-on and cycle guards, and most state and capability answers. This page lists every
prompt, when it is called, what it returns, and which config key picks its model. The config keys
are explained in [configuration.md](configuration.md#llm-configuration-agentsyaml).

| Agent | Prompt files |
|---|---|
| UserAssistant | [`agents/user_assistant/prompts/`](../ami_agents/agents/user_assistant/prompts/): `intent_segmentation_prompts.py`, `intent_parsing_prompts.py`, `goal_structuring_prompts.py`, `intent_response_prompts.py` |
| InteractionSolver | [`agents/interaction_solver/prompts.py`](../ami_agents/agents/interaction_solver/prompts.py) and [`bt_planning/planning/prompts.py`](../ami_agents/bt_planning/planning/prompts.py) |
| EnvExplorer | none: discovery, semantic resolution (SPARQL) and memory lookup are deterministic |

---

## UserAssistant

### 1. `ATOMIC_SEGMENTATION_SYSTEM_PROMPT`: segmentation

- **When.** Once per utterance, in the `SEGMENTING` state (`pipeline.py`).
- **Does.** Splits the utterance into atomic intents and types each one `GOAL_REQUEST`,
  `ENV_STATE_REQUEST` or `ENV_CAPABILITIES_REQUEST`. Each goal gets qualifiers: one specificity
  (`explicit` / `incomplete` / `ambiguous`), one kind (`achievement` / `maintenance`), and any
  structure labels (`logical_dependency`, `temporal_dependency`, `sequential_dependency`).
- **Environment.** It sees none; the split is purely linguistic. Its main rule: a split must not
  create a fragment that is incomplete only because it was cut away.
- **Model.** `user_assistant.llm.atomic_segmentation`.

### 2. `ENV_STATE_REQUEST_PARSER_PROMPT` and `ENV_CAPABILITIES_REQUEST_PARSER_PROMPT`: query structuring

- **When.** In `EXTRACTING`, once per state or capability intent
  (`behaviours/env_state_structuring.py`, `behaviours/env_capability_structuring.py`).
- **Does.** Extracts the ontology *classes* the question is about: location class, device class,
  device property or environment variable. A capability question also gets a command class and
  its performative (`query` or `query_if`). The prompts never name individual devices;
  EnvExplorer resolves the classes over the TD graph.
- **Model.** `user_assistant.llm.intent_parsing`.

### 3. `GOAL_STRUCTURING_SIMPLE_PROMPT` and `GOAL_STRUCTURING_DEPENDENCY_PROMPT`: goal structuring

- **When.** In `EXTRACTING`, once per goal (`behaviours/goal_structuring.py`).
  `utils/goal_routing.py` picks the prompt from the segmenter's qualifiers: *simple* for an
  achievement with no dependency label, *dependency* otherwise.
- **Does.** Turns the goal into a `GoalStructure` (`shared/models/goal_structure.py`):
  - the simple prompt gives one goal;
  - the dependency prompt gives goals plus the predicates that condition them, each with its
    role, the goals it applies to, and its timing.
- **Shared fragments.** Both prompts are built from the same fragments (`GOAL_SPEC_RULES`,
  `PREDICATE_SPEC_RULES`, `BINDING_RULES`, …), so goal fields cannot drift between them.
- **Context.** The home goes in through `render_goal_prompt` (`{goal_context}`): the actions,
  and where needed device properties and environment variables, fetched from EnvExplorer.
- **Plans.** Neither prompt composes a plan.
- **Model.** `user_assistant.llm.intent_parsing`.

### 4. `STATE_ANSWER_PROMPT`: state answers

- **When.** Only when the answer cannot be built in code (`utils/state_answers.py`). Examples:
  choosing one field out of a structured reading, or telling a collective question from a
  single-device one (`behaviours/state_answer.py`).
- **Model.** `user_assistant.llm.state_answer`.

### 5. `CAPABILITY_ANSWER_PROMPT`: capability answers

- **When.** Phrases a capability answer that `utils/capability_answers.py` cannot build directly
  (`behaviours/capability_answer.py`).
- **Model.** `user_assistant.llm.capabilities_analysis`.

### 6. `PLAN_SUMMARY_SYSTEM_PROMPT`: plan summary

- **When.** In `SUMMARIZING`, for a plan with action nodes (`pipeline.summarize_plan`).
- **Does.** A 2–4 sentence plain-language summary ending with a confirmation question. Read-only
  plans are not summarised.
- **Model.** The UA's default client, which is `atomic_segmentation`. The
  `user_assistant.llm.plan_summarization` key exists in `agents.yaml`, but the summary does not
  read it today.

---

## InteractionSolver

### 7. `MODIFY_RECIPE_PROMPT`: explicit goals the TDs cannot settle

- **When.** An explicit goal could not be planned deterministically
  (`behaviours/explicit_goal_llm_planning.py`):
  - a `modify` whose read → compute → set recipe cannot be read off the TDs, or
  - a `set` whose value is not determined.
- **Does.** Names the pieces: the property to read, the computation, and the action to set.
  The ISA checks them against the device's TD and builds the tree itself.
- **Model.** `interaction_solver.planning.explicit_planning` (the small model).

### 8. `GOAL_PROFILE_PROMPT`: dependency profiling

- **When.** In dependency planning, when the deterministic checklist (`utils/goal_profile.py`)
  has open questions (`behaviours/dependency_goal_planning.py`, `utils/profiling.py`).
- **Does.** Answers only the checklist questions: which event a dependency refers to, and which
  properties the plan must read to know when it happens. It never chooses commands.
  Unanswered questions take documented fallbacks.
- **Model.** `interaction_solver.planning.explicit_planning`.

### 9. `CONDITION_GUARD_PROMPT`: condition guards

- **When.** A planned step targets a device that has a condition in
  `config/planning_guidelines.yaml` and the guard memory has no decision for it
  (`utils/condition_check.py`, called from `behaviours/condition_guard.py` and from dependency
  planning).
- **Does.** For each such step, gives either a check (property, `==` / `!=`, value) and the action
  that establishes it, or "no guard". Decisions are remembered after a successful run.
- **Model.** `interaction_solver.planning.explicit_planning`.

### 10. `BT_PLANNING_SYSTEM_PROMPT`: the general BT planner

- **When.** In `AsyncBTPlanner.generate_bt` (`bt_planning/planning/bt_planner.py`), called for:
  - ambiguous goals (`behaviours/ambiguous_goal_planning.py`);
  - the plain-text fallback workflow (`behaviours/bt_plan_generation.py`).
- **Does.** Produces a BT JSON IR through the `generate_behavior_tree` tool. Node types are
  `sequence`, `selector`, `parallel`, `action`, `condition`, `wait_condition` and
  `ignore_failure`, and affordances are named by artifact, name and class. Invalid output is
  retried with validation feedback, up to `max_planning_attempts`.
- **Context formatters** (same module):
  - `format_capability_context()` adds the affordances and current state;
  - `format_observable_property_hints()` adds readings of the variables involved;
  - `format_signifier_hints()` adds what a remembered plan used. Hints take precedence over the
    context, but the LLM still plans.
- **Model.** `interaction_solver.planning.llm_planning`.
