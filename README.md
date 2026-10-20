# AmI HMAS: Hybrid Agents for Goal-Driven Interaction with Smart Homes

AmI HMAS is a multi-agent system that turns what a person says into plans that run in a smart home.
The home is described as W3C WoT Thing Descriptions in a Hypermedia MAS (HMAS) environment,
typed with semantic vocabularies (HomeOnt, SAREF, SOSA / TD-SOSA). Real homes come from
**Home Assistant**, and **Matter** devices come through the vendored Matter data model. The agents
are deterministic SPADE behaviours with LLM calls only at fixed points. Plans are
Behavior Trees, encoded as a JSON IR and ticked against the devices' affordances.

Branch: `ami-hmas-semantic`.

---

## What the agents do

| Agent | Role | LLM use |
|---|---|---|
| **UserAssistant (UA)** | Splits an utterance into atomic intents and structures each one against the home. Answers state and capability questions. Asks for clarification, summarises and confirms plans, executes them, and records what worked. | segmentation, structuring, answers, plan summary |
| **InteractionSolver (ISA)** | Plans each structured goal on the path its specificity calls for. Builds dependency plans (conditions, timing, maintenance) from templates. Guards every plan against device conditions. | only where a plan cannot be derived deterministically |
| **EnvExplorer** | Discovers the HMAS environment and keeps its state. Resolves properties and capabilities semantically (SPARQL over the TDs + HomeOnt + SAREF). Hosts the experience memory (signifiers and condition guards). | none |

### How a request flows

Take *"Turn on the kitchen light and keep the bedroom at 22 degrees; how humid is the bathroom?"*

1. **Segmentation (UA).** One LLM call splits the utterance into atomic intents and types each one:
   two `GOAL_REQUEST`s and one `ENV_STATE_REQUEST`. Each goal is qualified:
   - specificity: `explicit` / `incomplete` / `ambiguous`;
   - kind: `achievement` / `maintenance`;
   - structure: `logical_dependency` / `temporal_dependency` / `sequential_dependency`.
2. **Structuring (UA), all intents in parallel.**
   - The state question is parsed against the home and answered at once.
   - Each goal is routed to the *simple* or *dependency* structuring prompt (`utils/goal_routing.py`).
     It is structured into a `GoalStructure` against the goal context fetched from EnvExplorer.
   - Goals the home makes unambiguous are promoted to explicit (`utils/goal_resolution.py`).
   - Goals only the user can complete ("turn it off") get a clarification question and never reach the ISA.
3. **Planning (ISA).** Each structure goes to its path:
   - `explicit`: deterministic `set` / `modify` trees, with a small-model fallback.
   - `incomplete`: candidate devices, then a choice, an "impossible", or a question.
   - `ambiguous`: the planner model over the variables the request implies.
   - `dependency`: achievement steps, then predicates, profiling, time checks, and finally the template compiler.

   Every plan is then guarded against device conditions from `config/planning_guidelines.yaml`:
   the operational-cycle checks, power-on first, and condition guards.
4. **Summary and confirmation (UA).** Action plans are summarised and confirmed. Without an
   answer they auto-confirm after `confirmation_timeout`. Read-only plans run at once.
5. **Execution (UA).** Achievement plans tick to completion. Maintenance plans tick in bursts until
   cancelled. Dependency plans wait on their conditions and on the clock (system time, or
   a simulator's virtual time).
6. **Memory.** A successful plan made by an LLM is recorded under an exact key
   (`shared/utils/goal_signifiers.py`). The planner reuses it on an exact match, and can share it through
   the community store. Condition-guard decisions are remembered the same way.

The full lifecycle, with the files for each step and the message types, is in
[docs/workflow.md](docs/workflow.md).

---

## Where things are

### Agents

| Component | Code |
|---|---|
| UA request FSM (segment → extract → plan → summarise → confirm → submit) | [`agents/user_assistant/behaviours/user_request.py`](ami_agents/agents/user_assistant/behaviours/user_request.py) |
| UA goal structuring and routing | [`behaviours/goal_structuring.py`](ami_agents/agents/user_assistant/behaviours/goal_structuring.py), [`utils/goal_routing.py`](ami_agents/agents/user_assistant/utils/goal_routing.py), [`utils/goal_resolution.py`](ami_agents/agents/user_assistant/utils/goal_resolution.py) |
| UA plan execution (achievement / maintenance / dependency) | [`behaviours/plan_execution.py`](ami_agents/agents/user_assistant/behaviours/plan_execution.py), [`behaviours/plan_management.py`](ami_agents/agents/user_assistant/behaviours/plan_management.py) |
| Goal structure model | [`shared/models/goal_structure.py`](ami_agents/shared/models/goal_structure.py) |
| ISA routing | [`agents/interaction_solver/behaviours/goal_request.py`](ami_agents/agents/interaction_solver/behaviours/goal_request.py) |
| ISA planning paths | [`behaviours/explicit_goal_planning.py`](ami_agents/agents/interaction_solver/behaviours/explicit_goal_planning.py), [`incomplete_goal_planning.py`](ami_agents/agents/interaction_solver/behaviours/incomplete_goal_planning.py), [`ambiguous_goal_planning.py`](ami_agents/agents/interaction_solver/behaviours/ambiguous_goal_planning.py), [`dependency_goal_planning.py`](ami_agents/agents/interaction_solver/behaviours/dependency_goal_planning.py) |
| Dependency template compiler | [`interaction_solver/utils/dependency_compiler.py`](ami_agents/agents/interaction_solver/utils/dependency_compiler.py) |
| Plan guards | [`utils/cycle_guard.py`](ami_agents/agents/interaction_solver/utils/cycle_guard.py), [`utils/power_on_guard.py`](ami_agents/agents/interaction_solver/utils/power_on_guard.py), [`utils/condition_guard.py`](ami_agents/agents/interaction_solver/utils/condition_guard.py), rules in [`config/planning_guidelines.yaml`](ami_agents/config/planning_guidelines.yaml) |
| EnvExplorer request handlers | [`agents/env_explorer/behaviors/`](ami_agents/agents/env_explorer/behaviors/) |
| Semantic property / capability resolution | [`env_explorer/utils/state_resolution.py`](ami_agents/agents/env_explorer/utils/state_resolution.py), [`utils/capability_resolution.py`](ami_agents/agents/env_explorer/utils/capability_resolution.py), ontology [`shared/ontologies/homeont.ttl`](ami_agents/shared/ontologies/homeont.ttl) |

### Planning and execution

| Component | Code |
|---|---|
| LLM BT planner (`AsyncBTPlanner`) | [`bt_planning/planning/bt_planner.py`](ami_agents/bt_planning/planning/bt_planner.py) |
| BT JSON IR schema | [`bt_planning/planning/schema.py`](ami_agents/bt_planning/planning/schema.py) |
| Tick-based executor (`IRExecutor`) | [`bt_planning/execution/ir_executor.py`](ami_agents/bt_planning/execution/ir_executor.py) |
| Node types and registry (affordance, compute, dependency nodes) | [`bt_planning/nodes/`](ami_agents/bt_planning/nodes/) |

### Experience memory

| Component | Code |
|---|---|
| Exact signifier key (shared by recording and recall) | [`shared/utils/goal_signifiers.py`](ami_agents/shared/utils/goal_signifiers.py) |
| Signifier model, signifier store, guard store | [`shared/memory/src/models/signifier.py`](ami_agents/shared/memory/src/models/signifier.py), [`storage/memory_store.py`](ami_agents/shared/memory/src/storage/memory_store.py), [`storage/guard_store.py`](ami_agents/shared/memory/src/storage/guard_store.py) |
| Community store client (sharing across homes) | [`shared/community/community_client.py`](ami_agents/shared/community/community_client.py) |

### Environments

| Integration | What it serves | Code |
|---|---|---|
| **HASP** | A Home Assistant deployment as an HMAS platform. Areas become workspaces and devices become artifacts with semantically typed TDs. Changes are pushed through WebSub. | [`environment/integration/HomeAssistant/`](ami_agents/environment/integration/HomeAssistant/) |
| **SHTD** | A Matter-device home from the SimuHome simulator as TDs, built from the vendored Matter data model. This is the benchmark environment. | [`environment/integration/SimuHome/`](ami_agents/environment/integration/SimuHome/) |
| Matter data model | The CSA Matter registry (spec 1.5.1), vendored. It is the single source of Matter facts. | [`environment/integration/SimuHome/matter_model/`](ami_agents/environment/integration/SimuHome/matter_model/) |

### Prompts

| Prompt | Agent / step | File |
|---|---|---|
| `ATOMIC_SEGMENTATION_SYSTEM_PROMPT` | UA: segmentation | [`user_assistant/prompts/intent_segmentation_prompts.py`](ami_agents/agents/user_assistant/prompts/intent_segmentation_prompts.py) |
| `ENV_STATE_REQUEST_PARSER_PROMPT`, `ENV_CAPABILITIES_REQUEST_PARSER_PROMPT` | UA: query structuring | [`user_assistant/prompts/intent_parsing_prompts.py`](ami_agents/agents/user_assistant/prompts/intent_parsing_prompts.py) |
| `GOAL_STRUCTURING_SIMPLE_PROMPT`, `GOAL_STRUCTURING_DEPENDENCY_PROMPT` | UA: goal structuring | [`user_assistant/prompts/goal_structuring_prompts.py`](ami_agents/agents/user_assistant/prompts/goal_structuring_prompts.py) |
| `STATE_ANSWER_PROMPT`, `CAPABILITY_ANSWER_PROMPT`, `PLAN_SUMMARY_SYSTEM_PROMPT` | UA: answers and plan summary | [`user_assistant/prompts/intent_response_prompts.py`](ami_agents/agents/user_assistant/prompts/intent_response_prompts.py) |
| `MODIFY_RECIPE_PROMPT`, `GOAL_PROFILE_PROMPT`, `CONDITION_GUARD_PROMPT` | ISA: explicit fallback, dependency profiling, condition guards | [`interaction_solver/prompts.py`](ami_agents/agents/interaction_solver/prompts.py) |
| `BT_PLANNING_SYSTEM_PROMPT` | ISA: general BT planner | [`bt_planning/planning/prompts.py`](ami_agents/bt_planning/planning/prompts.py) |

All prompts: [docs/prompts.md](docs/prompts.md).

---

## Quick start

```bash
git clone https://github.com/aimas-upb/llm-agents-for-ami.git
cd llm-agents-for-ami && git checkout ami-hmas-semantic
conda create -n ami-agents python=3.12 -y
conda run -n ami-agents pip install -r agents-requirements.txt
export OPENAI_API_KEY="<your-key>"          # or put it in .env
```

You also need:
- an XMPP server: Prosody, see [docs/installation.md](docs/installation.md#step-4--xmpp-server);
- one environment, either HASP over Home Assistant, or SHTD over a SimuHome episode.

Then talk to the agents from a console:

```bash
YGGDRASIL_URL=http://127.0.0.1:8097 conda run -n ami-agents python tests/manual_interactive_cli.py
```

That URL is SHTD's; use `http://localhost:8080/` for HASP.

---

## Evaluating on a benchmark

The SimuHome bench runs the agents' reasoning steps in-process, without SPADE. It times each step,
counts its LLM calls, and scores every episode with SimuHome's own evaluators.

```bash
# one episode, every step printed, scored at once (~5 LLM calls)
conda run -n ami-agents python -m tests.simuhome.bench.run --episode qt4-2_feasible_seed_1

# the full harness from one config: see the plan and the cost first
conda run -n ami-agents python -m tests.simuhome.bench.harness \
    --config tests/simuhome/bench/variants/harness.example.yaml --dry-run
```

Setup, every command, the config formats, outputs, and how to add another benchmark:
[docs/evaluation-simuhome.md](docs/evaluation-simuhome.md).

---

## Documentation

| Document | Contents |
|---|---|
| [docs/installation.md](docs/installation.md) | Python env, XMPP, Home Assistant + HASP, SimuHome + SHTD |
| [docs/configuration.md](docs/configuration.md) | YAML config files, LLM per step, clock, memory, environment variables |
| [docs/workflow.md](docs/workflow.md) | The request lifecycle in detail: UA FSM, ISA paths, guards, execution, memory, messages |
| [docs/prompts.md](docs/prompts.md) | Every LLM prompt: when it is called, what it returns, its config key |
| [docs/project-structure.md](docs/project-structure.md) | Directory layout and extension points |
| [docs/evaluation-simuhome.md](docs/evaluation-simuhome.md) | Running the evaluation suite on SimuHome, and on new benchmarks |

---

## Technologies

- **[SPADE](https://spade-mas.readthedocs.io/)**: multi-agent framework over XMPP
- **[py_trees](https://py-trees.readthedocs.io/)**: Behavior Tree execution
- **[W3C WoT Thing Description](https://www.w3.org/TR/wot-thing-description/)** and the **[HMAS ontology](https://purl.org/hmas/)**: hypermedia environment
- **[SAREF](https://saref.etsi.org/)**, **[SOSA](https://www.w3.org/TR/vocab-ssn/)** / TD-SOSA, HomeOnt: semantic typing of devices, properties and effects
- **[Matter](https://csa-iot.org/all-solutions/matter/)** data model (vendored, spec 1.5.1)
- **[Home Assistant](https://www.home-assistant.io/)**: real smart-home platform
- **[SimuHome](https://github.com/holi-lab/SimuHome)**: Matter home simulator and benchmark
- **OpenAI API** or any OpenAI-compatible endpoint (local models through Ollama in the bench)
