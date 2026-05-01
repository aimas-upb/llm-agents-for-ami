# AmI HMAS: Hybrid Agents with Individual and Collective Experience-Aware Code-based Planning for Smart Environments

**EMAS 2026** — Engineering Multi-Agent Systems Workshop

> Vlad-Alexandru Florea, Alexandru Sorici, Vlad-Matei Drăghici, Andrei-Cătălin Barbu, Andrei Olaru
> Department of Computer Science and Engineering, UNSTPB, Bucharest

**[Paper](#)** | **[Demo video](https://youtu.be/qPZlZ1Rz6eY)** | **[Repository](https://github.com/aimas-upb/llm-agents-for-ami/tree/emas2026)**

---

## What This Repository Contains

This repository contains the full source code of **AmI HMAS**, a framework for goal-driven,
LLM-supported interactions with smart environments. It maps HomeAssistant deployments into
semantically represented Hypermedia Environments (HMAS / W3C WoT), and combines classical
agent control with LLM reasoning at controlled injection points to plan and execute user goals
as BehaviorTree-based procedural plans.

The four main contributions of the paper are implemented here:

| # | Contribution | Key code location |
|---|---|---|
| 1 | **HMAS integration pipeline** — HA → W3C WoT Thing Descriptions | [`ami_agents/environment/integration/HomeAssistant/`](ami_agents/environment/integration/HomeAssistant/) |
| 2 | **Hybrid agent architecture** — deterministic state machine + LLM injection | [`ami_agents/agents/`](ami_agents/agents/) |
| 3 | **Signifier Memory Engine** — experience storage, embedding + SHACL matching | [`ami_agents/shared/memory/`](ami_agents/shared/memory/) |
| 4 | **Community-based experience sharing** — cross-environment intent-level transfer | [`ami_agents/shared/community/`](ami_agents/shared/community/) |

---

## Quick Start

```bash
git clone https://github.com/aimas-upb/llm-agents-for-ami.git
cd llm-agents-for-ami && git checkout emas2026
python -m venv venv && source venv/bin/activate
pip install -r agents-requirements.txt
export OPENAI_API_KEY="<your-key>"
spade run   # in a separate terminal
python tests/manual_test_full_flow_plan.py --sequence 3 --clear-signifiers
```

Full setup (HomeAssistant, HASP): [docs/installation.md](docs/installation.md)

---

## Paper → Code Navigation

Use this table to go directly from a paper section, figure, or listing to the implementing code.

### Agents (Section 3.2)

| Agent | Role | Implementation |
|---|---|---|
| **UserAssistant** | User interface; NLU/NLG; deterministic 6-state lifecycle | [`ami_agents/agents/user_assistant/`](ami_agents/agents/user_assistant/) |
| **EnvExplorer** | WoT discovery; artifact state tracking; signifier management | [`ami_agents/agents/env_explorer/env_explorer_agent.py`](ami_agents/agents/env_explorer/env_explorer_agent.py) |
| **InteractionSolver** | Dual-path BT planning; community querying | [`ami_agents/agents/interaction_solver/interaction_solver_agent.py`](ami_agents/agents/interaction_solver/interaction_solver_agent.py) |

### BehaviorTree Planning (Section 3.3 — Table 1 in paper)

| Component | Description | Code |
|---|---|---|
| `AsyncBTPlanner` | LLM-based BT JSON IR generation (OpenAI tool call + retry) | [`ami_agents/bt_planning/planning/bt_planner.py`](ami_agents/bt_planning/planning/bt_planner.py) |
| BT JSON IR schema | Node types: `sequence`, `selector`, `parallel`, `action`, `condition` | [`ami_agents/bt_planning/planning/schema.py`](ami_agents/bt_planning/planning/schema.py) |
| `IRExecutor` | Compiles JSON IR → `py_trees` object; tick-based execution | [`ami_agents/bt_planning/execution/ir_executor.py`](ami_agents/bt_planning/execution/ir_executor.py) |
| BT leaf nodes | `ActionAffordanceNode`, `PropertyAffordanceNode`, `PropertyConditionNode` | [`ami_agents/bt_planning/nodes/affordance_nodes.py`](ami_agents/bt_planning/nodes/affordance_nodes.py) |
| Signifier bridge | Walks executed BT → extracts signifier records | [`ami_agents/bt_planning/signifier_bridge.py`](ami_agents/bt_planning/signifier_bridge.py) |

### Dual-Path Planning Algorithm (Section 3.4 — Algorithm 1 in paper)

The fast-path vs. LLM-path decision logic from Algorithm 1 is implemented in:
[`ami_agents/agents/interaction_solver/interaction_solver_agent.py`](ami_agents/agents/interaction_solver/interaction_solver_agent.py)

### Signifier Memory Engine (Section 4 — Listing 1 in paper)

| Component | Description | Code |
|---|---|---|
| Signifier data model | Fields: intent, affordance_uri, payload_hint, structured_conditions | [`ami_agents/shared/memory/src/models/signifier.py`](ami_agents/shared/memory/src/models/signifier.py) |
| Structured matcher | Artifact-type + action + parameter hard-match filter | [`ami_agents/shared/memory/src/matching/structured_matcher.py`](ami_agents/shared/memory/src/matching/structured_matcher.py) |
| SHACL validator | Context validation for implicit intent matching | [`ami_agents/shared/memory/src/validation/shacl_validator.py`](ami_agents/shared/memory/src/validation/shacl_validator.py) |

### Community-Based Sharing (Section 5)

| Component | Description | Code |
|---|---|---|
| Community client | XMPP-based inter-agent signifier query protocol | [`ami_agents/shared/community/community_client.py`](ami_agents/shared/community/community_client.py) |

### HomeAssistant Integration Pipeline (Section 3.1)

| Component | Description | Code |
|---|---|---|
| HASP | Maps HA areas/devices → HMAS workspaces/artifacts with WoT TDs | [`ami_agents/environment/integration/HomeAssistant/hasp.py`](ami_agents/environment/integration/HomeAssistant/hasp.py) |
| Lab308 device config | Virtual Lab308 device definitions (light, blinds, sensors) | [`ami_agents/environment/integration/HomeAssistant/lab308.yaml`](ami_agents/environment/integration/HomeAssistant/lab308.yaml) |
| Setup guide | Step-by-step HA + HASP setup | [`ami_agents/environment/integration/HomeAssistant/README.md`](ami_agents/environment/integration/HomeAssistant/README.md) |

---

## LLM Prompt Index

| Prompt | Agent | Phase | File:symbol |
|---|---|---|---|
| `INTENT_EXTRACTION_SYSTEM_PROMPT` | UserAssistant | `EXTRACTING_INTENTS` — NLU | [`user_assistant/prompts.py`](ami_agents/agents/user_assistant/prompts.py) |
| `PLAN_SUMMARY_SYSTEM_PROMPT` | UserAssistant | `SUMMARIZING_PLAN` — NLG | [`user_assistant/prompts.py`](ami_agents/agents/user_assistant/prompts.py) |
| `QUERY_RESPONSE_SYSTEM_PROMPT` | UserAssistant | Query response — NLG | [`user_assistant/prompts.py`](ami_agents/agents/user_assistant/prompts.py) |
| `BT_PLANNING_SYSTEM_PROMPT` | InteractionSolver | LLM planning path | [`bt_planning/planning/prompts.py`](ami_agents/bt_planning/planning/prompts.py) |
| `format_signifier_hints()` | InteractionSolver | Signifier hint injection | [`bt_planning/planning/prompts.py`](ami_agents/bt_planning/planning/prompts.py) |
| `format_capability_context()` | InteractionSolver | Affordance/state injection | [`bt_planning/planning/prompts.py`](ami_agents/bt_planning/planning/prompts.py) |

**EnvExplorer has no LLM prompts** — it is a classical SPADE agent using deterministic discovery
and embedding-based matching.

Full prompt documentation: [docs/prompts.md](docs/prompts.md)

---

## Reproducing the Experiments

| Experiment | Paper section | How to run |
|---|---|---|
| Explicit request BT planning (HomeBench, 400 cases, 4 models) | Section 6.1, Table 1 | `pytest tests/e2e/test_demo_homebench.py -m e2e` |
| Cold start — all LLM path | Section 6.2, Table 2 Phase A | `pytest tests/e2e/test_demo_lab308.py -m e2e -k cold` |
| Warm start — signifier fast-path | Section 6.2, Table 2 Phase B | `pytest tests/e2e/test_demo_lab308.py -m e2e -k warm` |
| Cross-env transfer | Section 6.2, Table 2 Phase C | `pytest tests/e2e/test_cross_env_sharing.py -m e2e` |

Full details, expected results, and per-phase analysis: [docs/evaluation.md](docs/evaluation.md)

---

## Running the Demo

The demo sequences correspond to the companion demo paper workflow:

```bash
# Sequence 3: explicit + implicit request (cold start, signifier recording)
python tests/manual_test_full_flow_plan.py --sequence 3 --clear-signifiers

# Sequence 4: implicit request with prior signifier (fast path)
python tests/manual_test_full_flow_plan.py --sequence 4
```

Full demo guide with all sequences and environment variables: [docs/demo.md](docs/demo.md)

---

## Documentation

| Document | Contents |
|---|---|
| [docs/installation.md](docs/installation.md) | Full setup: Python env, HomeAssistant, HASP, SPADE |
| [docs/configuration.md](docs/configuration.md) | YAML config files, environment variables, multi-environment setup |
| [docs/evaluation.md](docs/evaluation.md) | Reproducing Tables 1 and 2 from the paper |
| [docs/demo.md](docs/demo.md) | Demo sequences (Seq. 2–4) with expected outputs |
| [docs/prompts.md](docs/prompts.md) | All LLM prompts: intent, purpose, paper reference, call sites |
| [docs/project-structure.md](docs/project-structure.md) | Full directory layout and data flow |
| [docs/testing.md](docs/testing.md) | Yggdrasil integration tests and HMAS ontology notes |

---

## Technologies

- **[SPADE](https://spade-mas.readthedocs.io/)** — Multi-agent system framework (XMPP)
- **[py_trees](https://py-trees.readthedocs.io/)** — BehaviorTree execution engine
- **[W3C WoT Thing Description](https://www.w3.org/TR/wot-thing-description/)** — Semantic device representation
- **[Yggdrasil](https://github.com/Interactions-HSG/yggdrasil)** — HMAS platform
- **[HomeAssistant](https://www.home-assistant.io/)** — Smart environment platform
- **[HMAS ontology](https://purl.org/hmas/)** — Hypermedia MAS vocabulary
- **[CASHMERE ontology](https://github.com/aimas-upb/cashmere)** — Signifier context vocabulary
- **OpenAI API** — LLM provider (GPT-4o, GPT-4o-mini, GPT-5-mini, GPT-5-nano tested)
