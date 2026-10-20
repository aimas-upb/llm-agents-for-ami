# Project Structure

How the repository is laid out, and where to extend it. For how a request moves through these
pieces, see [workflow.md](workflow.md).

## Directory layout

```
ami_agents/
├── main.py                          # Orchestrator: boots the three agents
├── config/
│   ├── agents.yaml                  # Agents, LLM per step, timeouts, BT execution (env-var substituted)
│   ├── environment.yaml             # Discovery, clock, integration engine, demo environments, community API
│   ├── planning_guidelines.yaml     # Device conditions per class / manufacturer (Matter-sourced)
│   └── services.yaml                # External services as virtual Things (examples, disabled)
├── agents/
│   ├── user_assistant/
│   │   ├── user_assistant_agent.py  # State holder + behaviour registration
│   │   ├── behaviours/              # user_message, user_request (FSM), *_structuring,
│   │   │                            # *_answer, plan_management, plan_execution
│   │   ├── prompts/                 # segmentation, parsing, goal structuring, responses
│   │   ├── utils/                   # goal_routing, goal_resolution, state/capability answers, llm_client
│   │   ├── pipeline.py              # Segmentation and plan-summary calls
│   │   ├── queries.py               # ENV_STATE / ENV_CAPABILITIES round trips
│   │   ├── registry.py              # Open requests, one per utterance
│   │   ├── signifiers.py, guards.py # Recording successful plans and guard decisions
│   │   └── models.py
│   ├── interaction_solver/
│   │   ├── interaction_solver_agent.py
│   │   ├── behaviours/              # goal_request (routing), explicit/incomplete/ambiguous/dependency
│   │   │                            # planning, condition_guard, power_on_guard, planning_workflow
│   │   ├── prompts.py               # MODIFY_RECIPE, GOAL_PROFILE, CONDITION_GUARD
│   │   └── utils/                   # per-path planners, dependency_compiler, guards, profiling,
│   │                                # time_consistency, signifier_memory, llm_client
│   └── env_explorer/
│       ├── env_explorer_agent.py
│       ├── behaviors/               # one behaviour per request type (capabilities, state, resolve,
│       │                            # snapshot, signifier and guard match / record)
│       ├── experience/              # Memory-engine bootstrap
│       └── utils/                   # state_resolution, capability_resolution, actuation_context
├── bt_planning/
│   ├── planning/                    # AsyncBTPlanner, BT JSON IR schema, planner prompt
│   ├── execution/ir_executor.py     # Compiles the IR to py_trees and ticks it
│   └── nodes/                       # Affordance, compute and dependency nodes; registry.py
├── environment/
│   ├── connection/hmas_client.py
│   ├── discovery/                   # direct / well-known / mDNS discovery
│   └── integration/
│       ├── integration_engine.py    # YggdrasilIntegration: crawls an HMAS platform
│       ├── HomeAssistant/           # HASP: Home Assistant → HMAS (+ semantic_mappings.yaml, home configs)
│       └── SimuHome/                # SHTD: SimuHome → TDs; worker, inspector, notifications
│           └── matter_model/        # Vendored CSA Matter data model (spec 1.5.1)
└── shared/
    ├── models/                      # messages (MessageType), goal_structure, intents, plan, environment
    ├── protocols/                   # IAgent, ILLMProvider and other interfaces
    ├── memory/src/                  # models/signifier, storage/{memory_store,guard_store},
    │                                # validation (SHACL), api (FastAPI community store)
    ├── community/community_client.py
    ├── ontologies/                  # homeont.ttl, saref.rdf, hmas, td, hctl, http
    └── utils/                       # goal_signifiers, clock, spade_rpc, config_loader, logger, demo_log

tests/
├── unit/                            # No services needed (`pytest tests/unit`)
├── integration/                     # Live LLM (needs OPENAI_API_KEY)
├── simuhome/
│   ├── bench/                       # The evaluation suite (see evaluation-simuhome.md)
│   ├── td/                          # Matter → SAREF / SOSA / TD-SOSA mapping tables for SHTD
│   └── *.py                         # Component evaluations (segmentation, structuring, dependency plans)
├── e2e-lab308e/                     # JSON-driven end-to-end cases against HASP (lab308e)
└── manual_interactive_cli.py        # Console chat with the three agents
```

## Layers

- **Agents.** Each agent class only holds state and registers behaviours. Every task is a
  behaviour: one `CyclicBehaviour` per inbound message type, and `OneShotBehaviour`s for discrete
  sub-tasks. Pure helpers live in `utils/` and do not import SPADE.
- **Planning and execution.** Plans are Behavior Trees in a JSON IR, which keeps them
  serialisable between agents. The executor compiles them to `py_trees` and ticks them.
  Leaves invoke TD affordances over HTTP.
- **Environment.** The agents see only an HMAS platform of TDs. HASP and SHTD produce one from
  Home Assistant and from SimuHome. Both type devices with HomeOnt / SAREF and annotate
  effects with TD-SOSA, so the agents resolve "what can make this room brighter" semantically.
- **Shared.** Message types, the goal structure, the experience memory, ontologies and utilities.

## Extension points

| To add… | Do this |
|---|---|
| a BT node type | Define the `py_trees` node in `bt_planning/nodes/`, then `register_node_type(...)` in `nodes/registry.py`. Custom computation goes through a named op in `compute_node.py`, never code in the IR. |
| an inter-agent message | Add it to `MessageType` in `shared/models/messages.py`, plus a behaviour on the receiving agent. |
| an LLM step in an agent | Copy the `BTPlanGenerationBehaviour` pattern. Build the client with the agent's `utils/llm_client.py:build_llm_client`. Put the model under that agent's `llm:` / `planning:` section in `agents.yaml`. |
| a device condition | Add an entry under `conditions:` in `config/planning_guidelines.yaml`, sourced from the Matter (or Home Assistant) specification. |
| an environment | Serve it as an HMAS platform of TDs (HASP and SHTD are the two examples) and point `YGGDRASIL_URL` at it. |
| a benchmark | See [evaluation-simuhome.md § Adding another benchmark](evaluation-simuhome.md#adding-another-benchmark). |
