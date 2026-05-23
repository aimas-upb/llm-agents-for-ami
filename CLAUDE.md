# CLAUDE.md — AI-coding guidance for this repository

This file is loaded automatically by Claude Code (and other Claude-based
agents) when opening this repo. It is also the source of truth that
`.github/copilot-instructions.md` mirrors for GitHub Copilot. Keep both
files in sync.

The audience is an AI coding agent (Claude / Copilot / Cursor / Cody)
about to make a change in this codebase. The goal is for that agent to
match the conventions the human team has already settled on.

---

## 1. What this project is

**AmI HMAS** — Hybrid agents (deterministic state machines + LLM injection
points) that drive smart-home environments expressed as W3C WoT / HMAS
hypermedia. Three SPADE agents (XMPP):

| Agent | Role | LLM use |
|---|---|---|
| `UserAssistant` | NL <-> intents, plan summarisation, plan confirmation, plan execution, signifier recording | NLU + NLG |
| `InteractionSolver` | Receives goal requests; gathers context from EnvExplorer; produces a Behavior-Tree JSON IR plan | LLM-backed BT planner (`AsyncBTPlanner`) |
| `EnvExplorer` | Discovers HMAS environment, exposes capabilities/state, hosts the Signifier (Experience) Engine | None |

Plans are Behavior Trees encoded as JSON IR; executed by `IRExecutor` via the
Yggdrasil HMAS integration engine.

The project targets the **EMAS / AAMAS 2026 demo** (deadline 2026-05-15).
See `paper/` and `README.md` for paper context.

## 2. Repo layout (canonical)

```
ami_agents/
├── main.py                       # Orchestrator — boots all 3 agents
├── config/
│   ├── agents.yaml               # All agent + LLM + timeout config (env-var substituted)
│   └── environment.yaml          # Yggdrasil/HA URLs, demo environments
├── agents/
│   ├── user_assistant/
│   │   ├── user_assistant_agent.py   # Slim Agent class — state + behaviour registration
│   │   ├── behaviours/               # SPADE Behaviours (one per inbound message type / task)
│   │   ├── utils/                    # Pure helpers (json_io, plan, llm_client)
│   │   ├── models.py                 # Intent, ConversationState, ConversationPhase
│   │   └── prompts.py                # System prompts (NLU / NLG)
│   ├── env_explorer/
│   │   ├── env_explorer_agent.py     # Slim Agent class
│   │   ├── behaviors/                # One Behaviour per request type (cap, state, sig_match, ...)
│   │   ├── utils/                    # Pure helpers
│   │   └── experience/               # Experience-Engine bootstrap + matching helpers
│   └── interaction_solver/
│       ├── interaction_solver_agent.py  # Slim Agent class — RPC wrapper + state
│       ├── behaviours/               # GoalRequest, EnvironmentReady, PlanningWorkflow,
│       │                             # EnvContextQuery, SignifierMatchQuery,
│       │                             # CommunitySignifierQuery, BTPlanGeneration
│       └── utils/                    # plan_envelope, signifier_fast_path, llm_client, ...
├── bt_planning/
│   ├── planning/bt_planner.py    # AsyncBTPlanner — the LLM-backed BT generator
│   ├── execution/ir_executor.py  # Tick-based BT executor
│   ├── nodes/                    # BT node implementations (HTTP, action, etc.)
│   └── signifier_bridge.py       # Build BT from stored signifiers (fast-path)
├── environment/
│   ├── connection/hmas_client.py
│   └── integration/integration_engine.py   # YggdrasilIntegration
└── shared/
    ├── models/                  # Message, plan, environment models
    ├── protocols/               # IAgent, ILLMService
    ├── memory/                  # Signifier registry + matchers (v0/v1/v2)
    ├── community/               # Cross-environment signifier-sharing client
    └── utils/                   # logger, spade_rpc, demo_log, config_resolver
```

## 3. Conventions you MUST follow

### 3.1 Agent package shape

Every agent lives in `ami_agents/agents/<name>/` with this structure:

```
agent.py            # ~150 LoC. ONLY: state, lifecycle, send_message, RPC wrappers.
behaviours/         # One Behaviour per inbound message type; OneShot sub-behaviours
                    # for discrete tasks ("querying X is a behaviour").
utils/              # Pure helpers — no SPADE imports here, just functions/dataclasses.
models.py           # Dataclasses / enums for the agent's domain (intents, state, ...).
prompts.py          # LLM system prompts (string constants only).
```

The agent class is a **state holder + behaviour registrar**. It does NOT
contain orchestration logic. If you find a method on the agent class that
runs a multi-step workflow, hoist it into a Behaviour (most likely a
`OneShotBehaviour` spawned from a parent `CyclicBehaviour`).

### 3.2 Behaviour decomposition rule

Project-wide guidance from the team lead (Alex Sorici):

> All "todos" an agent has must be encapsulated in Behaviours, especially
> when handling one request is independent of another.

Concretely:
- One CyclicBehaviour per inbound SPADE message type, gated by a `Template`.
- For multi-step request handling, the entry behaviour parses the request
  and spawns a `PlanningWorkflowBehaviour` (or similar) per request.
- Each discrete sub-task (querying EnvExplorer, asking the community,
  generating a plan) is its own `OneShotBehaviour`. Spawn with
  `agent.add_behaviour(...)`, await with `await behaviour.join()`.
- Do NOT add helper methods like `_generate_plan` or `_gather_context`
  on the agent class — those belong in a behaviour or a utils module.

### 3.3 Configuration

All agent/LLM/timeout config lives in `ami_agents/config/agents.yaml`.
Rules:
- Only add a yaml key if Python code actually reads it. Dead config
  rots.
- Every key supports env-var substitution: `"${VAR:-default}"`.
- Read with `config.get("section", {}).get("subkey", DEFAULT)`. Define
  the `DEFAULT` as a module-level constant so unit tests stay self-contained.
- Never hardcode timeouts/URLs/models inside agents — push them to yaml.

### 3.4 Logging

Use `LoggerFactory.get_logger(name, logging_config)` from
`ami_agents.shared.utils.logger`. Each agent constructs its logger in
`__init__` from the merged logging config that `main.py` passes in.

For demo-friendly lines, wrap the format string with
`from ami_agents.shared.utils.demo_log import demo` — these are flagged
in the live demo log filter.

Do not use `print()` inside `ami_agents/`. (Tests under `tests/` may
keep `print()` for now.)

### 3.5 Inter-agent messages

All inter-agent traffic uses the `MessageType` enum from
`ami_agents.shared.models.messages`. Send replies that reuse the
correlation ID (`META_CORRELATION_ID`) and thread of the original
message — `rpc_call` / `rpc_call_response` in
`ami_agents.shared.utils.spade_rpc` handle this.

Request-type taxonomy (also listed in `agents.yaml > request_types`):
`ENV_CAPABILITIES, ENV_STATE, GOAL_REQUEST, PLAN_MANAGEMENT, PREFERENCE_MANAGEMENT`.

### 3.6 LLM clients

There are two LLM clients (UA and ISA), both built by their respective
`utils/llm_client.py:build_llm_client(config)` returning an
`LLMClientConfig` dataclass. Both honour:
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env vars
- Reasoning models (model name starting with `o`) — strip `temperature`,
  use `max_completion_tokens` instead of `max_tokens`, default
  `reasoning_effort=high`, longer `timeout`.
- The ISA's variant additionally honours `planning.llm_planning.*` overrides.

If you add a new agent that needs an LLM, copy this pattern; do NOT
re-implement client construction inline.

### 3.7 Intent semantics (EXPLICIT / IMPLICIT)

The user message is classified as EXPLICIT (user named exact artifact
IDs) or IMPLICIT (inferred from environment context). The classification
is threaded:

```
UA NLU prompt         -> Intent + intent_type
UA -> ISA GOAL_REQUEST {intents, intent_type, workspace_id}
ISA -> EnvExplorer SIGNIFIER_MATCH_REQUEST {intent, intent_type, query_structured_intent}
EnvExplorer signifier matcher v2 -> EXPLICIT skips context validation;
                                     IMPLICIT enforces context validation
BT planner format_signifier_hints -> labels matches >= 0.95 as "exact"
```

If you touch any link in this chain, preserve the `intent_type` and
`structured_intent` payload — do not silently drop them.

## 4. Workflows

### 4.1 Environment

**Always run Python via the `ami` conda env:**

```bash
conda run -n ami-agents python ...
conda run -n ami-agents pip install ...
conda run -n ami-agents pytest ...
```

If `conda` is not on PATH, the user's repo-local `ami-agents` env exists at
`~/miniconda3/envs/ami-agents` — do not pip-install into the system Python.

### 4.2 Tests

```bash
conda run -n ami-agents pytest tests/unit -q          # ~165 tests, ~1s
conda run -n ami-agents pytest tests/integration -q   # needs OPENAI_API_KEY
conda run -n ami-agents pytest tests/e2e -q           # needs full SPADE/HA stack
```

Target: **all unit tests must stay green** before opening a PR.
Integration / e2e require live services (XMPP server, HomeAssistant,
Yggdrasil, OPENAI_API_KEY).

### 4.3 Live demo (full stack)

Bring-up sequence is documented in `docs/installation.md` and
`docs/demo.md`. The local-dev XMPP server is **Prosody** (NOT
pyjabber — pyjabber's STARTTLS handshake is broken with
slixmpp). See `~/prosody-config/prosody.cfg.lua` for the dev config.

Sequenced demo runs:
```bash
conda run -n ami-agents python tests/manual_test_full_flow_plan.py --sequence 3 --clear-signifiers
```

### 4.4 Branching / commits

- Default branch: `master`. Feature branches: `feature/<topic>`.
- Demo / paper branches: `emas2026`, `aamas2026demo` — read-only for
  most work; PR into `master` then cherry-pick if needed.
- Commit messages: imperative mood, `Refactor X into ...` /
  `Fix Y in ...`.
- **Do NOT add `Co-Authored-By: Claude` or any AI-attribution trailer**
  to commits in this repo unless the human author explicitly asks.
- Squash-merge convention for PRs (the repo's PR template).

## 5. Things to never do

1. **Never bypass the conda env.** `python` / `pip` outside `conda run -n ami-agents`
   will install into the wrong place.
2. **Never add SPADE behaviour logic as helper methods on the Agent class.**
   See §3.1 — hoist into `behaviours/`.
3. **Never hardcode model / timeout / URL inside an agent.** Push to
   `config/agents.yaml` (with env-var substitution).
4. **Never add yaml config keys that no Python code reads.** Dead
   config misleads. Audit with
   `grep -rn "<key>" ami_agents --include="*.py"` before adding.
5. **Never use `--no-verify` / `--no-gpg-sign` on commits** unless the
   human asks.
6. **Never force-push to `master`, `emas2026`, or `aamas2026demo`.**
7. **Never call `print()` inside `ami_agents/`.** Use the logger.
8. **Never drop `intent_type` / `query_structured_intent` from the
   GOAL_REQUEST → SIGNIFIER_MATCH_REQUEST chain.** It breaks the
   EMAS demo's semantic-modelling story (§3.7).

## 6. Where to look first when …

| Task | Start at |
|---|---|
| Add a new inter-agent message type | `ami_agents/shared/models/messages.py` enum, then add a Behaviour for it on the receiving agent |
| Add a new LLM-driven step | Copy the `BTPlanGenerationBehaviour` pattern (`OneShotBehaviour`, takes the LLM client/config from `self.agent`) |
| Tweak NLU/NLG prompts | `ami_agents/agents/user_assistant/prompts.py` |
| Tweak BT planner prompts | `ami_agents/bt_planning/planning/prompts.py` |
| Change a timeout | `ami_agents/config/agents.yaml > timeouts:` |
| Add a config-driven feature | yaml key + `LLMClientConfig`-style dataclass in `utils/` |
| Add a new BT node type | `ami_agents/bt_planning/nodes/` + register in `signifier_bridge.py` if needed |
| Wire a new HA entity into the demo | `ami_agents/environment/integration/HomeAssistant/` + `config/environment.yaml` |
| Run a single sequence end-to-end | `tests/manual_test_full_flow_plan.py --sequence <N>` |
