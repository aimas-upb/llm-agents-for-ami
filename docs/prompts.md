# LLM Prompt Reference

This document provides the complete reference for all LLM prompts used in AmI HMAS.
The paper states (Section 3.3, footnote): *"Full system prompts are available in the source code repository."*

AmI HMAS follows a **controlled injection point** principle: LLM calls occur only at explicitly designed
locations, never inside the deterministic agent control flows. The three agents have the following LLM usage:

| Agent | LLM calls | Prompt file |
|---|---|---|
| **UserAssistant** | NLU (intent extraction), NLG (plan summary), NLG (query response) | [`ami_agents/agents/user_assistant/prompts.py`](../ami_agents/agents/user_assistant/prompts.py) |
| **InteractionSolver / BT Planner** | BT generation (structured JSON IR) | [`ami_agents/bt_planning/planning/prompts.py`](../ami_agents/bt_planning/planning/prompts.py) |
| **EnvExplorer** | None — classical SPADE agent | — |

---

## UserAssistant Prompts

File: [`ami_agents/agents/user_assistant/prompts.py`](../ami_agents/agents/user_assistant/prompts.py)
Called from: [`ami_agents/agents/user_assistant/behaviours.py`](../ami_agents/agents/user_assistant/behaviours.py)

### 1. `INTENT_EXTRACTION_SYSTEM_PROMPT` — NLU

**When called:** Every time the user sends a message. The UserAssistant enters the `EXTRACTING_INTENTS`
state of its lifecycle state machine (Figure 1 in paper, Section 3.2).

**What it does:** Classifies the user message into one of five categories and, for `goal` messages,
extracts one or more **structured intents** (the core NLU step described in Section 3.2):
- `action`: one of `check`, `set`, `modify`
- `artifact`: exact device ID or generic type (determines `explicit` vs. `implicit` intent type)
- `parameter`, `value`, `intent_text`
- `intent_type`: `explicit` (user names a specific artifact) or `implicit` (vague reference)

**Message categories:**
| Category | Meaning |
|---|---|
| `goal` | User wants to change something → structured intents extracted |
| `query_capabilities` | User asks about available devices/actions |
| `query_state` | User asks about current device state |
| `confirmation` | User confirms or rejects a proposed plan |
| `unclear` | Ambiguous message → agent asks one clarifying question |

**Output:** JSON only — no markdown fences. See the prompt for full output schema with examples.

**Paper connection:** The explicit/implicit classification drives Algorithm 1 (dual-path planning) —
explicit `set` intents can hit the fast path; implicit intents always go through LLM planning.

---

### 2. `PLAN_SUMMARY_SYSTEM_PROMPT` — NLG (plan presentation)

**When called:** After the InteractionSolver returns a BT JSON IR plan. The UserAssistant enters the
`SUMMARIZING_PLAN` state (Figure 1 in paper), calls this prompt, and presents the result to the user
for confirmation before execution.

**What it does:** Converts a BT JSON IR (machine-readable) into a concise 2–4 sentence human-readable
description, ending with "Does this plan look good to you?". No raw URIs, no JSON, no technical jargon.

**Paper connection:** Section 3.3 — "the UserAssistant uses [the JSON IR] as input to an LLM call
for human-readable plan summarization to be delivered for confirmation to the user."

---

### 3. `QUERY_RESPONSE_SYSTEM_PROMPT` — NLG (state and capability responses)

**When called:** When the user's message is classified as `query_capabilities` or `query_state`.
The UserAssistant fetches the raw data from EnvExplorer and passes it to this prompt.

**What it does:** Formats raw JSON environment state or capability data into a concise,
user-friendly natural language response. No raw URIs or JSON dumps.

---

## InteractionSolver / BT Planner Prompts

File: [`ami_agents/bt_planning/planning/prompts.py`](../ami_agents/bt_planning/planning/prompts.py)
Called from: [`ami_agents/agents/interaction_solver/interaction_solver_agent.py`](../ami_agents/agents/interaction_solver/interaction_solver_agent.py)
via [`ami_agents/bt_planning/planning/bt_planner.py`](../ami_agents/bt_planning/planning/bt_planner.py)

### 4. `BT_PLANNING_SYSTEM_PROMPT` — BT generation

**When called:** LLM planning path of Algorithm 1 (lines 16–17). Called by `AsyncBTPlanner.generate_bt()`
for any `implicit` intent or any `modify` action. Not called for `explicit set` intents (fast path).

**What it does:** Instructs the LLM to produce a **BT JSON IR** (intermediate representation) using
OpenAI function calling / tool use. The LLM must use only the node types from Table 1 of the paper
(`sequence`, `selector`, `parallel`, `action`, `condition`) and only the affordance URLs provided
in the dynamic context.

**Template variables (injected at call time):**
- `{signifier_hints}` — formatted signifier matches from EnvExplorer or community (see below)
- `{capability_context}` — available affordances + current environment state

**BT node types instructed** (matches Table 1 of paper):
| Type | Behavior |
|---|---|
| `sequence` | Left-to-right; fail on first failure |
| `selector` | Left-to-right; succeed on first success (used for idempotent patterns) |
| `parallel` | All children concurrently; `success_on_all` or `success_on_one` |
| `action` | HTTP POST to `action_url` |
| `condition` | HTTP GET `property_url`; compare to `expected_value` |

**Common patterns instructed:**
- Idempotent action: `selector → [condition(check state), action(set state)]`
- Sequential commands: `sequence → [action1, action2, ...]`
- Independent commands: `parallel(success_on_all) → [action1, action2, ...]`

**Retry logic:** Up to `max_attempts` (default 3) retries with validation feedback if the generated
JSON fails schema validation. Implemented in `AsyncBTPlanner` (`bt_planner.py`).

---

### 5. `format_signifier_hints()` — dynamic signifier context formatter

**Type:** Helper function (not a standalone prompt), but generates the `{signifier_hints}` section
injected into `BT_PLANNING_SYSTEM_PROMPT`.

**What it does:** Formats the signifier matches returned by EnvExplorer (or community) into a
structured hints section. Matches above `EXPLICIT_SIMILARITY_THRESHOLD = 0.95` are labelled
**"exact — reuse as-is"**; lower-similarity matches are labelled **"suggested — adjust to context"**.

**Paper connection:** Section 4.2 — "signifier matches are injected as hints into the LLM planning
context." This function implements that injection.

---

### 6. `format_capability_context()` — dynamic affordance + state formatter

**Type:** Helper function, generates the `{capability_context}` section of `BT_PLANNING_SYSTEM_PROMPT`.

**What it does:** Formats the list of available `ActionAffordance` and `PropertyAffordance` objects
(URLs, methods, input schemas) plus the current environment state snapshot into a structured
markdown section for the LLM context window.

**Paper connection:** Section 3.4 — "the gathered context and the structured intent information
constitute the input for LLM-based reasoning to generate the BT plan."

---

## EnvExplorer — No LLM Prompts

The **EnvExplorer** is a **classical SPADE agent** (no LLM calls). Its responsibilities are:
- W3C WoT discovery via RDF/HTTP crawling of the HMAS platform
- Maintaining an up-to-date artifact state map via WebSub push events
- Managing the Signifier Memory Engine (store, retrieve, match)

Signifier matching uses sentence embeddings (cosine similarity) and SHACL validation — no LLM.
See [`ami_agents/shared/memory/src/matching/`](../ami_agents/shared/memory/src/matching/) for
the matching implementations and [`ami_agents/shared/memory/src/validation/`](../ami_agents/shared/memory/src/validation/)
for SHACL context validation.

---

## Summary — Call Sites

| Prompt / Function | Call site | State machine phase |
|---|---|---|
| `INTENT_EXTRACTION_SYSTEM_PROMPT` | `behaviours.py:116` | `EXTRACTING_INTENTS` |
| `PLAN_SUMMARY_SYSTEM_PROMPT` | `behaviours.py:142` | `SUMMARIZING_PLAN` |
| `QUERY_RESPONSE_SYSTEM_PROMPT` | `behaviours.py:165` | `IDLE` (query response) |
| `BT_PLANNING_SYSTEM_PROMPT` | `interaction_solver_agent.py:372` via `bt_planner.py` | LLM planning path |
| `format_signifier_hints()` | `prompts.py` (called by `bt_planner.py`) | Context assembly |
| `format_capability_context()` | `prompts.py` (called by `bt_planner.py`) | Context assembly |
