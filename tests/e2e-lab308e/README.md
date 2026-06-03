# lab308e E2E Runner

This folder contains a small JSON-driven end-to-end harness for the `lab308e` Home Assistant scenario.

## Goal

Each test case is one JSON file that declares:

- `initial_state`: HTTP steps to drive HASP into a known setup
- `query`: the user request sent through `UserAssistant`
- `clarification_replies`: optional scripted answers to assistant clarification questions
- `pass_criteria`: assertions against HASP state or assistant replies

The runner walks the folder, executes every JSON case, and reports pass/fail.
It also appends cumulative per-case statistics to `tests/e2e-lab308e/results.csv`.

## Prerequisites

You need the same runtime pieces as `tests/manual_interactive_cli.py`:

- a running XMPP server on `localhost:5222` unless overridden
- `OPENAI_API_KEY`
- a running HASP/Yggdrasil endpoint, typically `http://localhost:8080/`
- the `lab308e` workspace already imported into Home Assistant and exposed by HASP

Useful env vars:

- `YGGDRASIL_URL`
- `HA_TOKEN`
  - used for direct Home Assistant `/api/states/...` setup steps; if unset, the runner loads the lab308e default from `ami_agents/environment/integration/HomeAssistant/prepare-adapter-env.sh`
- `SPADE_SERVER`
- `SPADE_PASSWORD`
- `OPENAI_MODEL`
  - default if unset: `gpt-5-mini`
- `OPENAI_BASE_URL`
- `OPENAI_REASONING_EFFORT`
- `BT_MAX_TICKS_USER_ASSISTANT`
  - default in this runner: `240`, so `wait_condition` nodes can poll across settling-time windows

## Run

From the repo root:

```bash
PYTHONPATH=ami_agents/shared/memory ~/aiml/env-spade-3/bin/python tests/e2e-lab308e/run_cases.py
```

The default runner response timeout is 900 seconds because some plans verify
settling-time effects with `wait_condition` polling.

Run one case:

```bash
PYTHONPATH=ami_agents/shared/memory ~/aiml/env-spade-3/bin/python \
  tests/e2e-lab308e/run_cases.py \
  --case tests/e2e-lab308e/cases/reduce_glare.json
```

Run one case by its JSON `name`:

```bash
PYTHONPATH=ami_agents/shared/memory ~/aiml/env-spade-3/bin/python \
  tests/e2e-lab308e/run_cases.py \
  --case-name reduce_glare_closes_blackout_blinds
```

Clear signifiers once before the batch:

```bash
PYTHONPATH=ami_agents/shared/memory ~/aiml/env-spade-3/bin/python \
  tests/e2e-lab308e/run_cases.py \
  --clear-signifiers
```

Clear signifiers and per-case in-memory assistant state before every case:

```bash
PYTHONPATH=ami_agents/shared/memory ~/aiml/env-spade-3/bin/python \
  tests/e2e-lab308e/run_cases.py \
  --clear-signifiers-per-case
```

Write per-case result JSON files:

```bash
PYTHONPATH=ami_agents/shared/memory ~/aiml/env-spade-3/bin/python \
  tests/e2e-lab308e/run_cases.py \
  --write-results
```

Dump all prompts sent to the LLM, one `.txt` file per case:

```bash
PYTHONPATH=ami_agents/shared/memory ~/aiml/env-spade-3/bin/python \
  tests/e2e-lab308e/run_cases.py \
  --dump-prompts
```

## Case Format

Minimal example:

```json
{
  "name": "reduce_glare_closes_blackout_blinds",
  "initial_state": [
    {
      "type": "request",
      "method": "POST",
      "path": "/workspaces/lab308e/artifacts/blackout_blinds_308e_cover/ha/cover/open_cover",
      "json": {},
      "expect_status": 200
    },
    {
      "type": "sleep",
      "seconds": 2
    }
  ],
  "query": "reduce glare",
  "clarification_replies": [
    {
      "match": "Do you want me to reduce glare by",
      "reply": "lower the blackout blinds"
    }
  ],
  "auto_confirm": true,
  "confirmation_text": "yes",
  "settle_seconds": 2,
  "pass_criteria": [
    {
      "type": "property_equals",
      "path": "/workspaces/lab308e/artifacts/blackout_blinds_308e_cover/properties/state",
      "equals": "closed",
      "timeout_seconds": 30
    }
  ]
}
```

`clarification_replies` entries support:

- `match`
- optional `match_type`: `contains` (default), `exact`, or `regex`
- `reply`

If an assistant reply matches one of these rules, the runner sends the scripted
`reply` back to the assistant before normal auto-confirmation. This is useful
for disambiguation questions such as screen selection or choosing between
multiple anti-glare strategies.

## Supported `initial_state` Steps

- `request`
  - fields: `method`, `path` or `url`, optional `headers`, optional `json`, optional `expect_status`
- `sleep`
  - fields: `seconds`

Relative `path` values are resolved against `YGGDRASIL_URL`.
String values inside `url`, `path`, `headers`, and `json` can reference
environment variables with `${VAR_NAME}` placeholders, for example
`"Authorization": "Bearer ${HA_TOKEN}"`.

## Supported `pass_criteria`

- `property_equals`
- `property_gte`
- `property_lte`
- `property_in`
- `assistant_reply_contains`
- `final_reply_contains`
- `any_of`
- `all_of`

Property assertions poll until success or `timeout_seconds`.

## Results CSV

Each run appends one row per case to `results.csv` with:

- `test_name`
- `timestamp`
- `model_name`
- `passed`
- `duration_seconds`
- `llm_calls`
- `input_tokens`
- `output_tokens`

Per-case JSON results written with `--write-results` also include:

- `first_reply`
- `final_reply` when auto-confirmation is enabled

## Prompt Dumps

When `--dump-prompts` is enabled, the runner writes one text file per case to:

- `tests/e2e-lab308e/prompt_dumps/`

Each file contains the LLM calls in order, including:

- model name
- reasoning effort when present
- response format when present
- full `messages` payload content sent through the OpenAI chat-completions wrapper

## Notes

- The runner starts `EnvExplorer`, `InteractionSolver`, and `UserAssistant` once for the whole batch.
- Cases should therefore be written to establish their own initial state explicitly.
- If you want a clean signifier store before the batch, use `--clear-signifiers`.
- If you want isolated signifier memory and no carried-over pending plans/conversation thread
  state for every JSON case, use `--clear-signifiers-per-case`.
