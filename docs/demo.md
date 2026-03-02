# Demo Sequences (CLI)

This project includes a thesis-demo friendly runner: `tests/manual_test_full_flow_plan.py`.

## Prerequisites

- Python 3.9+ with deps installed: `pip install -r requirements.txt`
- A running XMPP server (SPADE) with in-band registration enabled (or pre-created accounts)
- A running Yggdrasil instance (default: `http://localhost:8080/`)
- An OpenAI-compatible API key for the LLM calls (`OPENAI_API_KEY`)

## Environment variables (PowerShell)

```powershell
# required
$env:OPENAI_API_KEY = "<your key>"

# optional (defaults shown)
$env:YGGDRASIL_URL = "http://localhost:8080/"
$env:SPADE_SERVER  = "localhost"
$env:SPADE_PASSWORD = "password"

# LLM defaults used by the runner
$env:OPENAI_MODEL = "o4-mini"
$env:OPENAI_REASONING_EFFORT = "high"
```

Notes:
- The demo sequences use the Yggdrasil integration engine.

## Sequence 2 — Startup / Boot logs

Starts `EnvExplorer`, `UserAssistant`, and `InteractionSolver`, performs environment discovery, then exits (no user interactions).

```powershell
python tests\manual_test_full_flow_plan.py --sequence 2
```

Optional: keep agents alive until you press Enter:

```powershell
python tests\manual_test_full_flow_plan.py --sequence 2 --hold
```

## Sequence 3 — Interaction demo (EXPLICIT + IMPLICIT without prior signifier)

Demonstrates:
1) list active devices (UA → EnvExplorer)
2) query light state (UA ↔ EnvExplorer)
3) EXPLICIT goal (UA → Solver → EnvExplorer) + execution + signifier recording
4) IMPLICIT goal (no prior signifier): plan proposal + approval + execution + signifier recording

```powershell
python tests\manual_test_full_flow_plan.py --sequence 3 --clear-signifiers
```

Optional: pause right before the IMPLICIT request (useful for live demos):

```powershell
python tests\manual_test_full_flow_plan.py --sequence 3 --clear-signifiers --pause-for-sensor
```

## Sequence 4 — Reuse demo (IMPLICIT with prior signifier)

Requires a prior run of Sequence 3 (so the IMPLICIT signifier exists).

Sequence 4:
- simulates an external reset via Yggdrasil HTTP (light off, blinds mostly closed)
- repeats an IMPLICIT request with a PRIOR signifier
- expects the solver to recover a plan from signifiers without querying EnvExplorer for capabilities/state
- executes without recording a new signifier (reused signifier)

```powershell
python tests\manual_test_full_flow_plan.py --sequence 4
```

## Troubleshooting
- XMPP auth/registration fails: enable in-band registration on your XMPP server or pre-create the accounts used by the runner.
- LLM timeouts: increase `OPENAI_TIMEOUT` and/or `AMI_PLANNING_TIMEOUT` (seconds), then rerun.
