# Installation

This document covers the setup needed to run AmI HMAS against a home: a Home Assistant
deployment through HASP, or a SimuHome episode through SHTD. The SimuHome evaluation suite
needs only Steps 1, 2 and 5; it runs without XMPP.

## Prerequisites

| Component | Version / Notes |
|---|---|
| Conda | Python 3.12 env named `ami-agents` |
| XMPP server | Prosody (recommended), with in-band registration or pre-created accounts |
| An HMAS environment | HASP over Home Assistant, or SHTD over the SimuHome simulator |
| Home Assistant + Docker | Only for the HASP path |
| OpenAI API key | Or another OpenAI-compatible endpoint |

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/aimas-upb/llm-agents-for-ami.git
cd llm-agents-for-ami
git checkout ami-hmas-semantic
```

## Step 2 — Python environment

```bash
conda create -n ami-agents python=3.12 -y
conda run -n ami-agents pip install -r agents-requirements.txt
```

Run every Python command through `conda run -n ami-agents …`. A plain `venv` works too, if you
install the same requirements into it.

Put your key in a `.env` file at the repo root, or export it:

```bash
export OPENAI_API_KEY="<your-key>"
```

## Step 3 — Home Assistant and HASP (real homes)

HASP maps a Home Assistant deployment into a W3C WoT / HMAS environment. Areas become workspaces
and devices become artifacts. TDs are typed with HomeOnt / SAREF and annotated with TD-SOSA
effects, and state changes are pushed through WebSub.

### 3a. Run Home Assistant (Docker)

```bash
docker run -d \
  --name homeassistant \
  --privileged \
  --restart=unless-stopped \
  -e TZ=Europe/Bucharest \
  -v /path/to/your/config:/config \
  -v /run/dbus:/run/dbus:ro \
  --network=host \
  ghcr.io/home-assistant/home-assistant:stable
```

Open Home Assistant at `http://localhost:8123/` and complete onboarding.

### 3b. Install HACS and Virtual Components

1. Install HACS: see [hacs.xyz/docs/use/download](https://hacs.xyz/docs/use/download/download/#to-download-hacs)
2. In Home Assistant → HACS, search **"Virtual Components"** and install.
3. Settings → Devices & Services → Add Integration → "Virtual Components".

### 3c. Load a virtual home

The repo ships virtual homes in `ami_agents/environment/integration/HomeAssistant/`:
`lab308.yaml`, `lab308e.yaml` (a denser variant), and `home1.yaml`–`home3.yaml`. Each also
carries the `semantic:` block HASP uses for typing.

```bash
cp ami_agents/environment/integration/HomeAssistant/lab308e.yaml \
   /path/to/your/config/custom_components/virtual/lab308e.yaml
```

In Home Assistant: Settings → Devices & Services → Virtual Components → **Add entry**:
- Group name: `lab308e`
- File: `/config/custom_components/virtual/lab308e.yaml`
- Create an area named `lab308e` when prompted.

### 3d. Configure and start HASP

```bash
conda run -n ami-agents pip install -r ami_agents/environment/integration/HomeAssistant/requirements.txt

export HA_URL="ws://localhost:8123/api/websocket"
export HA_TOKEN="<your-long-lived-access-token>"   # HA Profile → Security → Create Token
export AREAS="lab308e"                             # the area_id(s) to expose
export BASE_WS_URI="http://localhost:8080"
export SEMANTIC_CONFIG="lab308e.yaml"              # the home's semantic typing

cd ami_agents/environment/integration/HomeAssistant
conda run -n ami-agents uvicorn hasp:app --port 8080 --log-level debug
```

The agents then use `YGGDRASIL_URL=http://localhost:8080/`.

For subscription endpoints, payload examples and the lab308e simulator loop
(`simulate_lab308e.py`), see
[ami_agents/environment/integration/HomeAssistant/README.md](../ami_agents/environment/integration/HomeAssistant/README.md).

## Step 4 — XMPP server

Use **Prosody**. `spade run` starts pyjabber, whose STARTTLS handshake fails with slixmpp.

```bash
sudo apt install prosody
```

In `prosody.cfg.lua`:
- declare `VirtualHost "localhost"`;
- either set `allow_registration = true`, or pre-create the agent accounts with
  `prosodyctl register <user> localhost <password>`.

The agent accounts are `user_assistant`, `env_explorer` and `interaction_solver`. The console runner
also uses `console_cli`. The runners take the host from `SPADE_SERVER` (default `localhost`) and one
password for all accounts from `SPADE_PASSWORD` (default `password`).

## Step 5 — SimuHome and SHTD (Matter homes, benchmark)

SHTD serves a home from the [SimuHome](https://github.com/holi-lab/SimuHome) simulator as Thing
Descriptions built directly from the Matter data model. Home Assistant is not involved.

### 5a. Clone SimuHome next to this repo

The bench and SHTD tooling look for SimuHome at `../SimuHome`. No flag or variable changes that
location; only the benchmark directory can be overridden, with `--benchmark`.

```bash
cd ..
git clone https://github.com/holi-lab/SimuHome.git     # → ../SimuHome, episodes in data/benchmark/
cd llm-agents-for-ami
```

The simulator and SimuHome's evaluators run under the `ami-agents` interpreter, so install
SimuHome's dependencies (from its `pyproject.toml`) there:

```bash
conda run -n ami-agents pip install fastapi uvicorn pydantic python-dotenv pyyaml requests \
    openai langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu
```

### 5b. The Matter data model

The CSA Matter data model (spec 1.5.1) is vendored in
`ami_agents/environment/integration/SimuHome/matter_model/vendor/1.5.1/`. Nothing needs to be
downloaded. To check it offline:

```bash
conda run -n ami-agents python ami_agents/environment/integration/SimuHome/matter_model/bootstrap.py --validate-only
```

### 5c. Run one home by hand

The evaluation suite starts its own servers. You only need these commands to talk to a home from
the console, or to look at its TDs. They are two processes, each in its own terminal, both run
from the repo root:

```bash
# 1. the simulator with the episode loaded (port 14000), plus the attribute inspector (port 8098)
conda run -n ami-agents python ami_agents/environment/integration/SimuHome/run_inspector.py \
    --episode qt2_feasible_seed_77 --sim-port 14000 --port 8098

# 2. SHTD, serving that home as TDs (port 8097)
conda run -n ami-agents python ami_agents/environment/integration/SimuHome/shtd.py \
    --sim http://127.0.0.1:14000/api --home qt2_feasible_seed_77 --port 8097
```

Then point the agents at SHTD:
- `YGGDRASIL_URL=http://127.0.0.1:8097`;
- for plans that wait on time, also `CLOCK_SOURCE=http CLOCK_URL=http://localhost:8097/_shtd/time`,
  so they run on the simulator's virtual time.

Set `SHTD_AMBIENT_SENSORS=0` to serve the home without the virtual ambient sensors. SHTD's routes,
worker model and units are documented in
[ami_agents/environment/integration/SimuHome/README.md](../ami_agents/environment/integration/SimuHome/README.md).

## Step 6 — Talk to the agents

With an XMPP server and one environment running:

```bash
YGGDRASIL_URL=http://127.0.0.1:8097 conda run -n ami-agents python tests/manual_interactive_cli.py
```

The script starts the three agents and opens a console prompt; type `exit` to stop. Options:
- `--demo` shows only the demo log lines;
- `--no-logs` shows only the dialogue;
- `--response-timeout` sets how long to wait for a reply;
- `--skip-prewarm` skips pre-warming the UserAssistant.

For scripted end-to-end cases against HASP's lab308e home, see
[tests/e2e-lab308e/README.md](../tests/e2e-lab308e/README.md).

## Step 7 — Evaluation

See [evaluation-simuhome.md](evaluation-simuhome.md).

---

## Community signifier store (optional)

Several homes can share remembered plans through the memory engine's FastAPI service:

```bash
cd ami_agents/shared/memory
conda run -n ami-agents uvicorn src.api.main:app --port 8085
```

Set `COMMUNITY_API_URL` (see [configuration.md](configuration.md)).

---

## Troubleshooting

**"Failed to dereference or validate Yggdrasil URL"**
The URL in `YGGDRASIL_URL` does not answer with an HMAS platform. Check that HASP or SHTD is
running on that port.

**LLM timeouts**
Increase `LLM_TIMEOUT_DEFAULT` / `LLM_TIMEOUT_REASONING` (agents) or `OPENAI_TIMEOUT` (console
runner).

**XMPP auth/registration fails**
Enable in-band registration on Prosody, or pre-create the agent accounts.

**SHTD cannot reach the simulator**
`run_inspector.py --sim-port` defaults to `0` (a free port). Pin it, as above, so SHTD's `--sim`
has a fixed address.
