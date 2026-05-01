# Installation

This document covers the full setup required to run the AmI HMAS system end-to-end.

## Prerequisites

| Component | Version / Notes |
|---|---|
| Python | 3.9+ |
| SPADE XMPP server | Any Prosody/ejabberd instance, or `spade run` |
| Yggdrasil HMAS platform | Required for agent environment |
| HomeAssistant | Required for the HA integration pipeline |
| Docker | Recommended for HomeAssistant |
| OpenAI API key | For LLM calls (or OpenRouter-compatible base URL) |

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/aimas-upb/llm-agents-for-ami.git
cd llm-agents-for-ami
git checkout emas2026
```

## Step 2 — Python environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r agents-requirements.txt
```

## Step 3 — HomeAssistant and HASP

The integration pipeline described in Section 3.1 of the paper maps a HomeAssistant deployment into a W3C WoT / HMAS environment. Follow these steps to set it up.

### 3a. Run HomeAssistant (Docker)

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

Open HomeAssistant at `http://localhost:8123/` and complete onboarding.

### 3b. Install HACS and Virtual Components

1. Install HACS: see [hacs.xyz/docs/use/download](https://hacs.xyz/docs/use/download/download/#to-download-hacs)
2. In HomeAssistant → HACS, search **"Virtual Components"** and install.
3. Settings → Devices & Services → Add Integration → "Virtual Components".

### 3c. Load the Lab308 virtual device configuration

```bash
# Copy the sample virtual device config included in the repo
cp ami_agents/environment/integration/HomeAssistant/lab308.yaml \
   /path/to/your/config/custom_components/virtual/lab308.yaml
```

In HomeAssistant: Settings → Devices & Services → Virtual Components → **Add entry**:
- Group name: `lab308`
- File: `/config/custom_components/virtual/lab308.yaml`
- Create an area named `lab308` when prompted.

### 3d. Configure and start HASP

```bash
# Install HASP-specific dependencies
pip install -r ami_agents/environment/integration/HomeAssistant/requirements.txt

# Set environment variables
export HA_URL="ws://localhost:8123/api/websocket"
export HA_TOKEN="<your-long-lived-access-token>"   # HA Profile → Security → Create Token
export AREAS="lab308"
export BASE_WS_URI="http://localhost:8080"

# Start HASP (exposes the HMAS platform on port 8080)
cd ami_agents/environment/integration/HomeAssistant
uvicorn hasp:app --reload --port 8080 --log-level debug
```

HASP also exposes callback subscription endpoints:
- `POST /workspaces/{workspace_id}/focus`
- `POST /hub/`

See [ami_agents/environment/integration/HomeAssistant/README.md](../ami_agents/environment/integration/HomeAssistant/README.md) for payload examples and notification format.

### 3e. Optional utility scripts (Lab308 simulation)

```bash
# Continuously update the clock sensor
python ami_agents/environment/integration/HomeAssistant/tick_clock.py

# Automatically adjust luminosity sensor on light/blind changes
python ami_agents/environment/integration/HomeAssistant/adjust_lux_on_events.py

# Manually set a sensor value
python ami_agents/environment/integration/HomeAssistant/set-property.py lights_308 brightness_pct 75
```

## Step 4 — SPADE XMPP server

```bash
spade run
```

Or configure an external XMPP server (Prosody / ejabberd) and update `ami_agents/config/agents.yaml` with the server hostname and agent JIDs.

## Step 5 — Environment variables for the agent system

```bash
export OPENAI_API_KEY="<your-key>"

# Optional overrides (defaults shown)
export YGGDRASIL_URL="http://localhost:8080/"
export SPADE_SERVER="localhost"
export SPADE_PASSWORD="password"
export OPENAI_MODEL="o4-mini"
export OPENAI_REASONING_EFFORT="high"
```

## Step 6 — Run the demo

See [docs/demo.md](demo.md) for the full demo sequences corresponding to the paper's evaluation scenarios.

---

## Signifier Memory Engine (standalone service)

The Signifier Memory Engine can be run as a standalone FastAPI service for the community sharing experiments:

```bash
cd ami_agents/shared/memory
uvicorn src.api.main:app --port 8085
```

See [docs/configuration.md](configuration.md) for the `community.api_url` setting.

---

## Troubleshooting

**"Failed to dereference or validate Yggdrasil URL"**
Run `python tests/check_localhost.py` to diagnose the HMAS platform connection.

**LLM timeouts**
Increase `OPENAI_TIMEOUT` and/or `AMI_PLANNING_TIMEOUT` environment variables.

**XMPP auth/registration fails**
Enable in-band registration on your XMPP server, or pre-create agent accounts using the JIDs in `ami_agents/config/agents.yaml`.
