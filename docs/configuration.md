# Configuration

AmI HMAS is configured through three YAML files in [`ami_agents/config/`](../ami_agents/config/) and environment variables.

## Configuration Files

| File | Purpose |
|---|---|
| [`agents.yaml`](../ami_agents/config/agents.yaml) | Agent JIDs, SPADE settings, LLM model/provider, signifier thresholds |
| [`environment.yaml`](../ami_agents/config/environment.yaml) | Environment discovery method, Yggdrasil/HomeAssistant connection |
| [`services.yaml`](../ami_agents/config/services.yaml) | External service definitions (calendar, weather, etc.) as virtual Things |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key (or OpenRouter key) |
| `YGGDRASIL_URL` | No | `http://localhost:8080/` | HMAS platform base URL |
| `SPADE_SERVER` | No | `localhost` | XMPP server hostname |
| `SPADE_PASSWORD` | No | `password` | XMPP agent password |
| `OPENAI_MODEL` | No | `o4-mini` | LLM model for planning |
| `OPENAI_REASONING_EFFORT` | No | `high` | Reasoning effort for `o*` models |
| `OPENAI_BASE_URL` | No | OpenAI default | Override for OpenRouter or local servers |
| `HA_URL` | HA only | — | HomeAssistant WebSocket URL |
| `HA_TOKEN` | HA only | — | HomeAssistant long-lived access token |
| `AREAS` | HA only | — | Comma-separated HA area IDs to map |
| `BASE_WS_URI` | HA only | — | Public base URI for the HA adapter |
| `COMMUNITY_API_URL` | No | — | Community signifier API URL (Sec. 5) |

---

## Environment Discovery (`environment.yaml`)

Three discovery methods are supported:

### Direct
Provide a known TD Directory URL:
```yaml
environment:
  discovery_method: 'direct'
  direct:
    td_directory_url: "http://192.168.1.100:8080/td-directory"
```

### Well-Known URI
Compose URI from a known structure:
```yaml
environment:
  discovery_method: 'well_known'
  well_known:
    base_structure: "house/{building}/{room}"
```

### mDNS (default)
Discover on local network:
```yaml
environment:
  discovery_method: 'mdns'
  mdns:
    service_type: "_ami-hmas._tcp.local"
    timeout: 5
```

### Yggdrasil (direct platform URL)
Used for all demo and evaluation runs:
```yaml
environment:
  integration_engine:
    environment_type: 'yggdrasil'
    yggdrasil:
      url: "http://localhost:8080"
```

---

## LLM Configuration (`agents.yaml`)

The shared LLM defaults (provider, default model, retry timeout) live under the top-level `default_llm` key. An agent's own `llm:` section (e.g. `user_assistant.llm.atomic_segmentation`) sets per-behaviour models and falls back to `default_llm` for anything it omits. API keys should be set via environment variables, not hardcoded.

```yaml
default_llm:
  default_provider: "openai"
  providers:
    openai:
      api_key: null          # use OPENAI_API_KEY env var
      model: "gpt-4o-mini"
      temperature: 0.7
  retry:
    max_attempts: 3
    timeout: 30
```

The BT planning step uses the `planning.llm_planning` sub-key to allow a different model for plan generation vs. intent extraction:

```yaml
interaction_solver:
  planning:
    llm_planning:
      model: "o4-mini"
      reasoning_effort: "high"
      max_tokens: 2000
      max_planning_attempts: 3
```

Dependency structures (conditions, timing, maintenance) are planned by templates, with these
settings under `planning.dependency`:

| Key | Default | Meaning |
|---|---|---|
| `poll_seconds` | 30 | how often a regulated maintenance goal reads its value and acts again |
| `default_tolerance` | 0.5 | allowed deviation from a numeric target when the device states none |
| `stated_time_tolerance_seconds` | 60 | a start / end the user gave both as a clock time and relative to an anchor ("at 5:48 PM, which is 23 minutes before washer 1 finishes"): how far apart the two may be before the user is asked which one they mean |

Planning restrictions and guidelines live in their own file,
[`ami_agents/config/planning_guidelines.yaml`](../ami_agents/config/planning_guidelines.yaml):
plain-language operating conditions per device class or manufacturer ("while switched
off, other commands may be refused"). `planning.guidelines.path` (`PLANNING_GUIDELINES_PATH`)
points to another file; empty uses the default one. Only `power_on_first` is applied so
far, deterministically, to every plan (`interaction_solver/utils/power_on_guard.py`): in
front of a mode, level or operational-state command on a device the entry lists (by
class, or by exact manufacturer), the plan checks the device's on/off state and switches
it on first -- unless the plan already switches that device on before the command, in
execution order (dependency plans included). A device without a single on/off command is
left unguarded, with a note.

---

## Signifier Memory Engine (`agents.yaml`)

```yaml
env_explorer:
  signifiers:
    storage_backend: "sqlite"
    database_path: "data/signifiers.db"
    intent_similarity_threshold: 0.8
    context_matching_enabled: true
    matcher_version: "v1"
```

The explicit similarity threshold for fast-path vs. hint injection is set in [`ami_agents/bt_planning/planning/prompts.py`](../ami_agents/bt_planning/planning/prompts.py):

```python
EXPLICIT_SIMILARITY_THRESHOLD = 0.95
```

---

## Multi-Environment Setup (`agents.yaml`)

For the cross-environment experiment (Section 6.2 of the paper), each environment runs its own agent trio sharing one community API:

```yaml
environments:
  lab308:
    yggdrasil_url: "http://localhost:8080"
    agents:
      user_assistant:   { jid: "user_assistant_308@localhost" }
      env_explorer:     { jid: "env_explorer_308@localhost" }
      interaction_solver: { jid: "interaction_solver_308@localhost" }

  homebench:
    yggdrasil_url: "http://localhost:8090"
    agents:
      user_assistant:   { jid: "user_assistant_hb@localhost" }
      env_explorer:     { jid: "env_explorer_hb@localhost" }
      interaction_solver: { jid: "interaction_solver_hb@localhost" }

community:
  api_url: "http://localhost:8085"
```
