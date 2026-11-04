# Home Assistant for the HASP lab308e harness

The harness (`tests/hasp/hasp_lab308e_harness.py`) needs a running Home
Assistant with the Virtual Components integration. It imports `lab308e.yaml`,
assigns the entities to the `lab308e` area, and starts HASP and the lab308e
dynamics simulator itself. Nothing else has to be configured in HA by hand.

## One-time setup (Windows, Docker Desktop)

```powershell
# Docker Desktop must be running; docker.exe lives in
# %LOCALAPPDATA%\Programs\DockerDesktop\resources\bin if it is not on PATH.
python tests/hasp/ha/bootstrap_ha.py
```

`bootstrap_ha.py` is idempotent. It installs Virtual Components (hass-virtual
release into `ha-config/custom_components/virtual`), starts the container from
`docker-compose.yml` (port 8123 on loopback only), completes onboarding on a
fresh HA, creates a long-lived token, and writes `HA_URL`, `HA_TOKEN` and
`HA_VIRTUAL_DIR` into the repository `.env`. The HA login it creates is saved
in `.ha-credentials`; everything generated here is gitignored.

## Run

```powershell
python tests/hasp/hasp_lab308e_harness.py --config tests/hasp/ollamamix1-local.yaml `
    --runs 10 --daylight --virtual-dir tests/hasp/ha/ha-config/custom_components/virtual
```

- `ollamamix1.yaml` points at the lab Ollama (`10.205.0.124`, reachable only
  from the lab network). `ollamamix1-local.yaml` is the same mix on the Ollama
  of this machine (`localhost:11434`); both need the models `qwen3:4b`,
  `qwen3:8b`, `gemma3:1b` and `fufufuyangyang/Qwen2.5-Coder-7B-Instruct`.
- `gpt-5-nano` phases need `OPENAI_API_KEY` in `.env`.
- Results go to `tests/hasp/results/harness/<timestamp>/`; pass `--out` to
  resume or to rebuild the statistics with `--stats-only`.
- No XMPP server is needed: this harness does not start the SPADE agents.

## Notes

- The `--virtual-dir` in an upstream command line (`/Users/<name>/homeassistant/...`)
  is the host folder that HA sees as `/config/custom_components/virtual`; here
  that is the bind mount under `ha-config`.
- Port 8123 (HA) belongs to the container; the harness starts HASP itself on a
  free port unless `--hasp-port` is given.
