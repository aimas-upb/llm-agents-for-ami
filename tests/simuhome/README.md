# SimuHome Workflow

This directory contains helper scripts for turning [SimuHome](https://github.com/holi-lab/SimuHome.git) benchmark scenarios into Home Assistant Virtual Devices, checking that `HASP` exposes them, and removing them again.

## 1. Clone SimuHome

Clone the SimuHome repository somewhere on your machine:

```bash
git clone https://github.com/holi-lab/SimuHome.git
```

The benchmark scenarios used by these scripts are expected to be the JSON files from the SimuHome repo, for example under:

```bash
SimuHome/data/benchmark/
```

## 2. Install Requirements

You need:

- Python 3.9+
- Home Assistant running locally or remotely
- the Home Assistant `virtual` custom component installed

Python packages needed by the scripts in this directory:

```bash
pip install pyyaml httpx websockets rdflib
```

## 3. Prepare SimuHome Scenarios as Home Assistant YAML

Use:

- `initial_home_config_to_homeassistant_yaml.py`

It walks a directory recursively, reads each SimuHome JSON file, extracts `initial_home_config`, and writes a sibling `.yaml` file next to the original `.json`.

Example:

```bash
python tests/simuhome/initial_home_config_to_homeassistant_yaml.py \
  /path/to/SimuHome/data/benchmark
```

After that, a file like:

```bash
/path/to/SimuHome/data/benchmark/qt1_feasible_seed_1.json
```

will produce:

```bash
/path/to/SimuHome/data/benchmark/qt1_feasible_seed_1.yaml
```

## 4. Environment Variables

The remaining scripts use these environment variables:

- `HA_URL` — Home Assistant websocket URL, for example `ws://localhost:8123/api/websocket`
- `HA_TOKEN` — Home Assistant long-lived access token
- `HASP_URL` — running HASP base URL, default `http://localhost:8080/`
- `AREAS` — optional, used by the exposure-check script when you do not pass a workspace id explicitly

Example:

```bash
export HA_URL="ws://localhost:8123/api/websocket"
export HA_TOKEN="your-long-lived-token"
export HASP_URL="http://localhost:8080/"
```

## 5. Add a SimuHome YAML to Home Assistant

Use:

- `add_virtual_devices.py`

This script:

- copies a YAML into the Home Assistant host directory used by the `virtual` component
- creates or reuses a Home Assistant area
- imports the YAML through the `virtual` config flow
- assigns created devices/entities to that area

Usage:

```bash
python tests/simuhome/add_virtual_devices.py \
  /path/to/qt1_feasible_seed_1.yaml \
  QT1FeasibleSeed1 \
  /path/to/homeassistant/custom_components/virtual
```

Important:

- the third argument is the host directory that Home Assistant sees as `/config/custom_components/virtual`
- inside Home Assistant, the script registers the file as `/config/custom_components/virtual/<filename>`

## 6. Identify the Workspace ID After Adding It

Once the area is added, you need the Home Assistant `area_id` if you want `HASP` to expose that workspace.

The easiest way to get it is from the output of `add_virtual_devices.py`. It prints:

- `area_name`
- `area_id`
- `hasp_areas_env`
- `hasp_hint`

Example:

```json
{
  "area_name": "QT1FeasibleSeed1",
  "area_id": "qt1feasibleseed1",
  "hasp_areas_env": "qt1feasibleseed1",
  "hasp_hint": "Set AREAS=qt1feasibleseed1 for HASP"
}
```

That `area_id` is the exact value to use for `HASP`:

```bash
export AREAS="qt1feasibleseed1"
```

Then start `HASP` with that environment.

## 7. Check That HASP Exposes the Devices

Use:

- `check_virtual_devices_exposed.py`

This script:

- reads the YAML
- derives expected artifact slugs
- queries `HASP`
- reports which YAML devices are exposed and which are missing

Usage with explicit workspace id:

```bash
python tests/simuhome/check_virtual_devices_exposed.py \
  /path/to/qt1_feasible_seed_1.yaml \
  qt1feasibleseed1
```

Usage relying on `AREAS`:

```bash
export AREAS="qt1feasibleseed1"

python tests/simuhome/check_virtual_devices_exposed.py \
  /path/to/qt1_feasible_seed_1.yaml
```

The script outputs JSON including:

- `all_devices_exposed`
- `exposed_devices`
- `missing_devices`
- `hasp_artifact_slugs`

## 8. Run the SimuHome Sidecar Simulator

Use:

- `simulate_simuhome_scenario.py`

This script watches the Home Assistant entities generated from one SimuHome
benchmark JSON and updates the room environment sensors:

- `temperature`
- `humidity`
- `illuminance`
- `pm10`

Usage:

```bash
python tests/simuhome/simulate_simuhome_scenario.py \
  /path/to/SimuHome/data/benchmark/qt2_feasible_seed_1.json
```

Optional controls:

```bash
SIMUHOME_TICK_SECONDS=0.1 SIMUHOME_SIM_SPEED=1 SIMUHOME_HA_SYNC_SECONDS=1 \
python tests/simuhome/simulate_simuhome_scenario.py \
  /path/to/SimuHome/data/benchmark/qt2_feasible_seed_1.json
```

`SIMUHOME_SIM_SPEED` controls how many simulated seconds elapse per real second.
The default is `1`, and `SIMUHOME_TICK_SECONDS` defaults to the scenario
`tick_interval`, matching SimuHome's real-time pacing.
`SIMUHOME_HA_SYNC_SECONDS` controls how often the sidecar refreshes actuator
state from Home Assistant and writes sensor updates back; the default is `1`.

Important:

- The script expects entity names produced by
  `initial_home_config_to_homeassistant_yaml.py`.
- It updates sensors only. Actuators remain under Home Assistant / agent control.
- Devices currently exported as read-only sensors, such as some HVAC entities,
  can only influence the simulation if their control/state sensors are changed
  elsewhere.

## 9. Run One SimuHome E2E Case

Use:

- `run_simuhome_e2e.py`

This script orchestrates one supported SimuHome benchmark JSON through the full
Home Assistant/HASP/AMI runner path:

1. Converts the benchmark JSON to Virtual Devices YAML.
2. Imports the YAML into Home Assistant.
3. Starts HASP for the imported workspace.
4. Starts `simulate_simuhome_scenario.py`.
5. Generates a temporary e2e case and runs `tests/e2e-lab308e/run_cases.py`.
6. Saves logs, generated YAML, generated case JSON, and a manifest under
   `tests/simuhome/results/`.
7. Stops subprocesses and removes the imported Virtual Devices workspace.

Current scope:

- `qt2` feasible environmental-control cases.

Usage:

```bash
HA_URL=ws://localhost:8123/api/websocket HA_TOKEN=... OPENAI_API_KEY=... \
~/aiml/env-spade-3/bin/python tests/simuhome/run_simuhome_e2e.py \
  /path/to/SimuHome/data/benchmark/qt2_feasible_seed_1.json \
  --virtual-yaml-dir /path/to/homeassistant/custom_components/virtual
```

Useful options:

- `--hasp-port 8081` if port 8080 is already in use.
- `--keep-workspace` to inspect the imported Home Assistant workspace after a run.
- `--dump-prompts` to forward prompt dumping to the underlying e2e runner.
- `--delta temperature=0.1` to loosen or tighten a generated directional assertion.

## 10. Remove a Workspace from Home Assistant

Use:

- `remove_virtual_devices.py`

This script:

- resolves the Home Assistant area by name
- finds devices/entities assigned to that area
- removes linked `virtual` config entries

Usage:

```bash
python tests/simuhome/remove_virtual_devices.py QT1FeasibleSeed1
```

Notes:

- it removes Virtual Devices config entries from Home Assistant
- it does not delete the YAML file from the filesystem
- if a config entry is linked across multiple areas, it is skipped rather than removed

## Typical End-to-End Flow

1. Clone SimuHome.
2. Convert benchmark JSON files with `initial_home_config_to_homeassistant_yaml.py`.
3. Add one generated YAML to Home Assistant with `add_virtual_devices.py`.
4. Read the printed `area_id` and set `AREAS=<area_id>` for `HASP`.
5. Start `HASP`.
6. Check exposure with `check_virtual_devices_exposed.py`.
7. Run `simulate_simuhome_scenario.py` for scenarios that need environment dynamics.
8. Or use `run_simuhome_e2e.py` for a one-command supported qt2 run.
9. Remove the imported workspace later with `remove_virtual_devices.py`.
