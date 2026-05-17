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

## 8. Remove a Workspace from Home Assistant

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
7. Remove the imported workspace later with `remove_virtual_devices.py`.
