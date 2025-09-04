# Home Assistant Adapter Setup

Follow these steps to run Home Assistant and the Yggdrasil ↔ Home Assistant adapter locally.

## 1) Run Home Assistant (Docker)
Replace `MY_TIME_ZONE` and `PATH_TO_YOUR_CONFIG` with your values.

```
docker run -d \
  --name homeassistant \
  --privileged \
  --restart=unless-stopped \
  -e TZ=MY_TIME_ZONE \
  -v /PATH_TO_YOUR_CONFIG:/config \
  -v /run/dbus:/run/dbus:ro \
  --network=host \
  ghcr.io/home-assistant/home-assistant:stable
```

Open Home Assistant at http://localhost:8123/ (or http://your-server:8123) and complete onboarding.

## 2) Install HACS and Virtual Components
- Install HACS (container instructions): https://hacs.xyz/docs/use/download/download/#to-download-hacs
- In Home Assistant → HACS, search “Virtual Components” and install (defaults).
- Settings → Devices & Services → Add Integration → “Virtual Components”.

## 3) Add sample virtual devices
- Copy `lab308.yaml` to: `PATH_TO_YOUR_CONFIG/custom_components/virtual/lab308.yaml`
- Settings → Devices & Services → Virtual Components → “Add entry”:
  - Group name: `lab308`
  - File: `/config/custom_components/virtual/lab308.yaml`
  - Create an area named `lab308` when prompted.

## 4) Install adapter dependencies
```
pip install -r requirements.txt
```

## 5) Configure environment
Create `prepare-adapter-env.sh` (or export in your shell) with:

```
export HA_URL="ws://localhost:8123/api/websocket"
export HA_TOKEN="<your-long-lived-access-token>"  # Profile → Security → Create Token
export AREAS="lab308"                              # area_id created above
export BASE_WS_URI="http://localhost:8080"        # public base for adapter URIs
# Optional monitor/explorer endpoints
export MONITOR_URL="http://localhost:8081"
export EXPLORER_URL="http://localhost:8082"
```

## 6) Start the adapter
```
source prepare-adapter-env.sh
uvicorn ygg_ha_adapter:app --reload --port 8080 --log-level debug
```

On startup the adapter will:
- Reset MONITOR_URL (/reset) and EXPLORER_URL (/admin/reset)
- Register artifacts for AREAS to the Explorer
- Begin forwarding HA state changes to MONITOR_URL

Health check: `GET http://localhost:8080/_forwarder/status`.

## Utilities
adjust_lux_on_events.py - runs in the background and adjusts the internal light sensor up and down when blinds or lights change states. Uses HA_URL and HA_TOKEN environment variable to connect to HA. Light increments and decrements are configurable via INCREMENT_LIGHT, DECREMENT_LIGHT, INCREMENT_BLINDS and INCREMENT_BLINDS.

set_property.py - set a property in HomeAssistant using the same environment varibles. Examples:
- ./set-property.py person_counter_308 state 1
- ./set-property.py internal_light_sensing_308 state 350
- ./set-property.py temperature_sensing_308 state 25
- ./set-property.py external_light_sensing_308 state 4000
- ./set-property.py clock_308 state "2025-08-01T08:30:00Z"
- ./set-property.py lights_308 state on
- ./set-property.py lights_308 brightness_pct 75
- ./set-property.py lights_308 brightness 192
- ./set-property.py blinds_308 state closed
- ./set-property.py blinds_308 position 60
