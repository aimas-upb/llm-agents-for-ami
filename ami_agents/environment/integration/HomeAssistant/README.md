# HASP Setup

Follow these steps to run Home Assistant and HASP locally.

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

## 4) Install HASP dependencies
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
```

## 6) Start HASP
```
source prepare-adapter-env.sh
uvicorn hasp:app --reload --port 8080 --log-level debug
```

## 7) Subscriptions
HASP supports two callback-based notification entry points:

- `POST /workspaces/{workspace_id}/focus`
- `POST /hub/`

### Focus
Use `focus` to subscribe to a workspace or a specific artifact with a JSON body:

```json
{
  "callbackUrl": "http://listener.example/callback",
  "artifactName": "Temp Sensor"
}
```

Notes:
- `callbackUrl` is required.
- Omit `artifactName` to subscribe to the whole workspace.
- Include `artifactName` to subscribe only to that artifact.
- HASP performs callback intent verification before accepting the subscription.

Callback verification:
- HASP sends a `GET` to the callback URL with `hub.mode`, `hub.topic`, `hub.callback`, and `hub.challenge`.
- The callback must respond with the exact `hub.challenge` value as the response body.

### WebSub
Use `POST /hub/` for WebSub-style subscribe/unsubscribe requests.

Form fields:
- `hub.mode`: `subscribe` or `unsubscribe`
- `hub.topic`: workspace or artifact topic URI
- `hub.callback`: callback URL
- `hub.lease_seconds`: optional
- `hub.secret`: optional

Example:

```bash
curl -X POST http://localhost:8080/hub/ \
  -d "hub.mode=subscribe" \
  -d "hub.topic=http://localhost:8080/workspaces/lab308/artifacts/Temp%20Sensor#artifact" \
  -d "hub.callback=http://listener.example/callback"
```

### Notification Payload
When Home Assistant emits an unsolicited `state_changed` event, HASP updates its cache and sends a JSON `POST` to matching subscribers:

```json
{
  "topic": "http://localhost:8080/workspaces/lab308/artifacts/Temp%20Sensor#artifact",
  "workspaceId": "lab308",
  "artifactUri": "http://localhost:8080/workspaces/lab308/artifacts/Temp%20Sensor#artifact",
  "artifactTitle": "Temp Sensor",
  "entityId": "sensor.temp_sensor",
  "timestamp": "2025-01-01T00:00:00Z",
  "state": "21.5",
  "attributes": {
    "device_class": "temperature",
    "unit_of_measurement": "°C"
  }
}
```

Topic shapes:
- workspace topic: `http://localhost:8080/workspaces/lab308`
- artifact topic: `http://localhost:8080/workspaces/lab308/artifacts/Temp%20Sensor#artifact`

## 8) Useful scripts
- `./tick_clock.py` updates the `clock_308` artifact every second.
- `./adjust_lux_on_events.py` reacts to light and cover changes by updating one illuminance sensor.

## 9) Extended lab308e scenario
`lab308e.yaml` is a denser variant of `lab308` intended to make ambiguous requests harder without TD-SOSA semantics.

It adds:
- multiple light sources: `ambient_lights_308e`, `task_lights_308e`, `desk_lamp_308e`
- multiple covers: `blinds_308e`, `blackout_blinds_308e`, `window_308e`
- presentation/display context: `projector_308e`, `display_wall_308e`, `presentation_mode_308e`
- multiple climate actuators: `air_conditioner_308e`, `heater_308e`, `ceiling_fan_308e`
- more sensors: desk light, glare, humidity, CO2, presence

Import it the same way as `lab308.yaml`:
- copy `lab308e.yaml` to `PATH_TO_YOUR_CONFIG/custom_components/virtual/lab308e.yaml`
- add a new Virtual Components entry:
  - Group name: `lab308e`
  - File: `/config/custom_components/virtual/lab308e.yaml`
  - Create an area named `lab308e`

When running HASP for this workspace:
- set `AREAS` to the Home Assistant `area_id` created for `lab308e`
- keep `BASE_WS_URI` aligned with the port you use to start `uvicorn`

## 10) lab308e simulator loop
`./simulate_lab308e.py` is a hardcoded real-time environment loop for `lab308e`.

What it updates:
- `clock_308e`
- `external_light_sensing_308e` from real time of day
- `internal_light_sensing_308e`
- `desk_light_sensing_308e`
- `glare_sensing_308e`
- `temperature_sensing_308e`
- `humidity_sensing_308e`
- `co2_sensing_308e`

What it reads from Home Assistant before computing the next state:
- lights, covers, projector/display state
- presentation mode
- heater, AC, fan
- occupancy and person count

How it runs:
- subscribes to Home Assistant `state_changed` events over WebSocket
- recomputes immediately when relevant `lab308e` entities change
- also performs a periodic refresh so daylight and outdoor conditions continue to evolve with real time

Examples of modeled interactions:
- `desk_light_sensing_308e` is affected by `desk_lamp_308e`, `task_lights_308e`, `ambient_lights_308e`, blinds state, blackout blinds state, and presentation/projector context
- `internal_light_sensing_308e` depends on ambient/task/desk lighting plus daylight through the blinds
- `glare_sensing_308e` depends on daylight, blinds openness, and projector/display context
- temperature, humidity, and CO2 depend on occupancy, window openness, and climate actuator state

Run it with:

```bash
source prepare-adapter-env.sh
python simulate_lab308e.py
```

Optional environment variables:
- `LAB308E_TICK_SECONDS` default `5`

This script is hardcoded to the entity names generated from `lab308e.yaml`. If the Virtual Components integration creates different entity ids in your HA instance, update the `ENTITY` mapping inside `simulate_lab308e.py`.

## Utilities
set_property.py - set a property in HomeAssistant using the same environment variables. Examples:
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
