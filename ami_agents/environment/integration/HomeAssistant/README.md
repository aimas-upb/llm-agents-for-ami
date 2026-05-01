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

### 7) Useful scripts
./tick_clock.py will update the clock in lab_308 with the current time every second
./adjust_lux_on_events.py will react to lights being turned on or off and to the blinds being moved by updating the value of the luminosity sensor

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
