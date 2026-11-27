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
export SEMANTIC_CONFIG="lab308.yaml"               # the home's `semantic:` block
```

`SEMANTIC_CONFIG` is all HASP needs beyond the connection: the room variables,
the sensors observing them and the actions moving them are derived from the
deployment (see "Room variables" below). The former `TD_SOSA_ENV_VAR_OVERRIDES`,
`TD_SOSA_PROPERTY_RANGES` and `TD_SOSA_SETTLING_TIMES` are no longer read.

## 6) Start HASP
```
source prepare-adapter-env.sh
uvicorn hasp:app --reload --port 8080 --log-level debug
```

### The home's clock: `GET /_clock`
With `CLOCK_ENTITY` set to an HA timestamp entity (the BEIA virtual lab's
`sensor.sim_time`), HASP serves that entity's time:

```
export CLOCK_ENTITY="sensor.sim_time"
curl -s http://localhost:8080/_clock
# {"now": "2026-11-27 13:05:30", "iso": "2026-11-27T13:05:30+02:00", "entity_id": "sensor.sim_time"}
```

The agents' plan clock reads it (`CLOCK_SOURCE=http
CLOCK_URL=http://localhost:8080/_clock`), so a plan's waits ("held for 5
minutes") run on the simulator's time, sped up by its `number.sim_speed`. Unset,
or with the entity unavailable, `/_clock` is a 404.

## 6b) Artifacts, affordance names and routes
One artifact is one Home Assistant **device**, holding all of its entities. An
entity with no device (a helper, a template sensor) is an artifact of its own.

- **Artifact slug** (URL segment, `td:title`): the device name, slugified
  ("Blue Air Garage" -> `blue_air_garage`). `schema:name` is HA's display name.
  A request may also name the artifact by its device name or by any of its
  entities' ids / object ids.
- **Affordance names** come from each entity's *key*: its object id with the
  device prefix stripped; the domain when nothing is left; the domain in front
  only when two entities of the device would get the same key.
  - state property: `<key>`
  - attribute property: `<key>_<attribute>`; descriptive attributes: `<key>_metadata`
  - action: `<key>_<service>`; a climate entity's whole state: `<key>_get_state`

  Example, the SolarAssistant Effekta device:
  `sensor.solarassistant_effekta_battery_state_of_charge` -> property
  `battery_state_of_charge`; `select.solarassistant_effekta_charger_source_priority`
  -> property `charger_source_priority`, action
  `charger_source_priority_select_option`. A one-entity device such as
  lab308e's `desk_lamp_308e` gets `light`, `light_brightness`, `light_turn_on`.
- **Routes** are HASP's own; no entity id or HA service path appears in a TD:
  - `GET  /workspaces/{ws}/artifacts/{artifact}/properties/{name}`
  - `POST /workspaces/{ws}/artifacts/{artifact}/actions/{name}`

  HASP keeps, per artifact, which entity and service each name stands for. A
  name the artifact does not have returns 404 with the names it does have.

## 7) Subscriptions
HASP supports two callback-based notification entry points:

- `POST /workspaces/{workspace_id}/focus`
- `POST /hub/`

### Focus
Use `focus` to subscribe to a workspace or a specific artifact with a JSON body:

```json
{
  "callbackUrl": "http://listener.example/callback",
  "artifactName": "temp_sensor"
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
  -d "hub.topic=http://localhost:8080/workspaces/lab308/artifacts/temp_sensor#artifact" \
  -d "hub.callback=http://listener.example/callback"
```

### Notification Payload
When Home Assistant emits an unsolicited `state_changed` event, HASP updates its cache and sends a JSON `POST` to matching subscribers. The topic is the artifact (device) holding the entity; `affordance` is the property that reads the entity's state:

```json
{
  "topic": "http://localhost:8080/workspaces/lab308/artifacts/temp_sensor#artifact",
  "workspaceId": "lab308",
  "artifactUri": "http://localhost:8080/workspaces/lab308/artifacts/temp_sensor#artifact",
  "artifactTitle": "temp_sensor",
  "affordance": "sensor",
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
- artifact topic: `http://localhost:8080/workspaces/lab308/artifacts/temp_sensor#artifact`

## 8) Useful scripts
- `./tick_clock.py` updates the `clock_308` artifact every second.
- `./adjust_lux_on_events.py` reacts to light and cover changes by updating one illuminance sensor.

## 9) Extended lab308e scenario
`lab308e.yaml` is a denser variant of `lab308` intended to make ambiguous requests harder without TD-SOSA semantics.

It adds:
- multiple light sources: `ambient_lights_308e`, `task_lights_308e`, `desk_lamp_308e`
- multiple covers: `blinds_308e`, `blackout_blinds_308e`, `window_308e`
- presentation/display context: `projector_308e`, `display_wall_308e`
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
- set `SEMANTIC_CONFIG=lab308e.yaml` so the TDs carry the room and the
  per-device overrides (see "Semantic typing" below)

## Semantic typing (homeont / SAREF)
HASP types every TD it serves with the home ontology
(`ami_agents/shared/ontologies/homeont.ttl`), the same way SimuHome's TD builder
does, so the agents' class-based ENV_STATE and capability queries work on a Home
Assistant home:
- the workspace's room (`homeont:StudyRoom`, ...) and its environment;
- each device's class (`homeont:ColorLight`, `homeont:AirConditioner`, ...);
- each property's class (`homeont:DimmableLightBrightness`, `homeont:Illuminance`, ...);
- each action's command (`saref:OnCommand`, `homeont:SetModeCommand`, ...);
- the room's environment variables, which sensors observe and actions move
  (below).

Defaults live in `semantic_mappings.yaml`. A home overrides them in the
`semantic:` block of its config, read from `SEMANTIC_CONFIG`: `location_class`,
`areas`, `environments`, `ranges`, and per device `device_class`, `properties`,
`commands`, `zone`, `environment` and `effects` (see `lab308e.yaml`, one
room, and `homelab.yaml`, several areas). Anything unmapped stays untyped and is
logged once. No `ex:` types are emitted.

- **Per area** (`areas:`, keyed by HA area_id): `location_class` says which
  room an area is (`homeont:Kitchen`). A home-wide `location_class` still
  types every area that has none of its own.
- **Device groups.** `group_class: "homeont:DeviceGroup"` declares an area that is
  no place at all: an energy system, a telemetry set, where a device's physical
  placement means nothing.
  - Its workspace stands for the group: `<ws#group> a homeont:DeviceGroup ;
    rdfs:label "<area name>" ; homeont:isGroupOfWorkspace <ws>`. It has no
    `#place`, no environment, and no room variables.
  - Its devices are `hmas:contains`ed but not `homeont:isLocatedIn` anything, and
    observe or move no room variable. An inverter's °C reading is never a room's
    `air_temperature`.
  - The agents still find these devices by class or name, and an answer names
    the group ("… in Invertoare").
  - A request cannot yet be *scoped* to a group: the UA's location slot is a
    room class.
- **A device of several classes.** `device_class` may be a list, for a device
  that is several things at once: the BEIA Blue Air is
  `["homeont:AirPurifier", "homeont:Dehumidifier"]`. The artifact gets every
  class, its kind label names the first, and a rule's `device:` filter matches
  any of them, so each half (its fan, its humidifier entity) is typed by the
  rules of its own class.
- **Per property and action.** On a device with several entities, a
  `properties:` / `commands:` key may be the affordance name the TD shows
  (`max_temperature`, `send_command_set_value`). That types one entity. A
  signal or service key (`state`, `set_value`) types that signal of every
  entity of the device.

### Room variables
The same model SimuHome's TD builder serves, built from the HA deployment:

```
<ws#place>        homeont:hasEnvironment  <ws#environment>
<ws#environment>  ssn:hasProperty         <ws#illuminance>, <ws#air_temperature>, ...
<sensor#artifact> sosa:observes           <ws#illuminance>      (a sosa:Sensor, saref:Sensor)
<light action>    tdsosa:hasEffectActuation [ sosa:actsOnProperty <ws#illuminance> ]
<every artifact>  homeont:isLocatedIn     <ws#place>
```

- **Which variables**: none are listed. homeont says which of its classes are a
  property of a space's environment (`ssn:isPropertyOf homeont:Environment`);
  a device property typed with one of them (by the `properties` rules, from HA's
  `domain` / `device_class` / unit) observes that variable. A sensor with no
  `device_class` is typed by its unit when exactly one environment class has it
  (`units` table: `lx` -> `unit:LUX` -> `homeont:Illuminance`).
- **Names**: the class in snake_case, as SimuHome names its room variables
  (`homeont:AirTemperature` -> `air_temperature`); a device's `zone` prefixes it
  (`desk_illuminance`), its `environment` moves it to another environment
  (`outdoor_illuminance`, of `<ws#outdoor_environment>`, which no place has).
- **Quantity kind and unit**: from the class in homeont.
- **Effects**: the `effects` rules (lights and covers move illuminance, climate
  air temperature, humidifiers humidity); a device's own `effects` replace them.
  The direction comes from the service name (`turn_on` increases) unless a rule
  states it; setters are directionless.
- **Reading a variable**: `GET /workspaces/{ws}/properties/{variable}` returns
  the reading of a device observing it (a sensor first); a variable no device
  observes has no form and returns 404.
- **Graph queries**: `/_graph/query/actions-affecting-observable-property`
  takes a variable name, its IRI, or one of the older keys (`luminosity`,
  `thermal_comfort`; `variable_aliases`).

Each entity's state is its own property, named by the entity's key (see
"Artifacts, affordance names and routes"). The old `<domain>State` names and
the `…/properties/state` alias are no longer served.

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
- heater, AC, fan
- occupancy and person count

How it runs:
- subscribes to Home Assistant `state_changed` events over WebSocket
- recomputes immediately when relevant `lab308e` entities change
- also performs a periodic refresh so daylight and outdoor conditions continue to evolve with real time

Examples of modeled interactions:
- `desk_light_sensing_308e` is affected by `desk_lamp_308e`, `task_lights_308e`, `ambient_lights_308e`, blinds state, blackout blinds state, and projector/display context
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
