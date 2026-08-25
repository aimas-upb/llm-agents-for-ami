# Actuation effects vs. what SimuHome actually simulates

**Status: deliberately narrowed to match the simulator. To be revisited.**

`actuation_effects.yaml` states which environmental variable each command affects.
Those claims were curated by hand and approved under the principle that
`actsOnProperty` / `affectsObservableProperty` describe **capability** — what a
command *can* affect in general — leaving it to a planning agent to judge whether
the effect will materialise under current conditions.

**What these claims are for.** They are a *hint*: they narrow the planner's search to
the commands that can move a given variable. They are not a recipe. How to combine
those commands — in what order, with what values, and with which co-requisites — is
the BehaviorTree planner's problem, and solving it is what the benchmark measures.
An effect row therefore says "this command can influence that variable", nothing
more.

That principle still stands. But for now the tables are aligned to SimuHome's own
causal model instead, so that **AMI agent results can be compared directly against
the published SimuHome benchmark**. A claim the simulator will never honour makes a
plan look correct while the measured value never moves — which would make any
comparison meaningless.

The ground truth is served live by the simulator itself:

```
GET /api/environment/control_rules/{temperature|humidity|illuminance|air_quality}
```

and is read in code by `SimuHomeClient.control_rules()`
(`ami_agents/environment/integration/SimuHome/sim_client.py`), which also unwraps
the JSON-string nesting the server uses.

---

## What was removed

Two families claimed effects SimuHome does not simulate. **10 rows** lost their
effect claim; the actions themselves are still emitted, they simply carry no
`tdsosa:hasEffectActuation`.

| family | claimed | why removed |
|---|---|---|
| `Fan` | `air_temperature` (On decreases) | SimuHome models only `air_conditioner` and `heat_pump` under `temperature`. A fan moves air; it does not change the room's temperature in this simulator. |
| `WindowCoveringController` | `illuminance` (Open increases) | SimuHome models only `on_off_light` and `dimmable_light` under `illuminance`. Daylight through a blind is not simulated. |

Removed at source in `tables.py::FAMILY_EFFECTS`, so regeneration will not
reintroduce them.

**To improve later**: both are physically real. A fan produces a cooling effect on
occupants (and the old HA-side sidecar did model it as such), and blinds obviously
govern daylight. If SimuHome gains those dynamics — or is extended locally — restore
the two entries and the rows regenerate.

---

## What SimuHome does simulate

Verified against a live server, all four states:

| state | device types | required actions |
|---|---|---|
| `temperature` | `air_conditioner` | `OnOff.On` **+** `Thermostat.SystemMode = 3` **+** `OccupiedCoolingSetpoint < current` |
| | `heat_pump` | `Thermostat.SystemMode = 4` **+** `OccupiedHeatingSetpoint > current` — *no On/Off command exists* |
| `humidity` | `humidifier` | `OnOff.On` |
| | `dehumidifier` | `OnOff.On` |
| `illuminance` | `on_off_light` | `OnOff.On` raises, `OnOff.Off` lowers |
| | `dimmable_light` | `OnOff.On`; `LevelControl.MoveToLevel` for fine adjustment |
| `air_quality` | `air_purifier` | `OnOff.On` |

Nothing SimuHome models is missing from our tables — the mismatch was only ever in
the other direction.

---

## Conjunctive recipes are the planner's job, not the TD's

The simulator's rules often need several actions together:

- The air conditioner needs `OnOff.On` **+** `Thermostat.SystemMode = 3` **+**
  `OccupiedCoolingSetpoint < current`, *and* at least one of
  `FanControl.PercentSetting` / `FanMode` / `Step`.
- The humidifier, dehumidifier and air purifier each need `OnOff.On` plus at least
  one optional fan action.
- The heat pump has no On/Off command at all, and its setpoint is relative
  (`> current`), not absolute.

`actuation_effects.yaml` models **one command per row**, and that is correct.
The TD's role is to say *which commands can influence which variable*, so the
planning agent can narrow its search to those. Working out that cooling requires
the mode and the setpoint together, in the right order, with a value below the
current temperature, is exactly what the BehaviorTree planner is for.

Encoding the recipe in the TD would hand the planner its answer and stop measuring
the thing the benchmark exists to measure. Every required and optional action is
already an affordance in its own right, so the search space is complete — it is just
not pre-solved.

This is a design decision, not an outstanding gap.

---

## Re-checking the alignment

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, "ami_agents/environment/integration/SimuHome")
from sim_client import SimuHomeClient
c = SimuHomeClient("http://127.0.0.1:<sim-port>/api")
for s in ("temperature", "humidity", "illuminance", "air_quality"):
    print(s, [e["device_type"] for e in c.control_rules(s)])
PY
```

Compare against the `family` / `affectsObservableProperty` pairs in
`tests/simuhome/td/mappings/actuation_effects.yaml`. Phase 4 of the SHTD plan turns
this into an automated check.
