# Phase 4 — TD-SOSA semantics: what can sense and what can actuate what

Phases 2 and 3 make the world legible and operable. Phase 4 says what each
affordance *means*: which observable property it senses, and which it can move.
Both halves use `tdsosa:` (`dev/ontologies/td-sosa-extension-v2.ttl`), so a
planner can answer "who can tell me this room's temperature?" and "what can I do
about it?" from the graph alone.

## A. Sensing — the half that is currently missing entirely

**Finding (measured, 200 episodes).** SimuHome has no standalone sensors: all 16
device types are appliances or actuators. But four families genuinely sense the
room's air, and their readings match the room state exactly:

| family | senses | instances | matches room? |
|---|---|---|---|
| `dehumidifier` | `RelativeHumidityMeasurement.MeasuredValue` | 375 | **yes** |
| `air_conditioner` | `Thermostat.LocalTemperature` | 344 | **yes** |
| `heat_pump` | `Thermostat.LocalTemperature` | 325 | **yes** |
| `humidifier` | `RelativeHumidityMeasurement.MeasuredValue` | 120 | **yes** |
| `freezer` | `TemperatureMeasurement.MeasuredValue` | 216 | **no** — own compartment |
| `refrigerator` | `TemperatureMeasurement.MeasuredValue` | 207 | **no** — own compartment |

The split is clean: zero ambiguous instances. Verified live — the utility-room
heat pump reads 23.69 °C where the room is 23.69 °C, while the kitchen freezer
reads −15.0 °C in a 23.62 °C kitchen.

**`illuminance` and `pm10_mass_concentration` have no sensor anywhere in the
corpus.** For those two the room's environmental property is the only source.
That is the simulator's shape, not a modelling shortcut, and it should be stated
rather than left to be discovered.

**The gap.** `sosa:hasFeatureOfInterest` is currently emitted only when a family
*affects* a room variable. That correctly excludes the freezer, but it links the
heat pump and dehumidifier as **actuators only** — the fact that they also
*observe* the room is nowhere in the TD. A planner cannot currently find out
which device can report a room's temperature, even though the answer exists.

**To add:**

1. A new mapping table (`sensing.yaml`) — family → the room property it senses,
   evidenced by the corpus measurement above, reviewed like every other table.
   Freezer and refrigerator appear with an explicit *not the room* verdict so the
   distinction is recorded, not merely absent.
2. On the artifact: `sosa:Sensor` typing where a family senses a room property.
3. On the property affordance: a link to the `sosa:ObservableProperty` it
   observes — `tdsosa:affordsProperty` is the existing term
   (`PropertyAffordance` → `Property`) and fits without minting anything.
4. On the room's observable property: `sosa:isObservedBy` pointing back at the
   device. This is the inverse direction the planner actually needs, and it is
   also the term I removed in phase 2 for pointing the environment at itself —
   it becomes correct once there is a real sensor to name.
5. Compartment sensors (`freezer`, `refrigerator`) get their own
   `sosa:FeatureOfInterest` — the appliance interior — rather than being left
   dangling. They observe something real; it just is not the room.

## B. Actuation effects — the 91 approved rows

`actuation_effects.yaml` is approved and unused so far. Phase 3 created the
`td:ActionAffordance` nodes these attach to, so they can now be emitted:

```turtle
<…/actions/onOff> tdsosa:hasEffectActuation [
    a tdsosa:IncreasingActuation ;
    sosa:actsOnProperty <…/kitchen#environment-illuminance> ;
    tdsosa:increasesObservableProperty <…/kitchen#environment-illuminance> ] .
```

Direction comes from the reviewed table, following the rule already settled:
**On/Off commands carry a direction; percentage- and level-wise commands are
directionless** (`tdsosa:affectsObservableProperty`).

The alignment was narrowed in this session to exactly what SimuHome simulates —
`air_temperature`→{AirConditioner, HeatPump}, `illuminance`→{DimmableLight,
OnOffLight}, `pm10_mass_concentration`→{AirPurifier},
`relative_humidity`→{Dehumidifier, Humidifier} — with 45 rows carrying no effect.
See `tests/simuhome/td/docs/effects-vs-simulator.md`.

**These claims are hints, not recipes.** They narrow the planner's search to the
commands that can move a variable. Working out that cooling needs `SystemMode=3`
*plus* a setpoint below current, in the right order, is the BehaviorTree
planner's job and is what the benchmark measures.

## C. Reconciliation

Diff the emitted effects against the simulator's own ground truth:

```
GET /api/environment/control_rules/{temperature|humidity|illuminance|air_quality}
```

read by `SimuHomeClient.control_rules()`. Every difference explained or the table
corrected. `tests/simuhome/td/verify.py` check `4b-ii` already pins the family
alignment; phase 4 turns the live comparison into an automated check too.

## D. Evaluation wiring

Read `src/pipelines/episode_evaluation/`'s interface and determine what it needs
from an agent run (action trace? final home state?), then wire the outcome check
to it. The in-scope evaluation is **end-of-scenario outcome only** — did the plan
achieve what was asked — which is the one thing comparable against the published
SimuHome benchmark. The current `property_gte`/`property_lte` assertions in
`run_simuhome_e2e.py::_build_qt2_case` are a proxy invented because the HA path
had no scorer, and can be replaced by the evaluator's verdict.

## Verification

- Every family in `sensing.yaml` that senses a room property emits
  `sosa:Sensor` + an observation link; freezer and refrigerator emit a
  compartment feature-of-interest and **no** room link.
- Every effect row in `actuation_effects.yaml` appears on the corresponding
  action affordance; count matches.
- Emitted effects reconcile with `control_rules` for all four states.
- A planner query — "which affordances can raise this room's illuminance?" —
  answers from the graph alone. The two `/_graph/query/*` endpoints in `hasp.py`
  are the shape to copy.
