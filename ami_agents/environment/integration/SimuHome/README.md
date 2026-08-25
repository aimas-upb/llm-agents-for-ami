# SHTD — SimuHome as Thing Descriptions

Serves a loaded SimuHome home as the HMAS/TD tree the AMI agents already crawl,
talking native Matter to the simulator instead of routing through Home Assistant.

**Status: phase 5 in progress** — Thing Descriptions, action affordances,
state-change notification over WebSub, TD-SOSA semantics, and an isolated worker
model scored by SimuHome's own evaluator. Remaining: the harness that assigns
episodes to workers and runs the BT planner in each.

## Why this exists

The agents consume Thing Descriptions and plan BehaviorTrees over them. Home
Assistant was only ever a way to produce those TDs, and it turned out to be lossy:
the `virtual` component has no `climate` platform and rejects the converter's
`cover` config, so **3,480 devices corpus-wide were silently never created** — every
air conditioner, heat pump and window covering. Mode labels ("Heavy") exceeded HA's
255-char state limit and stored as `unknown`.

SimuHome ships its own FastAPI server speaking Matter directly. Going straight to
it removes Home Assistant, the YAML converter and the sidecar simulator, and
recovers everything the HA path dropped.

## Layout

| file | role |
|---|---|
| `matter_model/` | the vendored CSA Matter registry (connectedhomeip v1.5.1.0) plus its bootstrap, normalizer and selftest — every Matter fact comes from here, never from model memory |
| `sim_client.py` | thin client for the simulator's HTTP API |
| `classify.py` | is an attribute observable, actuatable, or plumbing? Shared by the inspector and the TD generator, so they agree by construction |
| `mappings.py` | loads the approved Phase A tables (`tests/simuhome/td/mappings/`) |
| `td_builder.py` | renders the RDF: platform → home → rooms → devices |
| `shtd.py` | the server |
| `inspector.py` | human-readable view of the same home |
| `run_inspector.py` | one-command launcher: simulator + episode + inspector |
| `notifications.py` | WebSub registry, the state poller, and change fan-out |
| `verify_phase2.py` | crawls SHTD the way the agent crawler does and checks the result |
| `verify_phase3.py` | invokes actions and asserts notifications arrive |
| `worker.py` | one isolated simulator + SHTD pair, and the episode lifecycle |
| `verify_phase4.py` | checks the sensing/actuation semantics and reconciles them against the simulator |
| `verify_phase5.py` | worker isolation, episode lifecycle, and scoring with SimuHome's evaluator |
| `verify_dispatch.py` | static: every actuatable attribute in the corpus is dispatchable, with the argument names SimuHome accepts |
| `PHASE4.md` | what phase 4 must do, and the corpus evidence behind it |
| `AGENT_UPDATES.md` | changes owed to the agent layer, found while pointing the agents at SHTD |

## Running

```bash
# simulator
SERVER_PORT=8099 python -m src.simulator.api.app        # in dev/SimuHome
# load an episode, then:
python ami_agents/environment/integration/SimuHome/shtd.py \
    --sim http://127.0.0.1:8099/api --home qt2_feasible_seed_77 --port 8097
```

Point `YGGDRASIL_URL` at `http://127.0.0.1:8097` and the agents discover it
unchanged.

## Routes

```
GET  /                                              platform
GET  /workspaces                                    hosted workspaces
GET  /workspaces/{home}                             home workspace
GET  /workspaces/{home}/{room}                      room workspace
GET  /workspaces/{home}/{room}/artifacts            artifact directory
GET  /workspaces/{home}/{room}/artifacts/{device}   the Thing Description
GET  .../artifacts/{device}/properties/{name}       one property value
GET  /workspaces/{home}/{room}/properties/{name}    one environmental value
POST .../artifacts/{device}/actions/{name}           invoke  {"value": ...}
POST /hub/                                          WebSub subscribe/unsubscribe
GET  /_shtd/status                                  what is being projected
GET  /_shtd/subscriptions                           who is currently subscribed
```

A property read returns **the bare value**, matching its declared schema: a
`js:BooleanSchema` dereferences to `false`, a `js:NumberSchema` to `40.83`. Unit,
cluster and quantity kind live in the Thing Description, not in every read.

### Subscribing

Change notification is plain WebSub, declared as affordances on the resource it
concerns — there is no `focus` route (that was a JaCaMo notion):

| declared on | affordances | a subscriber receives |
|---|---|---|
| home workspace | `subscribeToWorkspace` / `unsubscribeFromWorkspace` | every device in the home, and every room's environment |
| room workspace | same pair | that room's devices and its environment |
| artifact | `subscribeToArtifact` / `unsubscribeFromArtifact` | that artifact's own state changes only |

All four target `POST /hub/` and name the resource in `hub.topic`; the input
schema pins `hub.mode` and `hub.topic` as `js:const`, so an agent copies the
affordance rather than constructing a subscription by hand. A topic this server
does not serve is rejected with 404, so nobody subscribes to something that will
never notify.

A notification carries any internal state change: the TD property name, the new
value and the previous one, reported in the same units a read returns.

An action returns an **acknowledgement**, never a state claim — SimuHome models
devices responding over ticks, so a plan confirms with a verification node
reading the property back. Device rules surface as HTTP 422 with the simulator's
own reason (`"Cannot perform operation when power is OFF"`).

`{home}` is the episode id and is **load-bearing** — each seed is a different
house (217 distinct device layouts across the 600 episodes). Matter identity stays
out of the URL *and* out of the TD: `{name}` is the TD's `td:name`
(`currentMode`, `onOff`), which is unique within a Thing, and SHTD resolves it
back to cluster/attribute/endpoint from the registry when it dispatches. Three
names that would otherwise collide on one device are qualified in
`attribute_map.yaml` (`runCurrentMode`/`cleanCurrentMode`,
`powerAccuracy`/`energyAccuracy`). A planner that has to construct a URL only
needs `…/artifacts/{device}/properties/{name}`.

## Two crawler contracts

Read out of `integration_engine.py`, not assumed. Both require the type triple in
the **containing** document:

- `_process_workspace_recursive` (line 1075) recurses into a contained resource
  only if `(sub, rdf:type, hmas:Workspace)` is in the *parent's* graph.
- `_map_artifacts` (line 1138) accepts a contained resource as an artifact only if
  `(art, rdf:type, hmas:Artifact)` is in the *workspace's* graph.

So the home document types each room inline, and each room document types its
devices inline. `verify_phase2.py` crawls under exactly these guards, so a
regression here fails the suite rather than surfacing as a mysteriously empty
discovery.

## What senses and what actuates what

Two questions a planner must answer from the graph, and now can:

```sparql
# what can raise this room's illuminance?
?art td:hasActionAffordance ?a .
?a tdsosa:hasEffectActuation ?act ; td:hasForm/hctl:hasTarget ?href .
?act sosa:actsOnProperty <…/kitchen#illuminance> .

# who can tell me this room's temperature?
?sensor sosa:observes <…/utility_room#air_temperature> .
```

**Sensing.** SimuHome ships no standalone sensors — all 16 device types are
appliances or actuators — but six families carry a measurement cluster, and
measured over all 600 episodes they split absolutely, with zero exceptions:

| family | reads | feature of interest |
|---|---|---|
| `air_conditioner`, `heat_pump` | `Thermostat.LocalTemperature` | the **room** (1,988 instances) |
| `dehumidifier`, `humidifier` | `RelativeHumidityMeasurement.MeasuredValue` | the **room** (1,532) |
| `freezer`, `refrigerator` | `TemperatureMeasurement.MeasuredValue` | their own **interior** (1,275) |

The first four are `sosa:Sensor`s observing the room's own observable property,
with `sosa:isObservedBy` stated in the inverse direction a planner queries. The
last two observe an `#interior` feature of interest of their own: a freezer reads
−15 °C inside a 23.6 °C kitchen, so recording it as an observer of the kitchen
would be a lie. `illuminance` and `pm10_mass_concentration` have no sensor
anywhere — for those the room property is the only source.

**Actuation.** The 103 approved `actuation_effects` rows attach to the action
affordances as `tdsosa:hasEffectActuation`. That table originally enumerated only
cluster *commands*, so attribute **writes** were never candidates — yet the
simulator's own `control_rules` list several as required or optional
participants. `FanControl.FanMode` on an air purifier is one: writing it
genuinely helps clear PM10, but the affordance carried no claim, hiding it from a
planner searching for ways to move that variable. Five of seven families
under-claimed; only the two light families were complete. The missing rows are
now generated from `control_rules` and marked approved with that provenance. Direction follows the settled rule —
On/Off commands carry one, value-carrying setters do not — and because an action
here is keyed by *attribute* (`onOff(true|false)` drives all six On/Off commands),
a direction survives only where every command driving that attribute agrees. An
action never claims a variable both rises and falls.

These are **hints, not recipes**: they narrow the search to commands that can move
a variable. Working out that cooling needs `SystemMode=3` *plus* a setpoint below
current is the planner's job, and is what the benchmark measures.

**Reconciled** against the simulator's own causal model
(`GET /api/environment/control_rules/{state}`) — exact on all four states, on two
episodes with different device sets, and at **per-action** granularity. The
device-type comparison alone was too coarse: it passes as soon as one action of a
family claims the effect, which is how the under-claiming went unnoticed.

## The worker model

One worker owns a simulator and an SHTD on ports it allocates itself, and runs
many episodes with `reset` between them. Workers are **fully isolated**, clocks
included — `Home` is module-level state in the simulator process, so its tick
counter, simulated time, tick interval and room states are all per-process.

Measured with two workers side by side at different tick rates:

```
clocks           107 vs 9 ticks (tick_interval 0.1 vs 1.0)
simulated time   13:20:22 vs 15:12:30
layouts          22 vs 30 devices
cross-talk       actuating w0's kitchen light: 490.6 -> 990.6 lx;
                 w1's kitchen unchanged at 928.3 lx
```

**Subscriptions are cleared between episodes**, in two layers. `end_episode()`
first asks the agent's engine to unsubscribe properly — a real WebSub
`hub.mode=unsubscribe` per artifact, exercising the production path — and then
restarts SHTD regardless, since subscriptions live in that process and the agent
may have crashed without unsubscribing. This matters because a notification
listener binds a fixed port: without it, a later episode inherits the earlier
one's subscriptions and is woken by devices it never subscribed to. Observed
during phase-3 testing, where 22 stale subscriptions kept delivering into a new
run.

Verified across three episodes on one worker (with ports reused, and one episode
repeated): every episode starts at **0** subscriptions, all artifacts subscribe,
and unsubscribe returns the hub to **0**.

## Scoring

SimuHome's own evaluator scores the run, so results are comparable with the
published benchmark. Its scoring is **counterfactual**: it resets the simulator,
fast-forwards a baseline to the episode's final tick with no agent acting, and
asks whether each goal's variable moved further than it would have on its own.
So it needs the final home state captured first, and the simulator to itself
afterwards — which `evaluation_payload()` arranges.

Verified end to end: actions dispatched **through SHTD** satisfy a real episode's
goals and the evaluator returns **score 1**, with `study_room` illuminance
201.26 → 701.26 and `living_room` 360.66 → 860.66.

`sim_client.py` gained `reset_simulation()` and `fast_forward_to()` because the
evaluator calls those by name, and both return the raw envelope it expects.

## Dispatch coverage

`verify_dispatch.py` is static — it reads the benchmark and the dispatch tables,
and needs no running server. That is the point: an episode only exercises the
devices it contains, so a missing mapping stays invisible until some other seed
loads. `TemperatureControl.TemperatureSetpoint` was unmapped for exactly that
reason — **2,156 instances** across freezers, refrigerators and laundry washers,
advertised in every one of their Thing Descriptions and returning HTTP 400 when
invoked.

It also pins the command argument names against SimuHome's own cluster handlers,
which disagree both with the Matter spec and with each other:

| command | SimuHome accepts |
|---|---|
| `MoveToLevel` | `Level` (the one PascalCase case) |
| `GoToLiftPercentage` | `lift_percent_100ths` |
| `SetTemperature` | `target_temperature` (centi-degrees) |
| `ChangeToMode` | `new_mode` — but **`mode`** for `RTCCMode` and `RVCCleanMode` |

Verified by injecting each original bug and confirming the check fails.

**Idempotence.** `td:isIdempotent true` is asserted on every action, and it holds:
all 18 distinct actuatable attributes in the corpus were invoked twice, and the
second invocation left the value where the first put it. Structurally, actions
are keyed by *attribute* and always carry a target value, so only absolute
commands are ever dispatched — Matter's relative commands (`Step`, `Move`,
`SetpointRaiseLower`), the ones that would break this, are unreachable. Exposing
one would require revisiting the flag.

## Place vs environment

A room is **two** resources, and conflating them is a modelling error:

| resource | is | who links to it |
|---|---|---|
| `…/{room}#place` | `ex:Kitchen`, `s4bldg:BuildingSpace` | **every** device, via `ex:locatedIn` |
| `…/{room}#environment` | `sosa:FeatureOfInterest` | only devices that perceive or affect one of its variables, via `sosa:hasFeatureOfInterest` |

The environment is what carries air temperature, relative humidity, illuminance
and PM10 — the things sensors observe and actuators change. Being *in* a room is
a separate fact from observing its air.

The freezer makes the distinction concrete: it sits in the kitchen and reports
**−15 °C** while the room is at **23.6 °C**. It measures its own compartment, so
linking it to the kitchen's environment would be factually wrong, not merely
redundant. Membership comes from the approved `actuation_effects` table, not from
whether a device happens to carry a temperature attribute — freezers and
refrigerators do.

On `qt2_feasible_seed_77`: 22/22 devices located, **10** observing the
environment (air purifiers, lights, dehumidifier, heat pump).

## Stateless projection

SHTD holds no world. Every request re-reads `GET /api/home/state`, so
`POST /api/simulation/reset` on the simulator switches the episode and SHTD
follows on the next request with no invalidation step. Measured: a reset takes
**0.13s**, and the projection tracked a 22-device / `study_room` home to a
30-device / `kids_room` home with no restart.

This is what makes the phase-5 worker model work: one long-lived simulator and one
long-lived SHTD per worker, looping episodes with `reset` between them.

## Verified

`verify_phase2.py`, **21/21** against a live pair, on two different episodes:

1. **Recovery** — artifact count == device count (22/22, then 30/30). Spot-checked
   on `qt2_feasible_seed_88`, chosen because it is the corpus's worst case for the
   HA path: **all 12** of its air conditioners, heat pumps and window coverings
   appear, correctly typed `saref:HVAC` / `saref:Actuator`, with 9–38 properties
   each.
2. **Containment** — one home workspace plus one per room.
3. Every artifact is a `td:Thing` linked to its room's `sosa:FeatureOfInterest`.
4. **Property coverage** — affordance count matches the classifier exactly
   (199, then 358); no attribute is dropped or invented.
5. Every form target resolves on this server, and sampled reads return 200.
6. Every room carries all four environmental properties.

`SupportedModes` labels survive intact — `{"label": "Heavy", "mode": 2, …}` —
which closes the Phase B item HA could not carry.

## Value descriptions

Schemas describe **shape only** — type, unit, scale, permitted values. The
reading comes from dereferencing the property's form, never from the TD, because
the simulation ticks continuously and a cached document would freeze a stale
instant.

Two machine-derived `js:description` sentences make values legible to an LLM
planner. Both are standard WoT JSON Schema terms; no new vocabulary was minted.

**Enum values** — the wire value is always the integer; the name is only how a
planner recognises which integer it wants:

```turtle
td:hasOutputSchema [ a js:IntegerSchema ;
    js:enum 0, 1, 3, 4, 5, 6, 7, 8, 9 ;
    js:description "0 = Off (the Thermostat does not generate demand for Cooling
      or Heating). 1 = Auto (demand is generated for either Cooling or Heating,
      as required). 3 = Cool (demand is only generated for Cooling). … 7 = FanOnly." ] ;
```

The spec's per-value `summary` is included where it says something the name does
not. Values with no summary get the bare name (`7 = FanOnly`), and summaries that
are only a cross-reference into the PDF spec (`"(see Terms)"`) are dropped by the
normalizer rather than emitted as a dangling reference.

This closes a real gap: SimuHome's own `control_rules` require `SystemMode = 3`
for cooling, and without this an agent would have to recall from pretraining that
3 means Cool.

**Per-device limits** — a bound constrains a value, so it is a schema
*restriction*, in the standard WoT JSON Schema vocabulary rather than prose:

```turtle
td:hasOutputSchema [ a js:IntegerSchema ;
    ex:matterType "temperature" ;
    ex:unit "°C" ;
    js:minimum 5.0 ;
    js:maximum 95.0 ] ;
```

Sourced from the device's own `MinTemperature`/`MaxTemperature`,
`MinLevel`/`MaxLevel` and `Min/MaxMeasuredValue` attributes, converted out of
centi-units where the Matter type says so. These are per-instance — one washer's
range is not another's — and Home Assistant could not carry them at all.

Those bound attributes are **not** also exposed as readable properties. A bound
is a constraint, not state; publishing `minTemperature` as its own affordance
would state the same fact twice and leave a validator with nothing to act on.
`classify.py` gives them the role `bound`, and they reach the TD only through the
schema of the attribute they bound.

Dropped for the same reason: measurement `Accuracy` (a nested list of
`AccuracyRanges` structs), `NumberOfMeasurementTypes`, and LevelControl's
`Min/MaxFrequency`, which are degenerate (0/0) because the feature is not
implemented in this simulator. No planner acts on any of them.

Measured on `qt2_feasible_seed_77`: 174 property affordances, 32 schemas carrying
enum descriptions, 9 carrying `js:minimum`/`js:maximum`.

Enum extraction was added to `matter_model/normalize_model.py` (`_cluster_enums`)
and reached through `registry.enum_items()`. Enums inherit through derived
clusters the same way attributes do — `DishwasherMode` picks up Mode Base's
`ModeTag`. 95 of 132 clusters define enums, 277 definitions in total.

## Units

Matter stores temperature and humidity in centi-units. The TD reports the human
value and says so explicitly rather than leaving it implicit:

```turtle
ex:currentValue 2.362e+01 ;
ex:matterRawValue 2.362e+03 ;
ex:matterScale "centi" ;
ex:unit "°C" ;
```

## Mapping tables

The Matter→semantics layer comes from the approved Phase A tables, not from this
code. `mappings.py` honours a row only when its `status` is `approved` or `auto`;
anything still under review is reported by `/_shtd/status` as
`unsettledMappingRows` rather than silently used. Currently **zero** outstanding
(`ha_binding.yaml` is skipped — SHTD replaces it with direct Matter calls).

The Phase A mapping tables remain under `tests/simuhome/td/mappings/`, as does
the Home Assistant YAML converter — that is the HA path, which SHTD replaces.
The registry moved here because SHTD is its primary consumer.

See also `tests/simuhome/td/docs/effects-vs-simulator.md` for why the actuation
effect claims are narrowed to what SimuHome actually simulates.
