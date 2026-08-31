# Phase 6 — Semantic classes for properties and commands

> **STATUS: implemented.** 174/174 property affordances and 53/53 action
> affordances on `qt2_feasible_seed_77` carry a class; 0 `homeont:mechanism`,
> `homeont:sosaRole` or `homeont:matterScale` triples remain; `COMMAND_TARGETS`
> is deleted; 0 `ex:` CURIEs anywhere. See "What was built" at the end.

Phase 4 said what an affordance *affects*. Phase 6 says what an affordance *is*.

Today a device property affordance carries `td:name "hvacMode"`, a schema, a
form, and nothing else. Its meaning is recoverable only by reading the string
and knowing Matter. Every action affordance is worse: name, idempotence,
mechanism, schema, form, plus a `tdsosa:hasEffectActuation` in the 103 cases the
approved effects table covers — and for the rest, nothing at all.

The room's four environmental properties are the one place this is already done
right (`td_builder.py:904`): each is typed `homeont:AirTemperature`,
`homeont:RelativeHumidity`, `homeont:Illuminance`, `homeont:Pm10MassConcentration`, and those
four classes carry `rdfs:label`, `rdfs:comment`, a QUDT quantity kind and unit
in `homeont.ttl`. **Phase 6 extends that treatment to device properties and to
commands.**

## The constraint that governs every decision here

`homeont.ttl` is not the SimuHome ontology. It has to serve **Home Assistant
deployments and directly-attached Matter devices** as well, and those back ends
have no clusters, no `attribute_map.yaml`, and no notion of a "mechanism".

So the rule for this phase:

> **Whatever is specific to SimuHome stays in the mapper. What reaches the
> Thing Description must be formal, standard, and inferable from typing alone.**

A term earns its place in a served TD only if a consumer holding *just* the TD
and the imported ontologies can act on it without knowing which back end
produced it.

**One namespace, one prefix: `homeont:`.** There is no separate `ex:`
vocabulary — `td_builder.py:52` binds `HOME` to
`http://example.org/homeont/`, and `_NS_BY_PREFIX` resolves `ex:` to that
same namespace purely as a legacy alias, because the Phase A tables were
written with it. Serialized documents already carry `homeont:`. Phase 6
finishes the job: the tables are rewritten to spell it `homeont:`, the `"ex"`
alias is dropped from `_NS_BY_PREFIX`, and no new row or class is ever written
with `ex:`. Two spellings for one namespace is how a term ends up looking like
an example placeholder in an ontology that is meant to be normative. Anything that fails that test is either promoted to a class in a
standard vocabulary, or it stays behind in `mappings.py` / `classify.py` and
never reaches RDF.

This is already the shape the agent layer expects. `integration_engine.py:298`
collects `affordance_semantic_types` as *every* `rdf:type` on an affordance
except the `td:PropertyAffordance` marker itself, and the same at line 397 for
actions — a deployment-agnostic channel that carries whatever classes a back end
asserts. The UA prompt (`prompts.py:382`) then tells the planner to match on
"**semantic_types** (look for domain types in the homeont: namespace)".

The channel exists, the planner is already instructed to read it, and on device
properties SHTD currently puts **nothing** into it. That, not the unused yaml
tables, is the real defect: the generic path is starved while two bespoke
strings carry the meaning.

## The evidence: two approved tables that nothing reads

This is not a modelling gap so much as a wiring gap. The classes were curated in
Phase A and then never connected.

| table | holds | rows | read by |
|---|---|---|---|
| `property_classes.yaml` | generic, cross-family property classes — `homeont:ThermostatHvacMode`, `homeont:OnOff`, `homeont:LevelControlBrightness`, each `subClassOf: sosa:ActuatableProperty`, each with its `specialised_by` leaves | 30 | **nothing** |
| `command_targets.yaml` | the 14 curated command→attribute links | 14 | **nothing** |
| `actuatable_properties.yaml` | per-family leaves — `homeont:AirConditionerOnOff subClassOf homeont:OnOff` | 63 | `mechanism`/`sosa_role` only |

`mappings.py:77` loads `property_classes` into `self.property_classes` and no
call site ever touches the dict. `command_targets` is not loaded at all —
`classify.py:44` carries a hand-written `COMMAND_TARGETS` constant whose comment
says it "mirrors the curated table", which means the table and the code can now
drift silently in either direction.

Meanwhile `attribute_map.yaml` has **86 observable affordances with no class in
any table** — `activePower`, `operationalState`, `supportedModes`,
`currentFanSpeed`, and the three that carry genuine measurements
(`temperature`, `humidity`). These are not filtered out: they become served
`td:PropertyAffordance` nodes like any other (§C measures 121 of them on a
22-device home).

So a class exists for the 30 actuatable generics, is absent for every
observable, and reaches the served graph for none of them.

## What Phase 6 must do

### A. Mint the classes in `homeont.ttl`

Every class the tables name becomes a real term with `rdfs:label` and an
`rdfs:comment` written as a natural-language description — the comment is what
an LLM planner reads, so it states what the property *means* and what moving it
does, not what Matter calls it.

Three tiers, mirroring what the tables already assert:

```turtle
:ActuatableDeviceProperty a owl:Class ;
    rdfs:subClassOf sosa:ActuatableProperty .

:OnOff a owl:Class ;                          # generic, from property_classes
    rdfs:label "On/Off"@en ;
    rdfs:comment "Whether a device is powered on. Most devices refuse every
other operation while off, so turning a device on is usually the first step of
any plan that involves it."@en ;
    rdfs:subClassOf :ActuatableDeviceProperty .

:AirConditionerOnOff a owl:Class ;            # leaf, from actuatable_properties
    rdfs:label "Air Conditioner On/Off"@en ;
    rdfs:comment "Whether the air conditioner is running. While off it neither
cools nor reports a useful local temperature."@en ;
    rdfs:subClassOf :OnOff .
```

Counts to mint: **30 generic** + **59 leaves** (actuatable), plus the observable
classes from §C. The leaf tier is what a device's TD instantiates; the generic
tier is what a planner queries when it wants "anything that can be switched on".

Four rows carry `subClassOf: null` and need a generic parent that does not yet
exist: `Thermostat.OccupiedCoolingSetpoint` and `OccupiedHeatingSetpoint`, on
`AirConditioner` and `HeatPump`. Their `property_type` is already
`homeont:AirTemperature` with `quantity: air_temperature`, which is the right
instinct — a cooling setpoint is a *target for* air temperature — but
`homeont:AirTemperature` is an `sosa:ObservableProperty`, so it cannot also be the
class of an actuatable setpoint. Mint `:TemperatureSetpoint` (subclass of
`:ActuatableDeviceProperty`, carrying `quantitykind:ThermodynamicTemperature`)
and relate it to the observable it targets rather than conflating the two.

This is the clearest case for §B's removal of `homeont:sosaRole`: the table already
draws the distinction as `quantity_target` vs `quantity_result`, but as an
opaque string on the affordance. Promoting it to the class — an actuatable
setpoint that targets an observable quantity — makes it queryable and lets the
string go.

### B. Emit them on property affordances, and retire the two string stand-ins

`_attribute_property` (`td_builder.py:545`) gains one lookup and one triple, and
**loses two**:

```turtle
[] a td:PropertyAffordance, tdsosa:ActuatablePropertyAffordance,
     homeont:AirConditionerOnOff ;
   td:name "onOff" .
```

Resolution is leaf-first: `(family, cluster.attribute)` →
`actuatable_properties.property_type`; falling back to the generic
`property_classes` row when the family has no leaf; and typing only the tiers
that exist rather than inventing one. The room-state path already does exactly
this at `td_builder.py:904` — reuse the shape, not a second mechanism.

**`homeont:mechanism` and `homeont:sosaRole` come out**, and the reason is the
generalization rule, not tidiness. Both are string annotations that no standard
vocabulary defines, both encode a distinction the typing will express properly,
and one of them is not even about the device:

| triple | emitted at | verdict |
|---|---|---|
| `homeont:sosaRole "device_state"` | `td_builder.py:573` | **replaced by the class.** `saref:State` vs `sosa:ObservableProperty` vs `ssn-system:SystemCapability` (§C) is the same distinction in standard vocabularies, drawn finer and drawn *correctly* — `sosa_role` calls 81 of 85 observables `device_state`. A class is queryable and reasonable-over; a string is neither. |
| `homeont:mechanism` on a property | `td_builder.py:567` | **belongs in the mapper.** `tdsosa:ActuatablePropertyAffordance` already says the property can be actuated; the form says where to POST. Whether SHTD gets there by cluster command or attribute write is SimuHome-internal routing. Home Assistant would have to invent a value for this field, and a native Matter binding would populate it with something a third back end could not interpret. |
| `homeont:mechanism` on an action | `td_builder.py:617` | same, and superseded by the action's `saref:Command` subclass (§D) plus `saref:actsUpon`. |

`homeont:mechanism` is the clearest failure of the rule: it is **not a fact about the
device at all**, but about how one particular server dispatches to one
particular simulator. `homeont.ttl`'s own comment says as much — "how SHTD
actuates this property against the simulator". A vocabulary meant to cover HA
and native Matter cannot carry a term whose definition names SHTD. And the
docstring on `_action_affordance` already states the split "is invisible to the
planner by design", while the triple publishes it on every actuatable property
and every action.

Removal is safe: nothing consumes either triple. Verified by grep across
`ami_agents/` and `tests/` — written in those three places, read nowhere.
Dispatch resolves `mechanism` from the classification record server-side
(`shtd.py:301`), never from the graph; `verify_dispatch.py` and `inspector.py`
read the yaml row, not the RDF. `classify.py` keeps returning it — the mapper
still needs it — it simply stops reaching the graph, which is exactly the
mapper/TD boundary this phase is drawing.

Both `owl:DatatypeProperty` definitions come out of `homeont.ttl` with the
triples. `sosa_role` stays in the mapping tables as the provenance for how a
class was chosen; that is a curation record, not a published term.

### B2. The other back-end-specific terms

Applying the rule consistently turns up more than the two strings. `homeont.ttl`
also publishes a Matter provenance block, emitted on schemas:

| term | emitted at | verdict |
|---|---|---|
| `homeont:matterScale "centi"` | `td_builder.py:838, 932` | **remove.** It says the *simulator* stores hundredths while the TD reports the human value — a statement about SimuHome's wire format, not the quantity. The TD already reports the converted value with a `qudt:unit`; a consumer needs nothing more, and a HA-sourced TD would never carry it. |
| `homeont:matterType`, `homeont:matterEntryType` | `td_builder.py:834, 865, 870` | **keep, but re-justify.** The README defends `matterType` because `power-mW` and `uint32` are both `js:IntegerSchema` and only one is milliwatts. That is a real gap — but the fix is `qudt:unit unit:MilliW` on the schema, which is standard and back-end-neutral. Keep `matterType` only where it survives that substitution; drop it where a QUDT unit says the same thing better. |

Decide these in phase 6 rather than later: they are the same defect as
`homeont:mechanism` (a term whose meaning is "what the Matter back end happened to
do") and leaving them in would mean the rule was applied to two triples and
abandoned for four. Anything genuinely needed for round-tripping belongs in the
mapper's own tables, which already record cluster and attribute per row.

### C. Class the observables — 121 of the 174 affordances

**These do reach the TD.** Everything `classify.py` calls `role == "affordance"`
becomes a `td:PropertyAffordance` with a name, schema and GET form
(`td_builder.py:519-535`); only `drop`, `bound` and `thing_metadata` are
filtered. Measured on `qt2_feasible_seed_77`:

| | count |
|---|---|
| property affordances served | **174** |
| of which actuatable (30 classes exist, unused) | 53 |
| of which observable (**no class anywhere**) | **121** |

So the unclassed group is the majority of what a planner reads, not a tail.

The existing `sosa_role` string cannot be reused as the class axis: across 200
episodes it labels **81 of 85** distinct observables `device_state`, lumping the
capability lists (`SupportedModes`, `SpinSpeeds`, `PhaseList`) and the ~20
electrical measurements (`ActivePower`, `RMSVoltage`) in with genuine state like
`OperationalState`. Phase 6 needs a finer axis, and it must be anchored in SAREF
and SSN rather than minted here.

**The parents, from the published vocabularies:**

`ssn:Property` (`http://www.w3.org/ns/ssn/Property`) is the superclass of both
`sosa:ObservableProperty` and `sosa:ActuatableProperty` — so the whole tree has
one root already, and nothing needs inventing at the top.

| group | n (distinct) | parent | why that parent |
|---|---|---|---|
| **measurements** — `Thermostat.LocalTemperature`, `RelativeHumidityMeasurement.MeasuredValue`, `TemperatureMeasurement.MeasuredValue`, plus the electrical family | 3 with a declared `quantity`, ~20 more electrical | `sosa:ObservableProperty` + `qudt:hasQuantityKind` | exactly what the four room variables already do (`homeont.ttl:390-416`). A measured quantity of a feature of interest is the textbook `sosa:ObservableProperty`. |
| **device state** — `OperationalState`, `MediaPlayback.CurrentState`, `WindowCovering.SafetyStatus`, `PowerSource.Status` | ~50 | `saref:State` | SAREF defines `State` as "identifiable conditions that features of interest are or may be in, and that can be acted upon by devices" — precisely these. It ships `OnOffState`, `OpenCloseState`, `StartStopState`, `MultiLevelState` as subclasses, which cover `onOff`, the window covering and the level properties directly. |
| **capability declarations** — `SupportedModes`, `SpinSpeeds`, `SupportedRinses`, `SupportedDrynessLevels`, `PhaseList`, `OperationalStateList`, `ChannelList`, `FanModeSequence` | 8 | `ssn-system:SystemCapability` | these enumerate what the device *can* do, so a planner reads them to pick a legal value for a sibling actuatable. They are not readings and must not be typed as `sosa:ObservableProperty`. `internal_properties.yaml` already reaches for this namespace (`ssn-system:hasSystemCapability/ssn-system:MeasurementRange`), so the choice is consistent with what was approved in Phase A. |
| **sensor metadata** — `RelativeHumidityMeasurement.Tolerance` | 1 | `ssn-system:Accuracy` | SSN-systems has a named class for it. The other bound/accuracy attributes never become affordances (they are `bound`/`drop`), so this group is a single row — confirm rather than assume. |

A capability list typed as a state, or a setpoint typed as a measurement, is the
error this section exists to prevent — so `verify_phase6.py` asserts the group,
not merely the presence of some class.

This wants a new approved table — `observable_property_classes.yaml`, generated
by `convert.py --phase-a` alongside its siblings and reviewed row by row — not
ad-hoc classes in the builder. It carries, per row: the `yaml_path`, the minted
`homeont:` class, its `subClassOf` parent from the table above, the `rdfs:label`, the
natural-language `rdfs:comment`, and where applicable `quantityKind`/`qudt_unit`.

### D. Class the commands

`command_targets.yaml`'s 14 rows say which attribute a command writes. What they
do not say is what the command *means*, which is the same gap one level up.

An action affordance here is keyed by attribute, not by command
(`onOff(true|false)` stands for On/Off/Toggle), so the class belongs on the
action and describes the *operation*. **SAREF already defines this hierarchy** —
`saref:Command` is "the lowest-level directives a function exposes to some
network; commands can act upon features, properties, or states" — with the
concrete subclasses this corpus needs:

| affordance | SAREF class | note |
|---|---|---|
| `onOff` | `saref:OnCommand` / `saref:OffCommand` / `saref:ToggleCommand` | but the affordance is value-carrying, so it is a `saref:Command` acting on a `saref:OnOffState`, not a bare `OnCommand` — the same reasoning that made the *effect* directionless in phase 4 |
| `brightness`, `fanSpeed`, `position` | `saref:SetAbsoluteLevelCommand` | subclass of `saref:SetLevelCommand`. Absolute is right and provable: phase 3 established only absolute commands are ever dispatched — Matter's relative `Step`/`Move` are unreachable, which is also why `td:isIdempotent true` holds |
| `temperatureSetpoint`, `coolingSetpoint`, `heatingSetpoint` | `saref:SetAbsoluteLevelCommand` acting on a `:TemperatureSetpoint` | the level being set is the target quantity from §A |
| `currentMode`, `hvacMode`, `runCurrentMode`, `cleanCurrentMode` | `saref:Command` + `:ChangeModeAction` | SAREF has no mode-selection command; this is the one place a local subclass is genuinely needed, and it must say so |

Reuse SAREF's `saref:actsUpon` to point the command at the state or property it
drives — that is the relation the vocabulary provides, and it makes the
command→attribute link from `command_targets.yaml` explicit in the graph instead
of implicit in the shared `td:name`.

Two things this must respect, both already settled elsewhere in the codebase:

1. Do not re-derive command→attribute from the registry. Load
   `command_targets.yaml` in `mappings.py` and have `classify.py` read it
   instead of `COMMAND_TARGETS`, deleting the constant. The drift risk is the
   whole reason the table was approved.
2. Do not let the action class restate the effect. `tdsosa:hasEffectActuation`
   already carries "this can move air_temperature"; the action class carries
   "this is an absolute level write". Overlapping them would give a planner two
   sources for one fact.

### E. Wire the tables

- `mappings.py`: add `command_targets` loading; add lookups
  `property_class(family, cluster, attribute)` and `action_class(...)`; keep the
  `status in {approved, auto}` gate so an unreviewed class cannot reach a TD.
- `classify.py`: read the table, drop the constant.
- `td_builder.py`: emit the class triples; delete the three `HOME.mechanism` /
  `HOME.sosaRole` `g.add(...)` calls (lines 567, 573, 617). `classify.py` keeps
  returning `mechanism` — `shtd.py` needs it to route dispatch — it simply stops
  reaching the graph.
- `homeont.ttl`: drop the `:mechanism` and `:sosaRole` datatype-property
  definitions once nothing emits them (see §B for the one condition to keep
  them).
- **prefix cleanup**: rewrite the `ex:` CURIEs in the mapping tables to
  `homeont:` (mechanical — same namespace, so no IRI changes and no regeneration
  of curated evidence), then delete the `"ex"` entry from `_NS_BY_PREFIX`
  (`td_builder.py:76`) so a stray `ex:` fails loudly instead of silently
  resolving.
- `verify_phase6.py`: for a live episode, assert every property affordance and
  every action affordance carries a class; that every emitted class IRI resolves
  in `homeont.ttl`; that every class there has both `rdfs:label` and
  `rdfs:comment`; and that the leaf→generic→SOSA chain closes. A class emitted
  but not defined, or defined but unlabelled, is the failure mode that makes
  this invisible again.

## Acceptance

On `qt2_feasible_seed_77` (22 devices, **174** property affordances — 53
actuatable, 121 observable, counted by running `classify.py` over the episode):

- 174/174 property affordances carry an `rdf:type` beyond
  `td:PropertyAffordance` and the tdsosa marker.
- every emitted class reaches one of the four §C parents (`sosa:ObservableProperty`,
  `saref:State`, `ssn-system:SystemCapability`, `ssn-system:Accuracy`) or
  `sosa:ActuatableProperty`, through a chain that closes in `homeont.ttl`. A
  capability list typed as a reading fails here.
- every action affordance carries an action class rooted in `saref:Command`.
- 0 emitted class IRIs undefined in `homeont.ttl`; 0 defined classes missing
  `rdfs:label` or `rdfs:comment`.
- `COMMAND_TARGETS` no longer exists in `classify.py`.
- **0 occurrences of `ex:` anywhere** — tables, code, docs. One namespace, one
  prefix.
- **0 `homeont:mechanism`, 0 `homeont:sosaRole`, 0 `homeont:matterScale` triples in any served
  document** — the planner's context carries semantic classes, not dispatch
  routing or wire format.
- action invocation still passes `verify_phase3.py` and `verify_dispatch.py`
  unchanged, proving the removal was presentational only.
- **the generalization check**: every predicate and every class in a served TD
  is drawn from `td`/`js`/`hctl`/`sosa`/`ssn`/`ssn-system`/`saref`/`qudt`/`hmas`
  or from `homeont`, and every `homeont` term has a definition that would hold
  for a Home Assistant or native-Matter deployment. Mechanically: no term whose
  `rdfs:comment` names SHTD, SimuHome, or Matter as the *reason* it exists. This
  is the criterion that keeps the mapper/TD boundary from eroding again.
- `affordance_semantic_types` (`integration_engine.py:298`) is non-empty for
  174/174 property affordances — today it is empty for every device property,
  which is the concrete symptom this phase fixes.
- `mappings.py:unsettled()` still reports zero outstanding rows, now including
  the new table.

---

## What was built

**171 classes minted** in `homeont.ttl` (+3 roots and 2 command classes), every
one carrying `rdfs:label` and an `rdfs:comment` written for an LLM planner.
All 171 reach a standard root; verified mechanically.

| tier | n | parent |
|---|---|---|
| generic actuatable | 30 | `homeont:ActuatableDeviceProperty` → `sosa:ActuatableProperty` |
| per-family leaves | 59 | their generic class |
| observable measurements | 19 | `sosa:ObservableProperty` + QUDT quantity kind |
| observable device state | 47 | `homeont:DeviceStateProperty` → `saref:State` |
| capability declarations | 19 | `homeont:DeviceCapabilityProperty` → `ssn-system:SystemCapability` |
| measurement tolerance | 1 | `ssn-system:Accuracy` |
| commands | 3 | `saref:Command` (+ `saref:SetAbsoluteLevelCommand` reused directly) |

**New table**: `observable_property_classes.yaml`, 86 rows, each recording the
parent, the rationale, and the `sosa_role` it replaced.

### Two defects found while wiring, both fixed

1. **The `COMMAND_TARGETS` drift was real, not hypothetical.** The constant
   carried four cluster entries the table lacked, and the table was missing a
   `WindowCovering.CurrentPositionLiftPercent100ths` row that `shtd.py`
   dispatches through. Swapping naively would have silently de-actuated every
   window covering's position and every freezer's cabinet mode. The missing row
   is now approved with that evidence, and the table is joined on **cluster id**
   rather than a munged display name — the previous string-stripping could never
   match `RTCCMode` to `Refrigerator And Temperature Controlled Cabinet Mode`.
2. **Four setpoints were typed as the observable they target.** The AC and heat
   pump cooling/heating setpoints carried `homeont:AirTemperature` — an
   `sosa:ObservableProperty` — as the class of an *actuatable*. They now take
   `homeont:{Family}{Cooling,Heating}Setpoint` under
   `homeont:TemperatureSetpoint`, and `verify.py` asserts no actuatable is ever
   typed as a room observable.

### Verification

| check | result |
|---|---|
| property affordances typed | **174/174** |
| action affordances typed | **53/53** |
| banned triples in served docs | **0** |
| homeont terms used but undefined | **0** of 128 |
| terms whose comment names a back end | **0** |
| `verify_dispatch.py` | 3/3 — 18 actuatable attributes all dispatchable |
| `tests/simuhome/td/verify.py` | PASSED, incl. byte-identical regeneration |
| `pytest tests/unit` | 199 passed, 8 failed — **identical to the pre-existing baseline** |

### Deliberately not done

The generic class IRIs still echo Matter cluster names
(`homeont:LevelControlBrightness`, `homeont:FanControlFanSpeed`). Renaming 171
curated IRIs is a table-wide migration well beyond this phase and would churn
every approved row; the names are opaque identifiers whose meaning now comes
from `rdfs:label` and `rdfs:comment`. Worth a follow-up if the vocabulary is
published.

`homeont:matterType` was kept (§B2 proposed dropping it where a QUDT unit says
the same thing). It still distinguishes `power-mW` from `uint32` on schemas
whose unit is not otherwise stated; narrowing it needs a per-schema audit of
which attributes carry a QUDT unit, which is its own piece of work.
