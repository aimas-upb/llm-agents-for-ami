# Plan: planning restrictions and guidelines per device category (ISA)

## Context

Several qt4-3 episodes failed on things a device knows but its Thing Description does not say:
- an appliance must be powered on before a cycle can start;
- settings must be chosen before the start;
- a cycle's length is not known before it starts.

The planners had no way to learn these except by being refused.

You want this knowledge to live in a **dedicated, file-configured section of the ISA**:
- text-based **restrictions** and **guidelines** for planning;
- given **per device category**;
- **sourced from Matter (or Home Assistant)** specifications, never from SimuHome, which is
  only one instance.

Part 1, below, checks which of the three items the specifications actually support, and for
which categories. Part 2 is the design, to be settled after your answers.

## Part 1 — what the specifications say (researched)

Sources:
- Matter Application Cluster Specification R1.2 (CSA 23-27350, Oct 2023) and R1.3.0.1
  (Apr 2024), official PDFs from csa-iot.org;
- Matter Device Library 1.4.2 (CSA 23-27351-007, Jul 2025);
- the Matter 1.5.1 data model vendored in
  `ami_agents/environment/integration/SimuHome/matter_model/vendor/1.5.1/device_types/*.xml`;
- Home Assistant's Home Connect integration documentation.

### 1. "Powered on before a cycle can start" — normative in Matter (Dead Front)

- **On/Off cluster, DeadFrontBehavior feature (DF)**, App. Cluster Spec R1.2 §1.5.4.2: "When
  this feature is supported, the device exposing this server cluster exhibits "dead front"
  behavior when the "OnOff" attribute is FALSE (Off). This "dead front" behavior includes:
  clusters other than this cluster that are also exposed MAY respond with failures to Invoke
  and Write interactions. Such failure responses when in a "dead front" SHALL be with an
  INVALID_IN_STATE status code."
- **Device Library 1.4.2**, for each category below: "the DF (Dead Front) feature is required
  for the On/Off cluster in this device type … the Off command of the On/Off cluster SHALL
  move the device into the "dead front" state, the On command of the On/Off cluster SHALL
  bring the device out of the "dead front" state". The 1.5.1 XML agrees: On/Off is optional,
  and DF is mandatory when it is present.
- **Categories:**
  - Laundry Washer (§13.1)
  - Room Air Conditioner (§13.3)
  - Dishwasher (§13.5)
  - Laundry Dryer (§13.6)
- **Not mandated** for: Oven, Microwave Oven, Cooktop, Refrigerator, Robotic Vacuum Cleaner, Fan.
- **Also relevant**, Operational State Start, R1.2 §1.13.6.3: the device "SHALL start its
  operation if it is safe to do so and the device is in an operational state from which it
  can be started"; otherwise it answers `UnableToStartOrResume` or `CommandInvalidInState`.
- **Strength:** "MAY respond with failures". Matter permits the refusal in dead front; it does
  not require it. A safe guideline is therefore "switch on before any other command".

### 2. "Settings chosen before the start" — permitted by Matter, but the states are vendor-defined

- **Laundry Dryer Controls**, R1.3 §8.9.5.2, SelectedDrynessLevel: "If an attempt is made to
  write this attribute while the device is not in a state that supports modifying the dryness
  level, an INVALID_IN_STATE error SHALL be sent as the response."
- **Laundry Washer Controls**, R1.2 §8.6.6.3, NumberOfRinses: "If the device is not in a
  compatible state to accept the provided value, an INVALID_IN_STATE error SHALL be sent."
- **Mode Base** (Laundry Washer / Dishwasher / … Mode), ChangeToMode status `InvalidInMode`:
  "The received request cannot be handled due to the current mode of the device."
- **Strict only for robot vacuums:** RVC Clean Mode, R1.3 §7.3.8, "SHALL NOT permit changing
  its mode while the RVC Run Mode cluster's CurrentMode attribute is set to a mode without the
  Idle mode tag" (also RVC Run Mode §7.2.9).
- **Strength:** Matter says a setting MAY be refused in some states. Which states is up to the
  manufacturer, except for robot vacuums. Choosing settings before starting is the order that
  is never refused for this reason.
- **What "a state that supports modifying" / "a compatible state" means:** Matter does **not**
  define it for laundry or dishwasher settings; it is left to the manufacturer. "Running" is
  not named as the incompatible state.
  - The closest thing is an example in Temperature Control, R1.3 §8.2.6.1.3: "e.g. due to the
    design of a device it cannot accept a change in its temperature setting after it has begun
    operation". It is an example, not a rule.
  - Oven / Microwave SetCookingParameters (R1.3 §8.13.6.2.6) is equally generic: "while the
    operational state of the server cannot support the command in that state".
- **Where Matter does define state compatibility:** for the lifecycle commands, not for
  settings. Operational State, R1.3:

  | Command | Running | Paused | Stopped | Error |
  |---|---|---|---|---|
  | Pause (Table 3) | compatible | compatible | not | not |
  | Resume (Table 4) | compatible | compatible | not | not |

  An incompatible Pause is answered `CommandInvalidInState`. Derived clusters and manufacturers
  may add their own states. Start: "if it is safe to do so and the device is in an operational
  state from which it can be started".

  This is normative and applies to every device with Operational State (washer, dryer,
  dishwasher, oven, microwave). It is a candidate guideline: "pause only a running cycle;
  resume only a paused one".
- **Robot vacuums** are the one category where Matter fixes a settings restriction to a state
  (§7.2.9 / §7.3.8, non-Idle run mode).
- **Categories:** Laundry Dryer, Laundry Washer, Dishwasher (through their Mode / Controls
  clusters); Robotic Vacuum Cleaner (strict).

### 3. "A cycle's length is not known before it starts" — Matter: generally true

- **Operational State CountdownTime**, R1.2 §1.13.5.3: "the estimated time left before the
  operation is completed … When this attribute is null, that represents that there is no time
  currently defined until operation completion. This MAY happen, for example, because no
  operation is in progress or because the completion time is unknown."
- **Device Library 1.4.2:** CountdownTime falls back to null in "dead front" (washer, dishwasher,
  dryer).
- **Exception — Device Energy Management, Forecast** (R1.3 §9.2.8.7, provisional): an appliance
  may share its intended forecast, whose slots carry `DefaultDuration`, "the expected time (in
  seconds) that the appliance expects to be in this slot". It is null if "a program has not yet
  been selected". So the length can be known before the start, but only on devices composed
  with Device Energy Management (optional for appliances).
- **Home Assistant (Home Connect integration):** "Finish time … available only if the program is
  running". But the appliance itself accepts **"Finish in relative"** (Dryer, Washer, Washer
  dryer) and **"Start in relative"** (Oven, Dishwasher): the device aligns the end itself.
  This is vendor-specific, not Matter.
- **Category:** any device with Operational State (Laundry Washer, Laundry Dryer, Dishwasher,
  Oven, Microwave Oven, …).
- **Not from a specification:** the per-setting durations, e.g. 45 min for a level. Those are
  SimuHome's own values and stay out.

### 4. Do SAREF or SSN model "an action depends on another action or a device state"? — No

Checked against SAREF core v3.2.1 (`ami_agents/shared/ontologies/saref.rdf`) and the W3C SSN /
SOSA recommendation (2017).

- **SAREF core:**
  - **No property links a command to a precondition.** Commands relate only through
    `actsUpon`, `hasInput` / `hasOutput`, `isCommandOf` and `hasCommandKind`; functions
    through `hasCommand`, `hasMandatoryCommand` and `hasOptionalCommand`.
  - The only conditional properties are **`saref:whenState`** and
    **`saref:whenPropertyValue`**. Their domain is `saref:Profile` (e.g. an energy profile),
    not a command: "Links a profile to a state that contributes to restricting its applicable
    context." They say in which state a profile applies, not that an action requires a state.
  - The lifecycle commands (Start / Pause / Stop) carry no ordering or prerequisites.
- **SSN / SOSA:**
  - Actuation is `sosa:Actuation` with `sosa:usedProcedure`, and `ssn:Procedure` has
    `ssn:hasInput` / `ssn:hasOutput`. None of these is a precondition.
  - SSN-System's `ssn-system:Condition` and `ssn-system:inCondition` describe the prevailing
    (environmental) conditions for operating / survival ranges ("operates as specified when
    humidity < 80 %"). That is a capability under conditions, not "this action requires that
    state or that prior action".
- **Closest existing vocabularies elsewhere, for a later formal model:**
  - OWL-S: `process:hasPrecondition` / `hasResult` (with effects);
  - WSMO: preconditions, assumptions, postconditions, effects;
  - for durations (item 3), SAREF4ENER's power-sequence slots, the counterpart of Matter's
    energy forecast.
  - Neither SAREF core nor SSN covers prerequisites, so a TD-level model would need one of
    these or a new property.

## Part 2 — protection guards, decided at profiling and remembered

### What it does

Some actions are refused unless the device is in a certain state. Three examples:
- a dryness level can't be changed while a cycle runs;
- a cycle can't start while the device is off;
- a pause is refused unless the cycle is running.

For each such action, the plan should be able to put a guard in front of it:
**check a property; if it is not the expected value, first do a protecting action**. Then
the action follows:
```
sequence
  selector                                   (the idempotent guard)
    condition: protection_check_property == protection_check_value
    action:    protection_check_action (with its value)
  achieve(G)                                 (the goal's own action, unchanged)
```
Whether a goal needs a guard, and which one, is decided **during profiling**. That is the
step where the devices and the commands are known. The ISA then **remembers** the decision
for that exact action of that exact device, so later plans apply it without asking.

### Settled with you

| Topic | Decision |
|---|---|
| Guidelines file | entries keyed by **device class** (artifact class name, e.g. `homeont:LaundryDryer`, matched with its subclasses) or by **manufacturer** (the TD's `manufacturer`); each holds plain-language text |
| Which goals are asked | goals whose action is a `homeont:SetModeCommand`, `saref:SetAbsoluteLevelCommand` or `homeont:SetOperationalStateCommand`, on a device matching an entry |
| Which structures | both dependency **and simple** structures (simple ones are routed through profiling for this question only) |
| Memory | its own store in the memory engine (not signifiers), hosted by the EnvExplorer |
| When recorded | after the plan ran successfully (the UA records it, as for signifiers) |
| What is recorded | both decisions: a guard, and "no protection needed" |

### 1. The guidelines file — `ami_agents/config/planning_guidelines.yaml`

The path is in `agents.yaml` (`interaction_solver.planning.guidelines.path`).
```yaml
entries:
  - id: settings_before_start
    applies_to:
      classes: [homeont:LaundryDryer, homeont:LaundryWasher, homeont:Dishwasher]
      manufacturers: []
    text: >-
      Choose mode and settings (programme, dryness, spin, rinses) before starting a cycle;
      while a cycle runs they may be refused.
  - id: power_on_first          # classes: washer, dryer, dishwasher, room air conditioner
  - id: pause_only_running      # classes: every appliance with a cycle
```
- The three draft entries come from Part 1. `pause_only_running` is new.
- **Loader:** `interaction_solver/utils/planning_guidelines.py` reads the file once, with a
  module default path. `guidelines_for(graph, artifact)` returns the texts whose class (or a
  superclass of the device's class) or manufacturer matches the device.

### 2. Which goals get the question — `goal_profile.build_checklist` (deterministic)

A new question, `protection`, with target `G`, is asked for every goal that meets all of:
- its action's command class is one of the three above (read from the planned subtree's
  action, in the TD);
- its device matches at least one guideline entry;
- the guard memory has no record for its key (§5). A hit is applied as recorded, and no
  question is asked.

The key is: `affordance_name`, `affordance_class`, `device_name` (artifact title),
`device_class` (most specific class), `room_class` (its workspace's class), all taken from the
TD of the device the subtree acts on.

### 3. The profiling prompt — a new `protection` section in `GOAL_PROFILE_PROMPT`

This adds a section to your profiling prompt. Its context, per target goal:
- the device's actions and properties (as today);
- the matching guideline texts;
- the goal's action and value.

The answer names the property and the action by their `td:title` **and their class**, as
they appear in the goal's context:
```json
{"protection": false}
{"protection": true,
 "protection_check_property": {"title": "<td:title>", "class": "<property class>"},
 "protection_check_value": "<value as text>",
 "protection_check_action": {"title": "<td:title>", "class": "<command class>",
                             "value_text": "<value as text>"}}
```

**Resolution — deterministic, against the real graph** (`resolve_profile` +
`utils/dependency_context.py`):
- **The property:** a SPARQL query on the TD graph looks for a property affordance of the
  goal's device that has that `td:title` and is of that class (`rdf:type/rdfs:subClassOf*`).
- **The action:** the same query, for an action affordance of the same device with that title
  and command class.
- **Both found:** the guard is created, from the property's read target and the action's
  invoke target (their `hctl:hasTarget`). Values are typed from the property's output schema
  and the action's input schema.
- **Either missing, or the answer has the wrong shape:** the question resolves to
  `{"protection": false}`, with a note naming what was not found.

**The memory** stores the resolved guard: title, class and target of the property and of the
action, plus both values. It is applied from there without another query.

### 4. Applying the guard

- **Dependency structures:** the compiler wraps `achieve(G)` in the pattern above, where the
  profile (or the memory) says so. Values are typed from the property and action schemas, as
  the existing `_typed` does.
- **Simple structures:** after the explicit / incomplete planners have planned them, a new
  OneShot behaviour applies the same step to the plan entry:
  - `ProtectionBehaviour(entries, graph)` in `interaction_solver/behaviours/protection.py`;
  - it builds the checklist (protection questions only), asks the small model once per
    structure, and wraps each tree.
  - It is spawned from `GoalRequestBehaviour._plan_mixed` after the route behaviours.
  - Ambiguous (LLM-built) trees have no single goal action and are not covered.
- **The entry** carries `guards: [{key, protection, check?, action?}]` for every goal decided,
  from the memory or from the model, so the UA can record them.

### 5. The guard memory — its own store, exact key

- **`GuardStore`** in `ami_agents/shared/memory/src/storage/guard_store.py`:
  - records on disk under `<storage>/guards/`, the same way as `MemoryStore`;
  - `match(key) -> record | None`, `record(record) -> id`, `list`, `clear`;
  - the key is the five fields above, digested exactly;
  - the record holds `protection`, plus for a guard the check property (URL, name, value)
    and the action (URL, name, value).
- **Messages** (`shared/models/messages.py`): `GUARD_MATCH_REQUEST` / `RESPONSE` and
  `GUARD_RECORD_REQUEST` / `RESPONSE`.
- **EnvExplorer:** `GuardMatchBehaviour` and `GuardRecordBehaviour`, mirroring the signifier
  match / record behaviours but sharing no code with them.
- **ISA:** `utils/guard_memory.py`, `recall_guard(agent, key)` over RPC, used by the checklist
  step.
- **UA:** after a plan entry with `guards` executed successfully, sends one
  `GUARD_RECORD_REQUEST` per guard decided by the model; guards that came from the memory are
  not re-recorded. "Successfully" means the plan concluded with success, and the guard's own
  check / action did not fail.
- **Bench:** the in-process explorer answers the two messages from a `GuardStore` in the run
  directory (`--memory run`), like signifiers.

### Files

| Area | Files |
|---|---|
| config | `config/planning_guidelines.yaml` (new); `agents.yaml` (path key); `docs/configuration.md` |
| ISA | `utils/planning_guidelines.py` (new); `utils/goal_profile.py` (question, shape check, fallback); `prompts.py` (protection section); `utils/dependency_compiler.py` (guard wrap); `behaviours/protection.py` (new); `behaviours/goal_request.py` (spawn it); `utils/guard_memory.py` (new) |
| memory | `shared/memory/src/storage/guard_store.py` (new) |
| messages | `shared/models/messages.py` |
| EnvExplorer | `behaviors/guard_match_behaviour.py`, `behaviors/guard_record_behaviour.py` (new), registered in the agent |
| UA | recording after a successful plan, next to the signifier recording (`plan_execution.py` / `signifiers.py` call site) |
| bench | `stubs.Explorer.query` (two messages), `memory.py` |

### Tests (unit, files touched only)

- guideline matching by class, subclass and manufacturer;
- checklist: asked only for the three command kinds on matched devices, and not on a memory hit;
- `resolve_profile`: protection shapes; malformed answers fall back to false;
- compiler and `ProtectionBehaviour`: the wrap pattern, and no wrap for false;
- `GuardStore` round trip and exact key;
- EnvExplorer match / record;
- UA records only after success, both true and false, never re-recording memory hits.

### Verification (handed over)

- `run --episode qt4-3_feasible_seed_41 --memory run --context prepend`: the dryness-level goal
  should get a protection question; the guard should be recorded after a successful run.
- Re-running with the same memory: the guard is applied without a question.
