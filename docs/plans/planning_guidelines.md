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

## Part 2 — condition guards, decided after planning and remembered

### What it does

Every entry of `config/planning_guidelines.yaml` is now a **condition**: the device state
that must hold before a command, per device class (and command class, and optionally the
property the command writes). There is no `kind`; restriction and guideline are one thing.
The conditions are listed in the file, each with its Matter source in a comment.

| Device class | Command | Condition | Established by |
|---|---|---|---|
| LaundryWasher, LaundryDryer, Dishwasher, AirConditioner, DimmableLight (ColorLight) | mode / level / op-state | switched on | the deterministic power-on guard |
| LaundryWasher, LaundryDryer, Dishwasher | programme, spin, rinses, dryness, temperature | operational state ≠ Running | pause, then resume after |
| LaundryWasher, LaundryDryer, Dishwasher | pause / resume / start | Running or Paused (pause, resume); settings set before a start | no guard can establish it |
| AirConditioner, HeatPump (cooling); + Heater (heating) | the setpoint | HVAC mode Cool / Heat or Auto | set the HVAC mode |
| WindowCoveringController | target position | Mode 0 (no calibration / maintenance bit) | set the mode to 0 |
| Rvc | clean mode, run mode non-Idle → non-Idle | run mode Idle | set the run mode to Idle |

`power_on_first` stays deterministic (`utils/power_on_guard.py`). Every other condition goes
through the **condition-guard check**, on simple and dependency plans:

1. **Steps** (`utils/planning_conditions.py`): each action of the plan whose device, command
   and written property match a condition. Actions inside a guard already there (a
   replayed tree) are skipped.
2. **Memory** (`utils/guard_memory.py`, `GuardStore` in the EnvExplorer): each step's exact
   key (action title and class, device title and class, room class, and the value for
   commands whose value selects the command). A hit is never applied blindly
   (`condition_guard.apply_remembered`): its property and action are found again on
   today's TD by title and class, its values re-typed, its restore derived again. If that
   fails, the hit counts as a miss.
3. **The model** (`utils/condition_check.py`, `CONDITION_GUARD_PROMPT`): one call to the
   profiling model for all misses. It gets the request, the plan's actions in execution
   order, the steps with their condition texts, and the devices' TDs. Per step it answers
   a guard (check property, `==` / `!=`, value; establishing action and value) or no
   guard (`not_applicable`, or `established_by_plan`, which skips the guard). The answer is
   resolved against the step's own device; anything wrong means no guard, with a note.
4. **Restore** (`condition_guard.restore_action`): a guard that switches off, pauses or
   stops is followed by the switch-on, resume or start. That is the same combined action
   with the opposite value, or the device's one separate on / start command (a tie is
   broken by name). If none is found, the step is not guarded.
5. **Wrap** (`condition_guard.wrap_steps`):
   ```
   selector "<step>: <condition> condition guard"
     sequence holds      condition(check), step
     sequence establish  condition(not check), guard action, step, [restore]
   ```
6. **Drop** (`condition_guard.drop_satisfied`): a guard whose check an earlier step of the
   plan already establishes, in execution order, is removed again. The walk is
   `utils/plan_order.walk_in_order`, shared with the power-on guard:
   - a sequence carries facts from child to child;
   - selectors, parallels and decorators don't carry facts past themselves;
   - a repeat's body starts without the keys it writes;
   - timers and waits pass facts through. The plan is trusted, which is an accepted risk:
     a change from outside the plan during a wait is not seen.

   For dependency plans the drop runs on the compiled tree, where "earlier" is the real
   order across goals.

### Where it runs

- **Simple plans:** `ConditionGuardBehaviour`, spawned by `GoalRequestBehaviour._plan_mixed`
  before `PowerOnGuardBehaviour`, which then also covers the guards' own actions. Both
  share one TD graph.
- **Dependency plans:** step 3c of `DependencyGoalPlanningBehaviour`, after profiling, on
  every goal subtree and per-device tree. Step 4b then drops the satisfied guards on the
  compiled tree, before the power-on guard.
- **The entry:** `guards`, with one decision per step (`dropped` marks a guard the plan
  didn't need), and `condition_call`, the model call.

### Learning

- **Messages:** `GUARD_MATCH_REQUEST` / `RESPONSE` and `GUARD_RECORD_REQUEST` / `RESPONSE`.
- **EnvExplorer:** `GuardMatchBehaviour` and `GuardRecordBehaviour`. The `GuardStore` is
  opened with the `MemoryStore`.
- **UA:** `user_assistant/guards.py`. After a successful plan it records the model's
  guards that stayed in the tree, and its `not_applicable` answers. It never records
  memory hits, dropped guards or `established_by_plan`, which depends on one plan, not on
  the device. The restore is not stored.
- **Bench:** `stubs.Explorer.query` answers `GUARD_MATCH` from the run's `GuardStore`, and
  `_remember` records the decisions by the UA's rule.

### Verification (handed over)

- `run --episode qt4-3_feasible_seed_41 --memory run`: the dryness-level step should get a
  condition question, and the guard should be recorded after a successful run.
- Re-running with the same memory: the guard is applied with no question.
