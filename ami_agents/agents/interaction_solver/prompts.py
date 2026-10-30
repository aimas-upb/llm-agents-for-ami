"""LLM prompts of the InteractionSolver (string constants only)."""


# Asked of the small model for a "modify" goal whose read-compute-set recipe
# could not be read off the Thing Descriptions alone. It names the pieces; the
# InteractionSolver checks them against the TDs and builds the tree itself.
MODIFY_RECIPE_PROMPT = """
You complete the plan for ONE relative change to ONE smart-home device ("dim the light by 20%", "lower the setpoint by 2 degrees").

The change is applied as: read the current value of a property, compute the new value, set it with an action. You only name the pieces; you do not write a plan.

You receive the structured goal and the device's affordances (actions with their input, properties with their output).

Return ONLY one JSON object:
{"action": "<action name>", "parameter": "value" | "<parameter name>", "property": "<property name>",
 "amount": <number>, "min": <number> | null, "max": <number> | null}

- action: the action that sets the value being changed, by its name.
- parameter: "value" when the action's input is a single value; otherwise the name of the input parameter that takes it.
- property: the property holding the current value of what the action changes, by its name.
- amount: the change from the goal, keeping its sign (positive increases, negative decreases). For a value from an ordered list (its schema has an "order", for example a fan speed low, medium, high), the amount is the number of steps along that list.
- min, max: the range the new value must stay in, from the property's or the action's schema; null if none is stated.

If the device has no action and property that can make this change, return {"impossible": "<why, in one sentence>"}. Never invent a name.
"""


# The profiling call of the dependency planner (`behaviours/dependency_goal_planning.py`):
# the small model answers the checklist built by `utils/goal_profile.py`, from
# the TDs of each goal's and predicate's artifacts. The three inputs are put in
# by plain replacement (`goal_profile.render_goal_profile_prompt`): the text
# quotes JSON.
GOAL_PROFILE_PROMPT = """
You complete a dependency goal structure so that it can be turned into a behavior tree by fixed templates. A structurer has already turned the user's request into goals and predicates, and every goal and predicate has already been resolved to artifacts on its own. What is still missing is how the goals depend on each other in terms the artifacts can actually report: which event a dependency refers to, and which properties the plan must read to know when it happens. You fill this in by answering a short list of questions.

You do not plan anything. You never choose commands or actions, and you never change goals or predicates. You only answer the questions you are given, using the Thing Descriptions in the context.

## Inputs

- `goal_structure`: the dependency structure (goals G1, G2, … and predicates P1, P2, …), with each goal's `goal_kind`, `timing` and `ends`, and each predicate's `role`, `applies_to` and offsets.
- `specific_goal_context`: the Thing Descriptions (TDs) of the artifacts resolved for each goal and predicate. Each TD gives the artifact's homeont class, its `td:title`, the place it is `homeont:isLocatedIn`, and its property and action affordances, each with a homeont class, a `td:title` and an input or output schema.
- `checklist`: the questions to answer. Each has an `id`, a `question` type, a `target` (a goal or predicate id) and, where relevant, a `ref` (the goal it depends on).

The checklist is built from the structure by fixed rules, described below under each question, so that you know why each question is asked. Answer every question in the checklist and no others.

## Shared types

**PropertyRef** points at one property affordance in one TD. Every value is copied from the TD; nothing is made up or paraphrased:

```json
{"location_class": "homeont:...", "artifact_class": "homeont:...", "artifact_name": "<artifact td:title>",
 "property_class": "homeont:...", "property_name": "<property td:title>", "property_field": "NA"}
```

- `location_class`: the homeont class of the place the artifact `homeont:isLocatedIn`, not the generic `homeont:BuildingSpace`.
- `artifact_class`: the artifact's homeont class, not generic types such as `hmas:Artifact`, `saref:Actuator` or `td:Thing`.
- `artifact_name`: the artifact's `td:title`.
- `property_class`: the property affordance's homeont class, not `td:PropertyAffordance` or the `tdsosa:` affordance kind.
- `property_name`: the property affordance's `td:title`.
- `property_field`: the field name when the output schema is an object and the value is one of its fields; otherwise `"NA"`.

**Condition** is a check on one property:

```json
{"ref": PropertyRef, "comparison": "==" | "!=" | ">" | ">=" | "<" | "<=", "target_value_text": "..."}
```

`target_value_text` must be a value the property's output schema allows: `"true"` or `"false"` for a boolean schema, one of the `jsonschema:enum` values for an enum, a number within `jsonschema:minimum` and `jsonschema:maximum` for an integer.

**TimeSource** names the property the plan reads, each time the behavior tree is ticked, to get a time value. You never give the value itself, only where it is read from and what kind of value it is:

```json
{"ref": PropertyRef, "reads_as": "duration" | "timestamp"}
```

- `duration`: the property holds an amount of time, such as a remaining time or a program length.
- `timestamp`: the property holds the clock time at which something happens.

You do not give a unit. The unit comes with the property's homeont class.

## The questions

### `maintenance_profile`

**Asked for:** every goal with `goal_kind: "maintenance"`.

**Means:** how the goal's value is kept over time. A maintenance goal has no end of its own, so the plan must either rely on the artifact to hold the value, or read the value repeatedly and act again when it drifts.

**Decide:**
- `native`: the artifact has an actuatable property (`tdsosa:ActuatablePropertyAffordance`) that holds a target for exactly the quantity the goal keeps, and it regulates to that target by itself (a thermostat's target temperature, for example). Setting it once is enough.
- `regulated`: the artifact can only be switched or stepped, or the goal keeps a state that someone or something else can change (keep a light on, keep a door locked). The plan must read the value and act again when it no longer holds.

**Answer:**

```json
{"mode": "native", "setpoint_property": PropertyRef, "observed_property": PropertyRef | null, "tolerance_property": null}
{"mode": "regulated", "observed_property": PropertyRef, "tolerance_property": PropertyRef | null}
```

- `setpoint_property`: the actuatable property that holds the target value on the regulating artifact.
- `observed_property`: the property that shows whether the goal currently holds. For an environment quantity, this is a property of a sensor artifact that observes that quantity in the same place. For an artifact state, it is the artifact's own property. Required for `regulated`, optional for `native`.
- `tolerance_property`: a property that holds the allowed deviation (a deadband or hysteresis property), read when the plan runs. `null` if the TD has none; never give a number instead.

### `dependency_anchor`

**Asked for:** every goal with `timing.starts: "after_goal"` whose `ref_goal` is an achievement goal. (A dependency on a maintenance goal is checked in code; you are never asked about it.)

**Means:** "after G1" has two possible meanings. G1's plan finishes as soon as its action has taken effect. But if that action starts something that continues afterwards (a program, a cycle, a level transition), "after G1" may mean after that has ended.

**Decide:**
- `goal_success` (the default): the dependent goal starts once the referenced goal's action has taken effect. Use it when the referenced goal does not start anything that continues, or when nothing in the request or the goals points to waiting for its end.
- `condition`: the dependent goal starts when what the referenced goal started has ended. Use it when either:
  - the request's words point at the end ("when it's done", "once it finishes", "after the fade"), or
  - the dependent goal only makes sense after the end (drying after washing, unloading after a dishwasher cycle).

  The condition must read a property of the referenced artifact that reports that end (a state, a remaining time, a finished flag). If the TD has none, the question is unresolved.

**Answer:**

```json
{"anchor": "goal_success", "basis": "default"}
{"anchor": "condition", "condition": Condition, "basis": "text" | "inferred"}
```

- `condition`: the check that becomes true when what the referenced goal started has ended. Prefer a state property over a remaining-time property when both exist.
- `basis`: `text` if the request's words point at the end, `inferred` if you concluded it from what the goals do, `default` for `goal_success`.

### `completion_timing`

**Asked for:** every goal with `timing.completes_with`; its `ref` is the goal or event named there.

**Means:** the target goal's run must finish at the same moment as its `ref`. When the goal's timing has no `starts`, the plan chooses the start: on every tick it reads how long the ref still needs and how long the target goal will take, and starts the target goal once the first is no more than the second. When the timing has a `starts`, the start is the user's, and the same two values are read once to check that the two runs finish together. Either way, you name the properties that are read; you do not supply the times.

**Answer:**

```json
{"ref_remaining": TimeSource | null, "ref_duration": TimeSource | null, "target_duration": TimeSource}
```

- `ref_remaining`: the property that tells how long the ref goal still needs while it runs: a remaining time (`duration`) or an end time (`timestamp`).
- `ref_duration`: the property that tells how long the ref goal takes from its start, used only if no remaining-time or end-time property exists. The plan then counts down from the ref goal's start.
- `target_duration`: the property that tells how long the target goal takes from its start to its finish, for the setting the goal asks for (use `property_field` when the output lists one duration per setting).

At least one of `ref_remaining` and `ref_duration` must be filled. If no property in the TD gives `target_duration`, the question is unresolved.

The `ref` may also be an event instead of a goal — a trigger predicate such as "the oven timer runs out". Then "how long the ref still needs" is how long until that event happens: `ref_remaining` is a property of the event's device (its remaining time, or its end time), and `ref_duration` cannot be used, since the plan does not know when the event's run started.

### `time_to_event`

**Asked for:** every predicate with `role: "trigger"` and `offset_direction: "before"` ("1 minute before the light switches off"); a trigger or termination predicate on a goal whose start or end the user also gave as a clock time; and every predicate a goal's `timing.completes_with` names.

**Means:** the plan cannot wait for the event and then go back in time. It can only act early if a property tells it, ahead of time, when the event will happen. You name that property. For the other two cases the property is read once, before planning, to check the user's times against when the event will happen.

**Answer:**

```json
{"time_to_event": TimeSource}
```

- The property must count down to (or give the time of) the exact moment the predicate becomes true, not some other moment. Read the homeont class and title of each candidate and check what it measures.
- If no such property exists (the event is a person's action, or the artifact does not report it), the question is unresolved. Do not estimate.

## Rules

1. Every class in your answer is a homeont class taken from the TD, and every name is a `td:title` taken from the TD. Never write a descriptive name of your own.
2. Never answer with a time, a tolerance or a current reading as a number. Name the property the plan should read; its value is read when the tree is ticked. The only literal values you write are `target_value_text` in a Condition.
3. Give `evidence` for every answer: the artifact and affordance titles you relied on, with their classes, in a few words.
4. If the TDs do not contain what a question needs, put it in `unresolved` with the reason. An unresolved question is a correct answer; a guessed one is not. Unresolved questions are handled after you, so you do not choose any replacement.
5. If you notice that the structure itself is inconsistent (a goal depending on something that can never happen, for example), report it in `unresolved` with `"question": "inconsistency"` and do not try to repair it.
6. Do not change, restate or add goals or predicates. Your output contains only answers.

## Output

Return only this JSON:

```json
{
  "answers": [
    {"id": "Q1", "question": "...", "target": "...", "answer": {...}, "evidence": "..."}
  ],
  "unresolved": [
    {"id": "Q2", "question": "...", "target": "...", "reason": "..."}
  ]
}
```

Every checklist `id` appears exactly once, in either `answers` or `unresolved`.

## Examples

These show only the shape of the output. The classes and titles in them are illustrations; in your answers, use only classes and titles from the TDs in your input.

### Example 1

Request: "Fade the living room light down to 10, and once the fade is done turn on the hallway light."

Checklist:

```json
[{"id": "Q1", "question": "dependency_anchor", "target": "G2", "ref": "G1"}]
```

Output:

```json
{"answers": [
  {"id": "Q1", "question": "dependency_anchor", "target": "G2",
   "answer": {"anchor": "condition",
              "condition": {"ref": {"location_class": "homeont:LivingRoom", "artifact_class": "homeont:DimmableLight", "artifact_name": "living_room_dimmable_light_1",
                                    "property_class": "homeont:RemainingTime", "property_name": "remainingTime", "property_field": "NA"},
                            "comparison": "==", "target_value_text": "0"},
              "basis": "text"},
   "evidence": "living_room_dimmable_light_1 / remainingTime (homeont:RemainingTime) counts down the level transition; 'once the fade is done'"}],
 "unresolved": []}
```

### Example 2

Request: "Keep the living room light on until 11 PM, and turn on the porch light 1 minute before the living room light switches off."

Checklist:

```json
[{"id": "Q1", "question": "maintenance_profile", "target": "G1"},
 {"id": "Q2", "question": "time_to_event", "target": "P1"}]
```

Output:

```json
{"answers": [
  {"id": "Q1", "question": "maintenance_profile", "target": "G1",
   "answer": {"mode": "regulated",
              "observed_property": {"location_class": "homeont:LivingRoom", "artifact_class": "homeont:DimmableLight", "artifact_name": "living_room_dimmable_light_1",
                                    "property_class": "homeont:DimmableLightOnOff", "property_name": "onOff", "property_field": "NA"},
              "tolerance_property": null},
   "evidence": "living_room_dimmable_light_1 / onOff (homeont:DimmableLightOnOff): an on/off state anyone can change, no self-regulation"}],
 "unresolved": [
  {"id": "Q2", "question": "time_to_event", "target": "P1",
   "reason": "living_room_dimmable_light_1 has no property that reports, ahead of time, when it will switch off"}]}
```

## Input

goal_structure:
{goal_structure}

specific_goal_context:
{specific_goal_context}

checklist:
{checklist}
"""


# The condition-guard check (`utils/condition_check.py`), run on every plan after
# its steps are planned (after profiling, for dependency plans): for each step on
# a device that has a condition in `config/planning_guidelines.yaml`, the small
# model says which check and which protecting action guard it -- or that no
# guard is needed. The InteractionSolver resolves the answer against the TDs and
# builds the guard; a pause, stop or switch-off it names is followed by the
# matching resume, start or switch-on, added by code. The four inputs are put
# in by plain replacement (`condition_check.render_condition_guard_prompt`).
CONDITION_GUARD_PROMPT = """
You check a smart-home plan against the conditions its devices impose. A plan has already been made; some of its steps send a command to a device that only accepts that command, or only acts on it, when the device is in a certain state. For each such step you decide whether the plan must check that state first, and how to establish it when it does not hold.

You do not change the plan. You never remove, reorder or reword its steps. You only answer, for each listed step, with a guard or with "no guard".

## Inputs

- `request`: the user's request as structured goals (goal ids G1, G2, … with their timing, when there are several).
- `plan_actions`: every action of the plan, in execution order, with its step id where it has conditions: the goal it serves, the device, the action and the value it sends.
- `steps`: the steps to decide. Each has an `id`, the device (`device_name`, `device_class`), the action (`action_name`, `action_class`), the `value` it sends, and the `conditions` that apply to it, in plain language.
- `devices`: every device a step acts on, by name, with its affordances. Each affordance has a `kind` ("action" or "property"), a `name`, its classes (`semantic_types`), a `description`, and its `input_schema` (actions) or `output_schema` (properties). A check names a property; a guard action names an action — even when an action and a property share a name, pick by `kind`.

## How to decide

Read each condition literally, against the step's action and value. A condition often covers several commands (start, pause, resume; programme, spin, rinses); only the part about this step's command and value counts.

Interpret the plan exactly as it is. If the user changes a setting while a cycle runs, the plan changes it during the run: the guard pauses the cycle so the change is accepted. The plan's own steps are never the guard.

Answer **no guard** when:
- the condition does not concern this command or this value (`"why": "not_applicable"`), or the condition itself says not to guard it;
- an earlier action in `plan_actions` already puts the device in the required state before this step runs, in execution order (`"why": "established_by_plan"`) — for example the plan pauses the cycle, then changes the spin.

Otherwise answer **a guard**:
- `check`: the property of the step's own device that tells whether the condition holds, the comparison (`==` or `!=`) and the value. Choose the comparison that states the condition exactly: "must not be Running" is `!=` Running; "must be Cool or Auto" cannot be one comparison, so check the value you will set (`==` Cool).
- `action`: the action of the same device that makes the check hold, and the value to send. Prefer a pause to a stop: a paused cycle can be resumed. Never name the step's own action.

Values are written as the TD gives them: if the property's or action's schema is a number, give the number (an operational state is 0 Stopped, 1 Running, 2 Paused, as in Matter); if it is an enum of names, give the name; booleans as `true` / `false`.

Copy every name from the `devices` TDs: the property's and the action's `name` and one of their homeont / SAREF classes. Never invent a name. If the device has no property or no action that can express the condition, answer no guard with `"why": "not_applicable"` and say so in `reason`.

## Output

Return one JSON object, one answer per step id, no others. Every answer has all six fields: a guard has `"why": null`; no guard has `"check": null, "action": null`:

```json
{"answers": [
  {"id": "S1", "guard": true,
   "check":  {"property": "<property name>", "property_class": "<its class>", "operator": "!=", "value": 1},
   "action": {"action": "<action name>", "action_class": "<its class>", "value": 2},
   "why": null, "reason": "<one sentence>"},
  {"id": "S2", "guard": false, "check": null, "action": null,
   "why": "established_by_plan", "reason": "<one sentence>"}]}
```

### Example

Request: "set the washer's spin speed to 1200". `plan_actions`: S1 washer_1 spinSpeedCurrent 1200. Condition: "Before changing the wash programme, the spin speed, … : the operational state must not be Running (Stopped or Paused). If it is Running, pause the cycle first." The washer has a property `operationalState` (homeont:LaundryWasherOperationalState, integer) and an action `operationalState` (homeont:SetOperationalStateCommand).

```json
{"answers": [
  {"id": "S1", "guard": true,
   "check":  {"property": "operationalState", "property_class": "homeont:LaundryWasherOperationalState", "operator": "!=", "value": 1},
   "action": {"action": "operationalState", "action_class": "homeont:SetOperationalStateCommand", "value": 2},
   "why": null, "reason": "the spin speed is refused while the cycle runs; pausing it lets the change through"}]}
```

Same washer, request "pause the washer, then set the spin to 1200": S2 (the spin) follows "washer_1 operationalState 2" in `plan_actions`, so `{"id": "S2", "guard": false, "check": null, "action": null, "why": "established_by_plan", …}`.

A step whose condition says not to guard it is `"guard": false, "why": "not_applicable"`. Never guard a step with its own action.

## Input

request:
{request}

plan_actions:
{plan_actions}

steps:
{steps}

devices:
{devices}
"""
