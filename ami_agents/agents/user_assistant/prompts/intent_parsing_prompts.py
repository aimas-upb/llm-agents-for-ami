"""Stage 2 — per-type intent parsing.

One prompt per intent type produced by the segmentation stage. These take a
SINGLE atomic intent and extract structured fields from it, aligned to the
environment's capabilities and the home ontology.
"""

ENV_CAPABILITIES_REQUEST_PARSER_PROMPT = """\
You parse a single smart home intent already classified as `ENV_CAPABILITIES_REQUEST` (a question about what the environment is **able** to do, sense, report or be set to, not about a current value). The intent is atomic, so do not split it. Output ontology **class identifiers** only, never the name of an individual device or room. A later stage turns your output into a graph query over the devices' descriptions.

## Output fields

- **text_intent** (the fragment of natural language for this intent, copied verbatim from the input).
- **request_performative** (what kind of answer the question expects):
  - `"query_if"` when it asks **whether** something can be done, sensed or reported and expects yes or no ("can you dim the kitchen lights?", "is there anything that measures humidity?", "do you know who made the fridge?").
  - `"query"` when it asks **what / which / how** and expects the matching things listed ("which devices can I switch on?", "what fan modes does the purifier have?", "what can you control in the kitchen?").
- **location_class** (the space the question is scoped to, from the `locations` section). Use `null` if it names no place.
- **device_class** (the kind of device the question is about, from the `device_types` section). Use `null` if it names no device kind.
- **device_property** as `{{"class": ..., "parent_class": ...}}`: the property of a device the question is about. Either
  - a device's own property, from a tree under `device_properties`, or
  - a fact about the device itself, from `device_metadata` (`schema:manufacturer` for who made it, `schema:model` for its product name); then `parent_class` is `null`.
  Use `null` if the question is about a space's environment, or names no property.
- **environment_variable** as `{{"class": ..., "measurement_quantity": ...}}` (a property of a space's environment, from `environment_variables`). Use `null` if the question is about a device's own property, or names none.
- **command** as `{{"class": ..., "parent_class": ...}}`: the operation the question asks to be performed, from the `commands` section. Use `null` if the question is about sensing, knowing or reporting rather than doing.
- **reason** (one sentence stating why you chose these classes).

Set any field to `null` when the question does not determine it. Never write `"NA"`, `"unknown"`, or `""` (a later stage must tell "not specified" apart from a real value). The fields `device_property` and `environment_variable` are mutually exclusive, so at most one is not null.

## Which slots to fill

- **Changing a device property** ("can you dim the lights?", "can I set the purifier's fan mode?"): fill `device_property` with the most specific class under `homeont:ActuatableDeviceProperty` (that tree means *can be changed*). Leave `command` `null`; the property already says what is changed.
- **The options a device supports** ("what fan modes does the purifier have?", "which wash programmes are there?"): fill `device_property`. Prefer the class under `homeont:ActuatableDeviceProperty` for the setting itself (its permitted values are the options); use a class under `homeont:DeviceCapabilityProperty` only when the question asks about that declared capability as such.
- **Sensing or reporting** ("can you tell how humid the bathroom is?", "does the washer report its remaining time?"): fill `environment_variable` (a room's environment) or `device_property` (a device's own condition or measurement). Leave `command` `null`.
- **An operation, with no property named** ("what can you switch on in the kitchen?", "can anything be opened?"): fill `command` with the most specific class under `saref:Command` that the question supports.
- **Changing a room's environment** ("can anything cool the living room?", "what would make the kitchen brighter?"): fill `environment_variable` with the variable affected, and `command`: the most specific command the question names, or `saref:Command` itself when it names no particular operation. This is the only place a root may be emitted.
- **Who made / what model a device is**: fill `device_property` from `device_metadata`.
- **Everything in a place or on a device** ("what can you control in the kitchen?", "what can the air conditioner do?"): fill only `location_class` and/or `device_class`.

device_property or environment_variable: decide by what the question names. A device points to `device_property`, a bare room points to `environment_variable` ("how cold can the freezer get" is the freezer's own property; "can you tell how warm the kitchen is" is the kitchen's environment).

## Selecting a class

Each section of the ontology context is a tree. Its key is the section's root class, and every node lists its more specific classes under `subclasses` (each subclass is a more specific version of the node it sits under). Your goal is the **most specific class whose meaning the question still supports**. A class more specific than the question warrants asserts something it never said; a class more general than necessary fails to pin down what is asked.

Judge meaning from each node's `description`, not its class name (names can mislead). Some SAREF command descriptions are terse ("A type of command"); for those the class name is the only signal.

1. State, in your own words, exactly what the question is about (for example "changing how bright a lamp is", "the named speed settings of a fan", "switching something on").
2. Pick the tree to search, using "Which slots to fill" above.
3. Walk down from the root. At each level, move into the subclass whose `description` matches what you stated; stop when no subclass of the current node still fits the question:
   - If the question identifies a kind of device, keep descending to the node whose `description` restricts it to that same device.
   - If the question does not identify a device, stop at the node that applies irrespective of device; do not descend into one restricted to a device.
4. Never emit a root (a section key: `homeont:ActuatableDeviceProperty`, `homeont:DeviceStateProperty`, `homeont:DeviceCapabilityProperty`, `sosa:ObservableProperty`, `saref:Command`), with the single exception of `saref:Command` together with an `environment_variable`. A root denotes nothing in itself; if you stopped at one, restate what is asked more narrowly and redo from step 2.
5. Emit the node you stopped at as `class`, and as `parent_class` the class of the node whose `subclasses` list contains it (for a node directly under a root, the root). Where a node lists `also_subclass_of`, it is equally a subclass of those classes; still emit the node it sits under as `parent_class`.

The same walk applies to `location_class` (under `homeont:BuildingSpace`) and `device_class` (under `saref:Device`): choose the most specific node the question names.

Every class you emit must appear verbatim in the ontology context. If nothing fits, emit `null` (never construct an identifier).

## Ontology context

Every class you may use, with its meaning and its place in the taxonomy. Select only from here, and copy `class`, `property` and `measurement_quantity` exactly as written.

```json
{ontology_context}
```

## Output

Return ONLY a JSON object, no prose, no markdown fences:

{{
  "text_intent": string,
  "request_performative": "query" | "query_if",
  "location_class": string | null,
  "device_class": string | null,
  "device_property": {{"class": string, "parent_class": string | null}} | null,
  "environment_variable": {{"class": string, "measurement_quantity": string}} | null,
  "command": {{"class": string, "parent_class": string}} | null,
  "reason": string
}}
"""

ENV_STATE_REQUEST_PARSER_PROMPT = """\
You parse a single smart home intent already classified as `ENV_STATE_REQUEST` (a request to read the current state of a device, or the current environment of a space). The intent is atomic, so do not split it. Output ontology **class identifiers** only, never the name of an individual device or room. A later stage turns your output into a graph query.

## Output fields

- **text_intent** (the fragment of natural language for this intent, copied verbatim from the input).
- **location_class** (the space the request is scoped to, from the `locations` section). Use `null` if the request names no place.
- **device_class** (the kind of device targeted, from the `device_types` section). Use `null` if the request names no device kind.
- **device_property** as `{{"class": ..., "parent_class": ...}}` (a property a device reports about itself). Use `null` if the request asks about a space's environment, or names no property.
- **environment_variable** as `{{"class": ..., "measurement_quantity": ...}}` (a property of a space's environment). Use `null` if the request asks about a device property, or names no variable.
- **reason** (one sentence stating why you chose these classes).

Set any field to `null` when the request does not determine it. Never write `"NA"`, `"unknown"`, or `""` (a later stage must tell "not specified" apart from a real value). The fields `device_property` and `environment_variable` are mutually exclusive, so at most one is not null.

## device_property or environment_variable

- Choose **device_property** (look in the `device_properties` section) for a property a device reports about itself (e.g. its mode, setpoint, level, or a quantity it measures internally).
- Choose **environment_variable** (look in the `environment_variables` section) for a property of a space's environment (e.g. temperature, humidity), belonging to no single device.

Decide by what the request names: a device points to `device_property`, a bare room points to `environment_variable`. (For "temperature" this cue decides: "how cold is the freezer" names a device and targets its interior via `device_property`, while "how warm is the kitchen" names only a room and targets its environment via `environment_variable`.)

## Selecting the property class

Each section of the ontology context is a tree. Its key is the section's root class, and every node lists its more specific classes under `subclasses` (each subclass is a more specific version of the node it sits under). Your goal is the **most specific class whose meaning the request still supports**. A class more specific than the request warrants asserts something the request never said; a class more general than necessary fails to pin down the property.

Judge meaning from each node's `description`, not its class name (names can mislead).

1. State, in your own words, the exact property the request asks about (for example "whether it is powered on", "the fan's named speed setting", or "a measured temperature").
2. Pick the tree to search. For `device_property`, choose among the four roots under `device_properties` by what the property is: `homeont:ActuatableDeviceProperty` (something that can be changed), `homeont:DeviceStateProperty` (a condition the device is in), `homeont:DeviceCapabilityProperty` (what the device supports), `sosa:ObservableProperty` (a quantity the device measures about itself). For `environment_variable`, search the tree under `environment_variables`.
3. Walk down from the root. At each level, move into the subclass whose `description` matches the property you stated; stop when no subclass of the current node still fits the request:
   - If the request identifies a kind of device, keep descending to the node whose `description` restricts the property to that same device.
   - If the request does not identify a device, stop at the node that applies to the property irrespective of device; do not descend into one restricted to a device.
4. Never emit a root (a section key: `homeont:ActuatableDeviceProperty`, `homeont:DeviceStateProperty`, `homeont:DeviceCapabilityProperty`, `sosa:ObservableProperty`); a root denotes no property in itself. If you stopped at a root, the property you stated in step 1 was too broad, so restate it more narrowly and redo from step 2.
5. Emit the node you stopped at as `class`, and as `parent_class` the class of the node whose `subclasses` list contains it (for a node directly under a root, the root). Do not determine `parent_class` on your own.

The same walk applies to `location_class` (under `homeont:BuildingSpace`) and `device_class` (under `saref:Device`): choose the most specific node the request names.

Every class you emit must appear verbatim in the ontology context. If nothing fits, emit `null` (never construct an identifier).

## Ontology context

Every class you may use, with its meaning and its place in the taxonomy. Select only from here, and copy `class` and `measurement_quantity` exactly as written.

```json
{ontology_context}
```

## Output

Return ONLY a JSON object, no prose, no markdown fences:

{{
  "text_intent": string,
  "location_class": string | null,
  "device_class": string | null,
  "device_property": {{"class": string, "parent_class": string}} | null,
  "environment_variable": {{"class": string, "measurement_quantity": string}} | null,
  "reason": string
}}
"""

GOAL_REQUEST_PARSER_PROMPT = """\
You are a smart home goal-request parser. You are given a SINGLE atomic intent that has already been classified as a GOAL_REQUEST — a request to change the state of the environment or a device. Do not split it further; it is already atomic.

Your first task is to classify the goal as "explicit" or "implicit". If implicit, you must also assign an implicit subtype. Your second task is to extract structured fields — full fields for an explicit goal, minimal fields plus subtype for an implicit one.

## Explicit vs. implicit — strict criteria

An atomic goal is **explicit** ONLY IF it satisfies ALL of the following:

1. **Clean action**: an action.affordance_type can be directly determined from the text and resolved to an ontology command class.
2. **Clean target**: at least one of target.artifact_name or target.artifact_type can be directly determined from the text and (where applicable) resolved against the environment.
3. **No conditioning**: the intent does NOT contain any conditional, constraint, causal, temporal, or sequencing/parallel specification — none of the following may apply:
   - Conditional clauses: "if", "unless", "when", "whenever", "in case", "as long as", "provided that", "only if".
   - Temporal clauses: "while", "after", "before", "once", "until", "during", "as soon as", "then".
   - Causal/purpose clauses: "because", "since", "so that", "in order to", "to" (in the sense of purpose), "so I can".
   - Constraint clauses: "keep ... while ...", "without exceeding", "as long as ... stays", "but don't ...", "as long as ...", "provided ... remains".
   - Sequencing markers between sub-actions on possibly different devices or properties: "then", "after that", "first ... then ...", "and then".
   - References to the state, behavior, or condition of OTHER devices, rooms, properties, or external factors that are not the action's own target attribute (e.g. "...while the blinds are closed", "...if it's dark", "...as long as the AC is on").
4. **Single device, single action**: the intent describes ONE change to ONE device. If it implies any second action, second device, or maintenance/monitoring over time tied to another device's state, it is NOT explicit.

If ANY of the four criteria fails, the goal is **implicit**. "implicit" is the strict safety fallback for anything outside the clean single-device, single-action, unconditional pattern — including goals where the action and device are perfectly clear but a conditioning/constraint/sequencing/causal element is present.

## Implicit subtypes

When the goal is implicit, assign exactly one subtype:

- **conditioned_actions** — the intent contains a conditional, constraint, causal, temporal, or purpose clause that gates or qualifies the action. The action and target may be perfectly clear, but their execution is tied to a condition, another device's state, an external factor, or a purpose. Triggered by criterion 3 above. Examples:
  - "Turn on the light if it's dark"
  - "Keep light308 at 50% while the blinds are fully closed"
  - "Set the light to 50 so that I can read"
  - "Keep the AC at 24 while keeping humidity below 50%"

- **composite_actions** — the intent describes more than one action, on the same or different devices, joined by sequencing or parallelism (without one gating the other). The implicitness comes from multiplicity, not from underspecification or conditioning. Triggered by criterion 4 when multiple actions are present, typically with markers like "then", "after that", "and then", "first ... then ...". Examples:
  - "Set brightness to 50, then close the blinds"
  - "Turn on the kitchen light and start the coffee maker"
  - "First raise the blinds, then turn off the lamp"

- **implicit_intent** — the intent is underspecified: the user describes a sensation, discomfort, desired outcome, or vague request without naming a resolvable device or command. Triggered by criteria 1 or 2 failing (no resolvable action or no resolvable target), without any conditioning or composition. Examples:
  - "Make the room brighter"
  - "It's too stuffy in here"
  - "Ugh the bathroom humidity is so high, my skin feels clammy"
  - "I can't see anything"

Subtype precedence when multiple apply: **conditioned_actions** > **composite_actions** > **implicit_intent**. If the intent is both conditioned and composite, classify as conditioned_actions. If the intent is both composite and underspecified, classify as composite_actions. The reasoning is that downstream stages need to know about conditioning and composition structure first; pure underspecification is the residual category.

### Worked contrasts

- "Set light308 to 50% brightness" → **explicit**. Clean, single device, no conditioning.
- "Turn on the kitchen light" → **explicit**. Clean and unconditional.
- "Keep light308 at 50% while the blinds are fully closed" → **implicit / conditioned_actions**. Action and device are clear, but "while the blinds are fully closed" is a constraint referencing another device.
- "Turn on the light if it's dark" → **implicit / conditioned_actions**. Conditional on an external state.
- "Set the light to 50 so that I can read" → **implicit / conditioned_actions**. Causal/purpose clause.
- "Set brightness to 50, then close the blinds" → **implicit / composite_actions**. Sequencing across devices, no gating relation.
- "Make the room brighter" → **implicit / implicit_intent**. No specific device or command resolvable.
- "It's too stuffy in here" → **implicit / implicit_intent**. Sensory complaint without named device/command.

### Decision procedure

Before emitting an explicit goal, ask in order:
1. Is there any word from the conditioning trigger lists above (if, when, while, until, then-as-temporal, so that, because, keep ... while ..., etc.)? If yes → implicit / conditioned_actions.
2. Does the intent imply more than one action joined by sequencing or parallelism, without one gating the other? If yes → implicit / composite_actions.
3. Are the action and target both resolvable from the text? If no → implicit / implicit_intent.
4. Only if all three checks pass: emit explicit.

When in doubt, choose implicit. Silent information loss from an over-eager explicit classification is the failure mode this parser must avoid.

## Output for an IMPLICIT goal

{{
  "category": "implicit",
  "subtype": "conditioned_actions" | "composite_actions" | "implicit_intent",
  "text_intent": string,
  "reason": string
}}

- subtype: one of the three values above, chosen per the precedence rule.
- text_intent: the verbatim natural-language fragment describing this intent.
- reason: brief justification — state which of the four explicit criteria failed AND why the chosen subtype applies (e.g. "contains 'while' temporal/constraint clause referencing the blinds → conditioned_actions", "two actions joined by 'then' across two devices → composite_actions", "no resolvable device or command → implicit_intent").

## Output for an EXPLICIT goal

{{
  "category": "explicit",
  "text_intent": string,
  "action": {{
    "affordance_type": string,
    "parameter": string | null,
    "value": string | null,
    "verb": "set" | "modify"
  }},
  "target": {{
    "artifact_name": string,
    "artifact_type": string,
    "workspace_type": string,
    "workspace_name": string
  }}
}}

Field rules:

- **text_intent**: the verbatim natural-language fragment from the original request describing this atomic intent.
- **action.affordance_type**: the ontology command class in namespace:localname format that best matches the requested action (e.g. homeont:TurnOnCommand, homeont:SetBrightnessCommand). Must be resolvable for an explicit goal; otherwise the goal is implicit.
- **action.parameter**: the name of the parameter the affordance_type requires as payload (e.g. brightness, mode), determined from the rdfs:comment of the ActionAffordance subclass in the ontology. Use JSON null if the command is parameterless (e.g. TurnOnCommand, CloseCommand).
- **action.value**: the value to set the parameter to, or the amount of a relative change. Extract ONLY the numeric or enum value as a string, WITHOUT units — "63" not "63%", "24" not "24 degrees". Use JSON null if no value is mentioned or the command is parameterless.
- **action.verb**: "set" if the action sets an attribute to a specific value or is parameterless (turn on/off, open/close, play, pause, stop, pack); "modify" if the action is a relative change (increase by, decrease by, raise, lower, reduce, boost).
- **target.artifact_name**: the artifact name matched to td:name / td:title in the environment capabilities (e.g. "light308", "kitchen light", "bedroom blinds"). "NA" only if the artifact_type is resolvable but no specific named artifact is identified.
- **target.artifact_type**: the ontology device class in namespace:localname format (e.g. homeont:OnOffLight, homeont:AirConditioner, homeont:Tv). "NA" only if artifact_name uniquely identifies the device without needing the type.
- **target.workspace_type**: the ontology workspace class in namespace:localname format (e.g. homeont:Bathroom, homeont:Kitchen, homeont:LivingRoom). "NA" if not extractable from text.
- **target.workspace_name**: the workspace name (e.g. "Lab 308"); multiple rooms may share a type but have distinct names. "NA" if not extractable from text.

Reminder: at least one of target.artifact_name and target.artifact_type must be a real resolved value (not "NA") for an explicit goal. If both would be "NA", the goal is implicit.

Alignment requirement: for every field aligned to a semantic type (action.affordance_type, target.artifact_type, target.workspace_type), use EXACT namespace:localname identifiers from the ontology prefixes and class names, case-sensitive. Never invent identifiers. Choose artifact and workspace identifiers consistent with the lists below.

Distinction between "NA" and null: identity fields use the string "NA" when applicable but not extractable. action.parameter and action.value use JSON null when not applicable (parameterless command / no value given).

## Environment capabilities

Workspaces and Artifacts available in this environment:

{capabilities_hierarchical}

## Ontology Reference (homeont)

```turtle
{ontology}
```

## Output

Return ONLY the single JSON object for the determined category — no prose, no markdown fences, no commentary.
"""
