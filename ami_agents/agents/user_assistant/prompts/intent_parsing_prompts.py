"""Stage 2 — per-type intent parsing.

One prompt per query type produced by the segmentation stage. Each takes a
SINGLE atomic query and extracts the ontology classes it asks about. Goals are
structured separately (see `goal_structuring_prompts`).
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

- **Changing something on a device** ("can you dim the lights?", "can I set the purifier's fan mode?", "can you switch the TV to channel 5?"): a capability may be modelled as a changeable property, as a command, or as both. Fill `command` with the most specific class under `saref:Command` for the operation, and `device_property` with the most specific class for what is changed: under `homeont:ActuatableDeviceProperty` when the ontology context has one (that tree means *can be changed*), otherwise the class in another tree that names it (a reported state, say). Fill whichever of the two the context offers, and both when both fit. A changeable property answers on its own; a property from another tree answers only through a command that acts upon it.
- **The options a device supports** ("what fan modes does the purifier have?", "which wash programmes are there?"): fill `device_property`. Prefer the class under `homeont:ActuatableDeviceProperty` for the setting itself (its permitted values are the options); use a class under `homeont:DeviceCapabilityProperty` only when the question asks about that declared capability as such.
- **Sensing or reporting** ("can you tell how humid the bathroom is?", "does the washer report its remaining time?"): fill `environment_variable` (a room's environment) or `device_property` (a device's own condition or measurement). Leave `command` `null`.
- **An operation, with nothing changed named** ("what can you switch on in the kitchen?", "can anything be opened?"): fill `command` with the most specific class under `saref:Command` that the question supports.
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
