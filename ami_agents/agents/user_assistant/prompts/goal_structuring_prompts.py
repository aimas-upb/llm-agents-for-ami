"""Stage 2 for goals -- structuring one atomic GOAL_REQUEST.

Which prompt a goal gets follows from the segmenter's qualifiers (see
`utils/goal_routing.py`):

    GOAL_STRUCTURING_SIMPLE_PROMPT       achievement, no dependency structure:
                                         one goal
    GOAL_STRUCTURING_DEPENDENCY_PROMPT   any dependency label, or maintenance:
                                         goals + the predicates conditioning
                                         them, each annotated with its role and
                                         ordering

Both prompts are assembled from the same fragments, so the goal fields cannot
drift between them. Neither composes a plan: the structurer names what each
part is and how it is ordered; the planner builds the behaviour tree.

The home context goes in through `render_goal_prompt`, a plain substitution of
`{goal_context}`: the prompts quote JSON, and `str.format` would need every
brace doubled.
"""

GOAL_CONTEXT_PLACEHOLDER = "{goal_context}"

CONTEXT_GUIDE = """\
## Home context

What this home offers, by workspace. It may be narrowed to the rooms and devices the request names. Each element starts on a line naming its kind (`Workspace`, `Environment`, `Variable`, `Artifact`, `Action`, `Property`); every fact about it is on its own labelled line beneath. Copy names and classes from the labelled lines EXACTLY, case-sensitive. A class or name that does not appear here matches nothing in this home: never invent one.

- `Workspace` — a room. `name` is the room's name; `room class` is a **location_class**.
- `Environment` with its `Variable`s — the room's environment variables. A variable's `name` is its title, `class` its class; `observed by` lists the device properties that read it (`none`: nothing in this home can). (Present only when needed.)
- `Artifact` — a device. `name` is its **artifact_name** — this device's own name. `class` is its **artifact_class**; the `class label` beside it names the *kind* of device, never this device, and is never an artifact_name. `semantic types` lists every type the device has.
- `Action` — something the device can do. `name` is the **affordance_name**; `command class` is the **affordance_class**. Its `input` is either a *direct value* (the action takes the value itself) or a *parameterized call* (named parameters, listed beneath it); `range` gives the lowest and highest value the input accepts. `acts upon` names the device property it changes; `environment effect` names the room variable it moves.
- `Property` — something the device reports. `name` is the **property_name**; `property class` is the **property_class**; `kind` is its branch (actuatable, state, capability, ...). Its `output` is a direct value or an object whose fields are listed beneath it. (Present only when needed.)
"""

GOAL_SPEC_RULES = """\
## Structuring a goal

A goal is ONE change to ONE device affordance (or, for an ambiguous goal, a desired change to the environment). Fill the fields according to its specificity.

`goal_specificity` starts from the segmenter's label, given with the request. The segmenter sees only the words; you see the home. Change "explicit" to "incomplete" when the home context does not single out one device and one action:
- several devices fit the request ("the living room lights" with two lights in the living room) — even when the request names their room and kind;
- no device in the home context fits it ("the bathroom blinds" in a bathroom without blinds);
- the device fits, but none of its actions can make the change ("make the hallway light blue" on an on/off light);
- several of its actions fit, and the request does not say which ("set the AC to 22 degrees" with a cooling and a heating setpoint).
Never pick one of several fitting devices or actions to keep a goal explicit: leave artifact_name (or affordance_name) null and let the quantifier say how many the request means.

### explicit and incomplete goals
- **goal_specificity**: "explicit" or "incomplete".
- **goal_kind**: "achievement" when the goal is done once its action is performed or it has a time-bounded objective; "maintenance" when it asks for a state to be kept or held ("keep the room at 22 degrees").
- **intent_text**: the part of the request this goal comes from, verbatim.
- **location_class**: the room class of the device's workspace.
- **artifact_class**: the device's homeont class.
- **artifact_name**: the `name` of the device's `Artifact`.
- **affordance_class**: the command class of the action that performs the change, copied from its action's `command class`. Its `acts upon` property tells you what the action changes — use it to choose the right action, never as the affordance_class.
- **parent_class**: the class `affordance_class` sits under (e.g. saref:Command); null if unknown.
- **affordance_name**: the `name` of the `Action`.
- **parameter_name**: only when the action's input is a parameterized call — the parameter the value goes into, copied from that input. "NA" when the action needs no parameter (a direct value, or no input at all). null when a parameter is needed but you cannot tell which. "NA" and null mean different things: keep them apart.
- **goal_effect**: "set" for an absolute value or a parameterless action (turn on/off, start, pause, open, close); "modify" for a relative change (increase by, lower, bump, ease).
- **target_value_text**: the words of the request that state the value to set, or the amount to change by ("cooling mode", "level 30", "minus 16.0 C", "80% closed"), copied verbatim; null when the request states none.
- **target_value_determined**:
  - for "set": the value the action takes for those words, read from the action's `input` in the home context: the enum member its description gives ("cooling mode" against "3 = Cool" → "3"; "on" against a boolean → "true"), or the number in the input's unit ("minus 16.0 C" against unit:DEG_C → "-16.0"), or a percentage as the user states it ("40% brightness" → "40"). For a parameterized call, the value of `parameter_name`. null when the input's type, unit and description do not determine it — never guess a scale they do not state.
  - for "modify": the change exactly as the user states it, never calculated from a range or a current value: a plain number in the user's own unit, or a percentage of change. Positive to increase, negative to decrease ("lower by 2 degrees" → "-2"; "dim by 20%" → "-20"; "twice as much" → "100"; "half as bright" → "-50").
- **target_value_is_percentage**: true when target_value_determined is a percentage, otherwise false.

- **quantifier**: incomplete goals only — null for an explicit goal. Whether the request means the change for every device that fits the goal or for one of them. It is the request's meaning, not a count of the devices in the home context. Decide by meaning, not by grammatical number:
  - "all" — every device that fits: a plural ("turn on the kitchen lights"), all/every ("close all the blinds"), and "any … that …" selecting devices by a condition ("set any living room light that is on to 20%" — every light that is on).
  - "one" — a single device is enough, whichever it is: "turn on the kitchen light", "turn on a fan", "turn on any kitchen light".
  - null — only when the request gives no indication, naming no device at all ("turn it off", "set it to 50%").

For an **incomplete** goal fill everything the request and the home context support, and leave the rest null. Do not guess a device or a room the request does not point to.

### ambiguous goals
The request expresses a desired change of the environment — a complaint, sensation or wish — without naming what to actuate or how.
- **goal_specificity**: "ambiguous"; **goal_kind**; **intent_text** as above.
- **implied_environment_vars**: the environment variables the goal implicitly refers to, each taken from an `Environment`'s `Variable`s:
  - **property_name**: the variable's `name`;
  - **property_class**: its class (e.g. homeont:RelativeHumidity), as listed as its `class`;
  - **property_quantity**: its quantity kind;
  - **property_measurement_unit**: its unit;
  - **property_sensed_space**: {"space_class": the room class, "space_name": the room's name} of the environment it belongs to.
  List one entry per variable and room the request implies.
"""

PREDICATE_SPEC_RULES = """\
## Structuring a predicate

A predicate is a condition the request depends on: the state of a device, or of a room's environment. It names what is read, not how it is read.

- **predicate_specificity**: "explicit" | "incomplete" | "ambiguous" (below).
- **predicate_subject**: "device_property" (a device's own property) or "environment_property" (a room's environment variable).
- **predicate_text**: the part of the request stating the condition, verbatim.
- **quantifier**: null when the condition is about one device or one room. When it may be met by several ("while the light is on" in a room with four lights): "any" by default, "all" only when the request says so ("all the lights", "every window", plural "the blinds").

### Device property predicates
- **location_class**, **artifact_class**, **artifact_name**: as for goals.
- **property_class**: the class of the property read; **parent_class**: the class it sits under.
- **property_name**: the `name` of the `Property`.
- **property_field**: only when the property's output is an object — the field compared. "NA" for a direct value; null if a field is needed but unclear.
- **comparison**: one of "==", "!=", ">", ">=", "<", "<=".
- **target_value_text**: the words stating the value compared against ("finished", "on", "half open", "loud"), copied verbatim; null if none.
- **target_value_determined**: the value of the property those words correspond to, read from the property's `output` in the home context — or, for a property shown "values as Action …", from that action's `input`: the enum member its description gives ("finished" against "0 = Stopped" → "0"), the boolean, or the number in its unit. null when they do not determine it ("loud" has no stated threshold).

explicit: device and property both identifiable. incomplete: some of location, device or property missing — leave those null. A device condition stated qualitatively ("while the TV is loud") is incomplete, not ambiguous: keep the words in target_value_text.

### Environment property predicates
- explicit (variable, room and threshold all stated — "while the bathroom humidity is above 60%"): **environment_var** (one dict, same fields as in implied_environment_vars), **comparison**, **target_value_text** ("60%"), **target_value_determined** (the number in the variable's unit: "60" against unit:PERCENT; null if the units do not match and no conversion is stated).
- incomplete: the same, with what is missing left null ("while the humidity is above 60%" — no room: property_sensed_space stays null, but property_class is still filled).
- ambiguous (the variable is not named: "while the room is very bright", "while it is stuffy", "once it gets dark"): **implied_environment_vars** (each also with **condition_direction**: "high" | "low" | null — "very bright" is illuminance high, "dark" is illuminance low), and **qualitative_condition**: the words stating the condition ("very bright"). Never turn them into a number.
"""

BINDING_RULES = """\
## Binding goals and predicates

Give every goal an id G1, G2, … and every predicate an id P1, P2, … in the order they appear. Then annotate how they relate. Do NOT build a plan or tree: only label what each part is for and when it starts.

### Each predicate
- **role**:
  - "guard" — "if X": checked once, when the goal is due.
  - "trigger" — "when / once / as soon as X", "N minutes after/before X finishes": the goal is due when X becomes true.
  - "scope" — "while X", X observable: the goal holds only while X holds.
  - "recurrence" — "whenever X", "every time X": the goal is due again each time X becomes true.
  - "termination" — "until X": the goal ends when X becomes true.
  - "filter" — "any / all … that are X": selects which candidates of the goal's device pool the goal applies to.
- **applies_to**: the ids of the goals it conditions.
- **connective**: "and" (default) or "or" — how it combines with the other predicates of the same role on the same goals ("while A and B" → both "and").
- for a **trigger** only: **offset_text** (the words, e.g. "23 minutes"; null if none) and **offset_direction** ("after" | "before").

State a trigger or termination as the condition that BECOMES TRUE: "after I close it" is a trigger on "the light is off".

### Each goal
- **timing**: {"starts": …, plus the field it needs}
  - "now" — no time given, or "now".
  - "at_time" + **time_text** — "at 1:39 PM".
  - "after_delay" + **offset_text** — "in 13 minutes", "13 minutes from now".
  - "after_goal" + **ref_goal** + **offset_text** — "then", "after that", "8 minutes after the previous action" (ref_goal is the goal referred to; offset_text null if none).
  - "with_goal" + **ref_goal** — at the same moment as another goal: "and set it to …", "while keeping it powered on", several devices under one trigger.
  - "on_trigger" — the goal has a trigger or recurrence predicate; that predicate carries the details.
  - "completes_with" + **ref** — "time it so A and B finish at the same time": ref is the goal or trigger predicate whose completion this goal's run must coincide with.
  - optional **ends**: {"kind": "duration" | "at_time", "text": …} — "for the next 5 hours", "until 6 PM".
- **rationale** (optional): the words giving a purpose or occasion ("so the laundry area is lit", "I'm clearing the table").

### Three uses of "while" that are NOT scope predicates
- "while keeping it powered on" — a second goal on the same device: starts "with_goal".
- "while I'm at the washer", "while I am working" — the user's activity, which no device observes: a rationale, never a predicate.
- A purpose or occasion clause: a rationale.

### One goal per affordance
Several actions on one device at one moment ("power on and set fan 30%") are several goals, the later ones starting "with_goal" the first.
"""

DEPENDENCY_EXAMPLES = """\
## Examples (home-specific fields abbreviated)

"Set dimmer light 1 in the living room to power on at level 30 at 13 minutes from now, and 8 minutes after the previous action drop it to level 20 while keeping it powered on."
- G1 power on: timing {"starts": "after_delay", "offset_text": "13 minutes"}
- G2 level 30: timing {"starts": "with_goal", "ref_goal": "G1"}
- G3 level 20: timing {"starts": "after_goal", "ref_goal": "G1", "offset_text": "8 minutes"}
- G4 stays powered on: timing {"starts": "with_goal", "ref_goal": "G3"}
- no predicates.

"23 minutes before washer 1 in the utility room finishes, set refrigerator 2 in the kitchen to 6.0°C and turn on dimmer light 1 in the living room to 60 percent."
- P1 washer 1 finished: role "trigger", applies_to ["G1", "G2", "G3"], offset_text "23 minutes", offset_direction "before".
- G1 refrigerator setpoint, G2 dimmer on, G3 dimmer level 60: timing {"starts": "on_trigger"} for each.

"Start dryer 1 in the utility room now on Max dryness level and pause dryer 1 in the utility room when washer 1 in the utility room finishes."
- G1 start: {"starts": "now"}; G2 Max dryness: {"starts": "with_goal", "ref_goal": "G1"}
- P1 washer 1 finished: role "trigger", applies_to ["G3"], offset_text null, offset_direction "after".
- G3 pause: {"starts": "on_trigger"}

"Start dryer 1 in the utility room at the right time … time it so dishwasher 1 in the kitchen and dryer 1 finish at the same time."
- P1 dishwasher 1 finished: role "trigger", applies_to ["G1"].
- G1 run dryer 1: timing {"starts": "completes_with", "ref": "P1"}

"Keep the living room at 22 degrees Celsius while the light is on, then 2 minutes after that set the heating pump level to 20 degrees."
- P1 a living room light is on: predicate_subject "device_property", predicate_specificity "incomplete", quantifier "any", role "scope", applies_to ["G1"].
- G1 hold 22 °C: goal_kind "maintenance", timing {"starts": "now"}.
- G2 heat pump to 20: goal_kind "achievement", timing {"starts": "after_goal", "ref_goal": "G1", "offset_text": "2 minutes"}.

"While it is bright outside and the living room blinds are half open, set any living room light that is on to a brightness of 20%."
- P1 bright outside: ambiguous environment predicate, role "scope", applies_to ["G1"], connective "and".
- P2 living room blinds half open: device predicate, quantifier "all", role "scope", applies_to ["G1"], connective "and".
- P3 light is on: device predicate with location and device left null — it is read on each candidate of G1 — role "filter", applies_to ["G1"].
- G1 brightness 20% on the living room lights: quantifier "all" (every light that passes P3), timing {"starts": "now"}.
"""

SIMPLE_OUTPUT = """\
## Output

Return ONLY one JSON object — no prose, no markdown fences. For an explicit or incomplete goal:

{"goal_specificity": "explicit" | "incomplete", "goal_kind": "achievement", "intent_text": string,
 "location_class": string | null, "artifact_class": string | null, "artifact_name": string | null,
 "affordance_class": string | null, "parent_class": string | null, "affordance_name": string | null,
 "parameter_name": string | "NA" | null, "goal_effect": "set" | "modify" | null,
 "target_value_text": string | null, "target_value_determined": string | null,
 "target_value_is_percentage": true | false,
 "quantifier": "all" | "one" | null}

For an ambiguous goal:

{"goal_specificity": "ambiguous", "goal_kind": "achievement", "intent_text": string,
 "implied_environment_vars": [{"property_name": string, "property_class": string | null,
   "property_quantity": string | null,
   "property_measurement_unit": string | null,
   "property_sensed_space": {"space_class": string | null, "space_name": string | null}}]}
"""

DEPENDENCY_OUTPUT = """\
## Output

Return ONLY one JSON object — no prose, no markdown fences:

{"goals": {"G1": {<goal fields as above>, "timing": {...}, "rationale": string (optional)}, ...},
 "predicates": {"P1": {<predicate fields as above>, "role": string, "applies_to": [string],
                       "connective": "and" | "or", "offset_text": string | null,
                       "offset_direction": "after" | "before" | null}, ...}}

"predicates" is {} when the request has none. Every goal has a timing; every predicate a role and applies_to.
"""

GOAL_STRUCTURING_SIMPLE_PROMPT = "\n".join([
    "You structure ONE atomic smart home goal: a single change the user wants, "
    "with no conditions, triggers or timing. The request and the segmenter's "
    "qualifiers are given as the user message.",
    "",
    GOAL_SPEC_RULES,
    CONTEXT_GUIDE,
    GOAL_CONTEXT_PLACEHOLDER,
    "",
    SIMPLE_OUTPUT,
])

GOAL_STRUCTURING_DEPENDENCY_PROMPT = "\n".join([
    "You structure ONE atomic smart home request that combines goals with "
    "conditions, timing or a state to maintain. Split it into its goals and the "
    "predicates (conditions) they depend on, and annotate how they relate. The "
    "request and the segmenter's qualifiers are given as the user message.",
    "",
    GOAL_SPEC_RULES,
    PREDICATE_SPEC_RULES,
    BINDING_RULES,
    DEPENDENCY_EXAMPLES,
    CONTEXT_GUIDE,
    GOAL_CONTEXT_PLACEHOLDER,
    "",
    DEPENDENCY_OUTPUT,
])

PROMPTS = {
    "simple": GOAL_STRUCTURING_SIMPLE_PROMPT,
    "dependency": GOAL_STRUCTURING_DEPENDENCY_PROMPT,
}


def render_goal_prompt(structure: str, goal_context: str) -> str:
    """The system prompt for one goal: its structure's prompt, with the context."""
    return PROMPTS[structure].replace(GOAL_CONTEXT_PLACEHOLDER, goal_context)
