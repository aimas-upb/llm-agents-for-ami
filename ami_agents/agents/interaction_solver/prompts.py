INFER_IMPLICIT_INTENT_DESIRE_PROMPT = """
You map a vague user complaint or desire about a smart home to the environment variables it affects.

You receive ONE atomic intent already classified as implicit (no specific device or command named). Your job: infer which environment variables the user wants changed, and in which direction.

## Environment variables (fixed list)

- temperature — how warm or cool the air is
- humidity — how moist or dry the air is
- air_quality — how clean / fresh / unpolluted the air is (more = cleaner)
- air_circulation — how much the air is moving / being ventilated
- scent — presence of fragrance / aroma in the air
- luminosity — how bright the space is (artificial or natural light combined)
- color — the color tone of ambient lighting
- sound — audible sound level (music, audio playback)

## Direction values (fixed list)

- increase — the user wants more of this variable
- decrease — the user wants less of this variable
- unknown — direction cannot be reliably inferred

## Rules

- Read the user's complaint or desire and map it to ONE OR MORE of the variables above. Multiple variables are allowed if the intent clearly implies several effects.
- Use the user's own framing: "too bright" → luminosity / decrease. "stuffy" → typically air_circulation / increase (and often air_quality / increase). "clammy" → humidity / decrease. "make it cozier" → temperature / increase (judgment call; only emit if the wording leans that way).
- ONLY emit a pair if BOTH the variable AND the direction can be reliably inferred. If you can identify a variable but not its direction, OR a direction but not a variable, OR neither, emit a SINGLE fallback pair: {"variable": "unknown", "direction": "unknown"}.
- Do NOT mix real pairs with unknown pairs. If at least one (variable, direction) pair is reliably inferable, return only the reliable ones and DO NOT add an unknown fallback. The unknown fallback is used only when nothing reliable can be inferred at all.
- Do NOT invent variables outside the fixed list. Do NOT invent directions outside the fixed list.
- Stay close to the text; do not speculate about causes ("clammy skin" implies humidity, not temperature, unless the user says so).

## Output

Return ONLY a JSON object, no prose, no markdown fences:

{
  "intent_text": string,
  "affected_env_vars": [
    { "variable": string, "direction": string }
  ]
}

- intent_text: copy the input intent_text verbatim.
- affected_env_vars: a non-empty list of {variable, direction} pairs. Each variable must be from the fixed list (or "unknown"); each direction must be from the fixed list (or "unknown").

## Examples

Input: "Make the room brighter"
Output: {"intent_text": "Make the room brighter", "affected_env_vars": [{"variable": "luminosity", "direction": "increase"}]}

Input: "It's too stuffy in here"
Output: {"intent_text": "It's too stuffy in here", "affected_env_vars": [{"variable": "air_circulation", "direction": "increase"}, {"variable": "air_quality", "direction": "increase"}]}

Input: "Ugh the bathroom humidity is so high, my skin feels clammy and the towels probably still damp"
Output: {"intent_text": "Ugh the bathroom humidity is so high, my skin feels clammy and the towels probably still damp", "affected_env_vars": [{"variable": "humidity", "direction": "decrease"}]}

Input: "Something feels off in here"
Output: {"intent_text": "Something feels off in here", "affected_env_vars": [{"variable": "unknown", "direction": "unknown"}]}
"""

# Asked of the small model for a "modify" goal whose read-compute-set recipe
# could not be read off the Thing Descriptions alone. It names the pieces; the
# InteractionSolver checks them against the TDs and builds the tree itself.
MODIFY_RECIPE_PROMPT = """
You complete the plan for ONE relative change to ONE smart-home device ("dim the light by 20%", "lower the setpoint by 2 degrees").

The change is applied as: read the current value of a property, compute the new value, set it with an action. You only name the pieces; you do not write a plan.

You receive the structured goal and the device's affordances (actions with their input, properties with their output).

Return ONLY one JSON object:
{"action": "<action name>", "parameter": "value" | "<parameter name>", "property": "<property name>",
 "mode": "add" | "scale", "amount": <number>, "min": <number> | null, "max": <number> | null}

- action: the action that sets the value being changed, by its name.
- parameter: "value" when the action's input is a single value; otherwise the name of the input parameter that takes it.
- property: the property holding the current value of what the action changes, by its name.
- amount: the change from the goal, keeping its sign (positive increases, negative decreases).
- mode: "add" when the amount is added as it is (a plain amount, or a percentage of a property that is itself in percent); "scale" when it is a percentage of the current value.
- min, max: the range the new value must stay in, from the property's or the action's schema; null if none is stated.

If the device has no action and property that can make this change, return {"impossible": "<why, in one sentence>"}. Never invent a name.
"""
