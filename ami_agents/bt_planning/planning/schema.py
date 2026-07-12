"""
JSON IR schema and tool definition for BehaviorTree generation.

Extracted from behaviortree-planning-for-ami-agents src/planning/output/json_ir.py.
"""

# JSON schema for behavior tree nodes
TREE_PARAMETER_SCHEMA = {
    "type": "object",
    "minProperties": 1,
    "properties": {
        "name": {"type": "string"},
        "type": {
            "type": "string",
            "enum": ["sequence", "selector", "parallel", "action", "condition", "wait_condition"],
        },
        "children": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "object"},
            "description": "Child nodes (required for sequence/selector/parallel).",
        },
        "policy": {
            "type": "string",
            "enum": ["success_on_all", "success_on_one"],
            "description": "Policy for parallel nodes.",
        },
        "affordance_id": {
            "type": "string",
            "description": (
                "Exact affordance id from the affordances list "
                "(e.g. 'light308/setBrightness'). For condition nodes a readable "
                "property URL from the hints or current state is also accepted."
            ),
        },
        "parameters": {
            "type": "object",
            "description": "Optional parameters for action nodes.",
        },
        "expected_value": {
            "description": "Expected value for condition nodes.",
        },
        "operator": {
            "type": "string",
            "enum": ["==", "!=", ">", "<", ">=", "<="],
            "description": "Optional comparison operator for condition nodes.",
        },
        "value_path": {
            "type": "string",
            "description": "Optional JSON path for nested condition values.",
        },
        "timeout_seconds": {
            "type": "number",
            "minimum": 0,
            "description": "Maximum seconds to poll before a wait_condition fails.",
        },
        "poll_interval_seconds": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Seconds between property polls for wait_condition nodes.",
        },
    },
    "required": ["name", "type"],
    "oneOf": [
        {
            "title": "Sequence",
            "properties": {"type": {"const": "sequence"}},
            "required": ["children"],
        },
        {
            "title": "Selector",
            "properties": {"type": {"const": "selector"}},
            "required": ["children"],
        },
        {
            "title": "Parallel",
            "properties": {"type": {"const": "parallel"}},
            "required": ["children"],
        },
        {
            "title": "Action",
            "properties": {"type": {"const": "action"}},
            "required": ["affordance_id"],
        },
        {
            "title": "Condition",
            "properties": {"type": {"const": "condition"}},
            "required": ["affordance_id", "expected_value"],
        },
        {
            "title": "WaitCondition",
            "properties": {"type": {"const": "wait_condition"}},
            "required": ["affordance_id", "expected_value"],
        },
    ],
    "description": (
        "Behavior tree specification in JSON format; "
        "DO NOT leave this empty if the goal is possible to achieve."
    ),
}


# Tool definition for generating behavior trees
GENERATE_BT_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_behavior_tree",
        "description": "Generate a behavior tree specification for executing actions in a smart environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "tree": TREE_PARAMETER_SCHEMA,
                "explanation": {
                    "type": "string",
                    "description": "Explanation of the plan",
                },
                "impossible": {
                    "type": "boolean",
                    "description": "If the goal is impossible to achieve, mark this as true",
                },
            },
            "required": ["tree", "explanation", "impossible"],
        },
    },
}
