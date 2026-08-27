"""
Unit tests for the BT-builder DSL and its restricted executor
(direct-code plan generation, ``python_code`` mode).
"""

import pytest

from ami_agents.bt_planning.planning.bt_dsl import (
    action,
    build_env_snapshot,
    condition,
    parallel,
    run_bt_code,
    selector,
    sequence,
    wait_condition,
)


# --------------------------------------------------------------------------- #
# Builders return the exact JSON IR shapes
# --------------------------------------------------------------------------- #
class TestBuilders:
    def test_action_minimal(self):
        assert action("light308/turnOn") == {
            "type": "action",
            "affordance_id": "light308/turnOn",
        }

    def test_action_with_parameters_and_name(self):
        node = action("light308/setBrightness", {"brightness": 75}, name="Brighten")
        assert node == {
            "type": "action",
            "affordance_id": "light308/setBrightness",
            "parameters": {"brightness": 75},
            "name": "Brighten",
        }

    def test_condition_default_op_omits_operator(self):
        node = condition("light308/state", "on")
        assert node == {
            "type": "condition",
            "affordance_id": "light308/state",
            "expected_value": "on",
        }

    def test_condition_with_comparison_operator(self):
        node = condition("room/temperature", 21.5, op="<=")
        assert node["operator"] == "<="
        assert node["expected_value"] == 21.5

    def test_wait_condition_with_timing(self):
        node = wait_condition(
            "room/temperature", 21.5, op="<=", timeout_seconds=95, poll_interval_seconds=2
        )
        assert node["type"] == "wait_condition"
        assert node["timeout_seconds"] == 95
        assert node["poll_interval_seconds"] == 2

    def test_wait_condition_omits_unset_timing(self):
        node = wait_condition("room/temperature", 21.5, op="<=")
        assert "timeout_seconds" not in node
        assert "poll_interval_seconds" not in node

    def test_sequence_nests_children(self):
        node = sequence("plan", action("a/b"), condition("c/d", 1))
        assert node["type"] == "sequence"
        assert node["name"] == "plan"
        assert [c["type"] for c in node["children"]] == ["action", "condition"]

    def test_selector_and_parallel(self):
        sel = selector("fallback", condition("a/b", "on"), action("a/c"))
        assert sel["type"] == "selector"
        par = parallel("both", action("a/b"), action("a/c"), policy="success_on_one")
        assert par["policy"] == "success_on_one"

    def test_composite_requires_children(self):
        with pytest.raises(ValueError, match="at least one child"):
            sequence("empty")

    def test_composite_rejects_non_node_children(self):
        with pytest.raises(ValueError, match="children must be nodes"):
            sequence("bad", "not-a-node")

    def test_condition_requires_expected_value(self):
        with pytest.raises(ValueError, match="expected value"):
            condition("a/b", None)

    def test_condition_rejects_unknown_operator(self):
        with pytest.raises(ValueError, match="op must be one of"):
            condition("a/b", 1, op="~=")

    def test_action_rejects_missing_affordance_id(self):
        with pytest.raises(ValueError, match="affordance_id"):
            action("")


# --------------------------------------------------------------------------- #
# Restricted execution
# --------------------------------------------------------------------------- #
class TestRunBtCode:
    def test_happy_path_with_env_arithmetic(self):
        env = {"room_environment": {"temperature": 24.0}}
        code = (
            "current = env['room_environment']['temperature']\n"
            "tree = sequence('cool',\n"
            "    action('ac/set_temperature', {'temperature': min(current - 2, 22)}),\n"
            "    wait_condition('room_environment/temperature', current - 0.2, op='<='))\n"
        )
        result = run_bt_code(code, env)
        assert result.errors == []
        assert result.tree["children"][0]["parameters"] == {"temperature": 22}
        assert result.tree["children"][1]["expected_value"] == 23.8

    def test_import_rejected(self):
        result = run_bt_code("import os\ntree = action('a/b')", {})
        assert result.tree is None
        assert any("import statements are not allowed" in e for e in result.errors)

    def test_dunder_name_rejected(self):
        result = run_bt_code("tree = __import__('os')", {})
        assert any("dunder names" in e for e in result.errors)

    def test_dunder_attribute_rejected(self):
        result = run_bt_code("tree = action.__globals__", {})
        assert any("dunder attributes" in e for e in result.errors)

    def test_runtime_error_surfaced_with_line(self):
        code = "x = 1\ny = env['missing']['key']\ntree = action('a/b')"
        result = run_bt_code(code, {})
        assert result.tree is None
        assert any("KeyError" in e and "line 2" in e for e in result.errors)

    def test_syntax_error_surfaced(self):
        result = run_bt_code("tree = sequence('x',", {})
        assert any("SyntaxError" in e for e in result.errors)

    def test_missing_tree_assignment(self):
        result = run_bt_code("x = action('a/b')", {})
        assert any("variable named 'tree'" in e for e in result.errors)

    def test_empty_code(self):
        result = run_bt_code("   ", {})
        assert any("empty" in e for e in result.errors)

    def test_mark_impossible(self):
        result = run_bt_code("mark_impossible('no matching affordance')", {})
        assert result.impossible is True
        assert result.explanation == "no matching affordance"
        assert result.errors == []

    def test_explanation_variable_captured(self):
        result = run_bt_code(
            "tree = action('a/b')\nexplanation = 'turns it on'", {}
        )
        assert result.explanation == "turns it on"

    def test_whitelisted_builtins_work(self):
        code = "tree = action('a/b', {'v': round(max(1.4, 2.6))})"
        result = run_bt_code(code, {})
        assert result.errors == []
        assert result.tree["parameters"] == {"v": 3}

    def test_open_is_unavailable(self):
        result = run_bt_code("open('/etc/passwd')\ntree = action('a/b')", {})
        assert any("NameError" in e for e in result.errors)

    def test_helper_function_and_loop_allowed(self):
        code = (
            "def bump(v):\n"
            "    return v + 1\n"
            "children = [action('a/b', {'i': bump(i)}) for i in range(2)]\n"
            "tree = sequence('loop', *children)\n"
        )
        result = run_bt_code(code, {})
        assert result.errors == []
        assert len(result.tree["children"]) == 2


# --------------------------------------------------------------------------- #
# env snapshot mirrors the prompt's Current State rendering
# --------------------------------------------------------------------------- #
class TestBuildEnvSnapshot:
    def test_nested_artifacts_shape(self):
        state = {
            "artifacts": {
                "http://localhost:8080/workspaces/lab/artifacts/light308": {
                    "brightness": 10,
                    "state": "off",
                }
            }
        }
        snap = build_env_snapshot(state)
        assert snap == {"light308": {"brightness": 10, "state": "off"}}

    def test_flat_state_shape(self):
        snap = build_env_snapshot({"light308": {"brightness": 10}})
        assert snap == {"light308": {"brightness": 10}}

    def test_empty_and_none(self):
        assert build_env_snapshot(None) == {}
        assert build_env_snapshot({}) == {}
