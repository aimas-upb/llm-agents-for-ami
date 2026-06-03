"""
Unit tests for BT JSON IR schema validation.
"""

import pytest

from ami_agents.bt_planning.execution.ir_executor import IRExecutor


@pytest.fixture
def executor():
    return IRExecutor(max_ticks=5)


class TestSchemaValidation:
    """Tests for IRExecutor._validate_tree()."""

    def test_valid_sequence_tree(self, executor, sample_sequence_spec):
        errors = executor.validate_tree(sample_sequence_spec)
        assert errors == []

    def test_valid_selector_tree(self, executor, sample_selector_spec):
        errors = executor.validate_tree(sample_selector_spec)
        assert errors == []

    def test_valid_parallel_tree(self, executor, sample_parallel_spec):
        errors = executor.validate_tree(sample_parallel_spec)
        assert errors == []

    def test_valid_action_with_params(self, executor):
        spec = {
            "name": "SetBrightness",
            "type": "action",
            "action_url": "http://localhost:8080/set_brightness",
            "parameters": {"brightness": 75},
        }
        errors = executor.validate_tree(spec)
        assert errors == []

    def test_valid_condition(self, executor, sample_condition_spec):
        errors = executor.validate_tree(sample_condition_spec)
        assert errors == []

    def test_valid_wait_condition(self, executor):
        spec = {
            "name": "WaitForGlare",
            "type": "wait_condition",
            "property_url": "http://localhost:8080/props/glare",
            "expected_value": 50,
            "operator": "<=",
            "timeout_seconds": 20,
            "poll_interval_seconds": 1,
        }
        errors = executor.validate_tree(spec)
        assert errors == []

    def test_valid_nested_tree(self, executor, sample_nested_spec):
        errors = executor.validate_tree(sample_nested_spec)
        assert errors == []

    def test_invalid_empty_tree(self, executor):
        errors = executor.validate_tree({})
        assert len(errors) > 0
        assert any("empty" in e for e in errors)

    def test_invalid_missing_type(self, executor):
        spec = {"name": "NoType"}
        errors = executor.validate_tree(spec)
        assert any("type" in e for e in errors)

    def test_invalid_unknown_type(self, executor):
        spec = {"name": "BadType", "type": "banana"}
        errors = executor.validate_tree(spec)
        assert any("type" in e for e in errors)

    def test_invalid_action_missing_url(self, executor):
        spec = {"name": "NoUrl", "type": "action"}
        errors = executor.validate_tree(spec)
        assert any("action_url" in e for e in errors)

    def test_invalid_condition_missing_property_url(self, executor):
        spec = {"name": "NoPropUrl", "type": "condition", "expected_value": "on"}
        errors = executor.validate_tree(spec)
        assert any("property_url" in e for e in errors)

    def test_invalid_condition_missing_expected_value(self, executor):
        spec = {
            "name": "NoExpected",
            "type": "condition",
            "property_url": "http://localhost:8080/props/state",
        }
        errors = executor.validate_tree(spec)
        assert any("expected_value" in e for e in errors)

    def test_invalid_wait_condition_missing_expected_value(self, executor):
        spec = {
            "name": "NoExpected",
            "type": "wait_condition",
            "property_url": "http://localhost:8080/props/state",
        }
        errors = executor.validate_tree(spec)
        assert any("expected_value" in e for e in errors)

    def test_invalid_sequence_empty_children(self, executor):
        spec = {"name": "EmptySeq", "type": "sequence", "children": []}
        errors = executor.validate_tree(spec)
        assert any("children" in e for e in errors)

    def test_invalid_sequence_no_children(self, executor):
        spec = {"name": "NoChildren", "type": "sequence"}
        errors = executor.validate_tree(spec)
        assert any("children" in e for e in errors)

    def test_invalid_nested_child(self, executor):
        spec = {
            "name": "Root",
            "type": "sequence",
            "children": [
                {"name": "Bad", "type": "action"},  # Missing action_url
            ],
        }
        errors = executor.validate_tree(spec)
        assert any("action_url" in e for e in errors)

    def test_invalid_not_dict(self, executor):
        errors = executor.validate_tree("not a dict")
        assert len(errors) > 0
