"""
Unit tests for CodeBTExecutor - safety checks and direct py_trees execution.
"""

import pytest

from ami_agents.bt_planning.execution.code_executor import (
    CodeBTExecutor,
    CodeSafetyError,
)


@pytest.fixture
def executor():
    return CodeBTExecutor(max_ticks=10)


class TestSafety:
    """FORBIDDEN_PATTERNS and import restrictions must reject dangerous code."""

    @pytest.mark.parametrize(
        "code",
        [
            "import os\ntree = None",
            "import subprocess\ntree = None",
            "import sys\ntree = None",
            "with open('x', 'w') as f:\n    pass\ntree = None",
            "shutil.rmtree('/')\ntree = None",
        ],
    )
    def test_forbidden_code_is_rejected(self, executor, code):
        result = executor.execute(code)
        assert result.success is False
        assert "Safety check failed" in (result.error or "")

    def test_disallowed_import_rejected(self, executor):
        result = executor.execute("import json\ntree = None")
        assert result.success is False
        assert "Safety check failed" in (result.error or "")

    def test_empty_code_rejected(self, executor):
        result = executor.execute("   ")
        assert result.success is False
        assert "Empty" in (result.error or "")

    def test_missing_tree_rejected(self, executor):
        result = executor.execute("x = 1")
        assert result.success is False
        assert "tree" in (result.error or "").lower()

    def test_invalid_syntax_rejected(self, executor):
        result = executor.execute("tree = (")
        assert result.success is False
        assert "Safety check failed" in (result.error or "")


class TestExecution:
    """Valid py_trees code executes and returns a matching ExecutionResult."""

    def test_tree_variable_success(self, executor):
        code = (
            "root = py_trees.composites.Sequence(name='s', memory=True, "
            "children=[py_trees.behaviours.Success(name='ok')])\n"
            "tree = root"
        )
        result = executor.execute(code)
        assert result.success is True
        assert result.final_status == "SUCCESS"
        assert result.tree_name == "s"

    def test_build_tree_function(self, executor):
        code = (
            "def build_tree():\n"
            "    return py_trees.composites.Sequence(name='b', memory=True, "
            "children=[py_trees.behaviours.Success(name='ok')])\n"
        )
        result = executor.execute(code)
        assert result.success is True
        assert result.tree_name == "b"

    def test_failure_tree(self, executor):
        code = (
            "tree = py_trees.composites.Sequence(name='f', memory=True, "
            "children=[py_trees.behaviours.Failure(name='no')])"
        )
        result = executor.execute(code)
        assert result.success is False
        assert result.final_status == "FAILURE"

    def test_inline_compute_node_reads_writes_blackboard(self, executor):
        code = (
            "class Doubler(py_trees.behaviour.Behaviour):\n"
            "    def __init__(self, name):\n"
            "        super().__init__(name)\n"
            "        self.bb = self.attach_blackboard_client(name=name)\n"
            "        self.bb.register_key(key='out', access=py_trees.common.Access.WRITE)\n"
            "    def update(self):\n"
            "        self.bb.set('out', 42)\n"
            "        return py_trees.common.Status.SUCCESS\n"
            "tree = py_trees.composites.Sequence(name='c', memory=True, "
            "children=[Doubler('doubler')])"
        )
        result = executor.execute(code)
        assert result.success is True
        assert result.final_status == "SUCCESS"

    def test_blackboard_compute_node_injected(self, executor):
        code = (
            "tree = py_trees.composites.Sequence(name='bc', memory=True, "
            "children=[BlackboardComputeNode(name='agg', op='any', "
            "inputs=[], output='result')])"
        )
        result = executor.execute(code)
        assert result.success is True
