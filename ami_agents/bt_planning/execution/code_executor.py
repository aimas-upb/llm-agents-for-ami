"""
Code BT Executor - executes LLM-generated py_trees Python code directly.

The Direct Code planning mode emits executable py_trees Python (including
inline ``py_trees.behaviour.Behaviour`` compute subclasses) that is NOT
representable in the JSON IR. This executor runs that code in a restricted
namespace and then delegates the tick loop to :class:`IRExecutor` so that
settling/poll behaviour and executed-action harvesting stay identical across
both planning modes.

The threat model is a hallucinated dangerous call from an internal LLM
component, not an adversarial attacker: the code runs via ``exec`` with a
restricted ``__builtins__`` and a forbidden-pattern scan, and only this repo's
node classes are injected into the namespace.
"""

import ast
import logging
import re

import py_trees
from py_trees.common import Status

from ..nodes.affordance_nodes import (
    ActionAffordanceNode,
    PropertyAffordanceNode,
    PropertyConditionNode,
    ComparisonPropertyConditionNode,
    WaitPropertyConditionNode,
    ComparisonOperator,
)
from ..nodes.blackboard_keys import BlackboardKeys
from ..nodes.compute_node import BlackboardComputeNode
from .base import ExecutionResult
from .ir_executor import IRExecutor

logger = logging.getLogger(__name__)


# Patterns that indicate potentially dangerous code. Preserved from the
# behaviortree-planning-for-ami-agents ablation executor. Do NOT trim this
# list -- it guards against dangerous calls hallucinated by the LLM.
FORBIDDEN_PATTERNS = [
    r'\bos\.(remove|unlink|rmdir|rmtree|system|popen|exec|spawn)',
    r'\bshutil\.(rmtree|move|copy|copytree)',
    r'\bsubprocess\.',
    r'\b__import__\s*\(',
    r'\beval\s*\(',
    r'\bexec\s*\(',
    r'\bcompile\s*\(',
    r'\bopen\s*\([^)]*["\']w',  # write mode
    r'\brm\s+-rf',
    r'\bimport\s+subprocess',
    r'\bimport\s+shutil',
    r'\bimport\s+os\b',
    r'\bfrom\s+os\s+import',
    r'\bimport\s+sys\b',
    r'\bfrom\s+sys\s+import',
    r'\b__builtins__',
    r'\b__class__',
    r'\b__bases__',
    r'\b__subclasses__',
    r'\b__mro__',
    r'\b__globals__',
    r'\b__code__',
]

# The only import the generated code is allowed to make.
ALLOWED_IMPORTS = {"py_trees"}


class CodeSafetyError(Exception):
    """Raised when generated code fails safety checks."""
    pass


class CodeBTExecutor:
    """
    Executor for LLM-generated py_trees Python code.

    Runs safety checks, execs the code in a restricted namespace populated with
    this repo's node classes, extracts the tree (``tree`` variable or
    ``build_tree()`` function), then reuses :class:`IRExecutor` for ticking so
    metrics match the JSON IR path.
    """

    def __init__(self, max_ticks: int = None, config: dict = None):
        self._ir = IRExecutor(max_ticks=max_ticks, config=config)
        self.max_ticks = self._ir.max_ticks

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute a Python-code behavior tree.

        Args:
            code: py_trees Python source that defines a top-level ``tree``
                variable or a ``build_tree()`` function.

        Returns:
            ExecutionResult (same shape as the JSON IR path).
        """
        if not isinstance(code, str) or not code.strip():
            return ExecutionResult(success=False, error="Empty or invalid code")

        try:
            logger.info("Checking generated code safety")
            self._check_safety(code)

            logger.info("Executing generated py_trees code")
            tree = self._build_tree(code)
        except CodeSafetyError as e:
            logger.error(f"Code safety check failed: {e}")
            return ExecutionResult(success=False, error=f"Safety check failed: {e}")
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return ExecutionResult(success=False, error=str(e))

        logger.info("Executing behavior tree")
        return self._ir._execute_tree(tree)

    def _check_safety(self, code: str) -> None:
        """Reject forbidden patterns and disallowed imports before exec."""
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                raise CodeSafetyError(f"Forbidden pattern detected: {pattern}")

        try:
            parsed = ast.parse(code)
        except SyntaxError as e:
            raise CodeSafetyError(f"Invalid Python syntax: {e}")

        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module not in ALLOWED_IMPORTS:
                        raise CodeSafetyError(f"Forbidden import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module not in ALLOWED_IMPORTS:
                        raise CodeSafetyError(f"Forbidden import from: {node.module}")

    def _build_tree(self, code: str) -> py_trees.behaviour.Behaviour:
        """Exec the code in a restricted namespace and extract the root behaviour."""
        safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "True": True,
            "False": False,
            "None": None,
            "Exception": Exception,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "type": type,
            "super": super,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sum": sum,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "any": any,
            "all": all,
            "__build_class__": __builtins__["__build_class__"]
            if isinstance(__builtins__, dict)
            else getattr(__builtins__, "__build_class__"),
            "__import__": __import__,
        }

        safe_globals = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
            "py_trees": py_trees,
            "Status": Status,
            "ActionAffordanceNode": ActionAffordanceNode,
            "PropertyAffordanceNode": PropertyAffordanceNode,
            "PropertyConditionNode": PropertyConditionNode,
            "ComparisonPropertyConditionNode": ComparisonPropertyConditionNode,
            "WaitPropertyConditionNode": WaitPropertyConditionNode,
            "ComparisonOperator": ComparisonOperator,
            "BlackboardKeys": BlackboardKeys,
            "BlackboardComputeNode": BlackboardComputeNode,
        }

        local_ns: dict = {}
        try:
            exec(code, safe_globals, local_ns)
        except Exception as e:
            raise CodeSafetyError(f"Code execution failed: {e}")

        if "tree" in local_ns:
            tree = local_ns["tree"]
        elif "build_tree" in local_ns:
            try:
                tree = local_ns["build_tree"]()
            except Exception as e:
                raise CodeSafetyError(f"build_tree() failed: {e}")
        else:
            raise CodeSafetyError(
                "Code must define a 'tree' variable or a 'build_tree()' function"
            )

        if isinstance(tree, py_trees.trees.BehaviourTree):
            tree = tree.root
        if not isinstance(tree, py_trees.behaviour.Behaviour):
            raise CodeSafetyError(
                f"'tree' must be a py_trees.behaviour.Behaviour, got {type(tree)}"
            )
        return tree
