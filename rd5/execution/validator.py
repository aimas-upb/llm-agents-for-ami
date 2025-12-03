"""AST-based code validator for safe execution.

This module performs static analysis on generated Python code to ensure
it only uses allowed imports and doesn't contain dangerous patterns.
"""

import ast
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from rd5.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation.

    Attributes:
        is_valid: Whether the code passed all safety checks.
        errors: List of error messages if validation failed.
        warnings: List of warning messages (non-blocking).
        imports_found: Set of import names found in the code.
        blocked_patterns_found: Set of blocked patterns detected.
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    imports_found: Set[str] = field(default_factory=set)
    blocked_patterns_found: Set[str] = field(default_factory=set)


class CodeValidator:
    """AST-based code safety validator.

    Analyzes Python code to ensure it adheres to the safety whitelist
    and doesn't contain dangerous patterns.
    """

    def __init__(
        self,
        allowed_imports: Optional[Set[str]] = None,
        blocked_patterns: Optional[Set[str]] = None,
    ):
        """Initialize the validator.

        Args:
            allowed_imports: Set of allowed module names. If None, uses settings.
            blocked_patterns: Set of blocked identifier names. If None, uses settings.
        """
        settings = get_settings()
        self.allowed_imports = allowed_imports or settings.allowed_imports
        self.blocked_patterns = blocked_patterns or settings.blocked_patterns

    def validate(self, code: str) -> ValidationResult:
        """Validate Python code for safety.

        Args:
            code: Python source code to validate.

        Returns:
            ValidationResult with validation status and details.
        """
        result = ValidationResult(is_valid=True)

        # Step 1: Parse the code into AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result.is_valid = False
            result.errors.append(f"Syntax error: {e.msg} at line {e.lineno}")
            return result

        # Step 2: Check imports
        self._check_imports(tree, result)

        # Step 3: Check for blocked patterns
        self._check_blocked_patterns(tree, result)

        # Step 4: Check for dangerous constructs
        self._check_dangerous_constructs(tree, result)

        # Set overall validity
        result.is_valid = len(result.errors) == 0

        if result.is_valid:
            logger.info(
                f"Code validation passed. Imports: {result.imports_found}"
            )
        else:
            logger.warning(
                f"Code validation failed. Errors: {result.errors}"
            )

        return result

    def _check_imports(self, tree: ast.AST, result: ValidationResult) -> None:
        """Check all import statements against the whitelist.

        Args:
            tree: Parsed AST.
            result: ValidationResult to update.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    result.imports_found.add(module_name)
                    if module_name not in self.allowed_imports:
                        result.errors.append(
                            f"Disallowed import: '{module_name}' "
                            f"(line {node.lineno})"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    result.imports_found.add(module_name)
                    if module_name not in self.allowed_imports:
                        result.errors.append(
                            f"Disallowed import from: '{module_name}' "
                            f"(line {node.lineno})"
                        )

    def _check_blocked_patterns(
        self, tree: ast.AST, result: ValidationResult
    ) -> None:
        """Check for blocked identifier patterns.

        Args:
            tree: Parsed AST.
            result: ValidationResult to update.
        """
        for node in ast.walk(tree):
            # Check function calls
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name and func_name in self.blocked_patterns:
                    result.blocked_patterns_found.add(func_name)
                    result.errors.append(
                        f"Blocked function call: '{func_name}' "
                        f"(line {node.lineno})"
                    )

            # Check attribute access (e.g., os.system)
            elif isinstance(node, ast.Attribute):
                if node.attr in self.blocked_patterns:
                    result.blocked_patterns_found.add(node.attr)
                    result.errors.append(
                        f"Blocked attribute access: '{node.attr}' "
                        f"(line {node.lineno})"
                    )

            # Check variable names
            elif isinstance(node, ast.Name):
                if node.id in self.blocked_patterns:
                    result.blocked_patterns_found.add(node.id)
                    result.errors.append(
                        f"Blocked identifier: '{node.id}' "
                        f"(line {node.lineno})"
                    )

    def _check_dangerous_constructs(
        self, tree: ast.AST, result: ValidationResult
    ) -> None:
        """Check for dangerous code constructs.

        Args:
            tree: Parsed AST.
            result: ValidationResult to update.
        """
        for node in ast.walk(tree):
            # Check for exec/eval as string literals
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name in ("exec", "eval", "compile"):
                    result.errors.append(
                        f"Dangerous function: '{func_name}' "
                        f"(line {node.lineno})"
                    )

            # Check for __builtins__ access
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    if node.attr not in ("__name__", "__doc__", "__init__"):
                        result.warnings.append(
                            f"Dunder attribute access: '{node.attr}' "
                            f"(line {node.lineno})"
                        )

            # Check for string formatting that might be code injection
            if isinstance(node, ast.JoinedStr):  # f-strings
                for value in node.values:
                    if isinstance(value, ast.FormattedValue):
                        if isinstance(value.value, ast.Call):
                            result.warnings.append(
                                f"Function call in f-string "
                                f"(line {node.lineno})"
                            )

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract function name from a Call node.

        Args:
            node: AST Call node.

        Returns:
            Function name string or None if cannot be determined.
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None


def validate_code(code: str) -> ValidationResult:
    """Convenience function to validate code with default settings.

    Args:
        code: Python source code to validate.

    Returns:
        ValidationResult with validation status and details.
    """
    validator = CodeValidator()
    return validator.validate(code)
