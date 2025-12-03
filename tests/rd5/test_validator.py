"""Tests for the AST-based code validator.

This module tests the safety checks implemented in rd5/execution/validator.py.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rd5.execution.validator import CodeValidator, validate_code, ValidationResult


class TestCodeValidator:
    """Test cases for CodeValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CodeValidator(
            allowed_imports={"json", "asyncio", "aiohttp", "datetime", "math", "typing"},
            blocked_patterns={"os", "sys", "subprocess", "eval", "exec", "open", "__import__"},
        )

    def test_valid_code_passes(self):
        """Test that safe code passes validation."""
        code = """
import json
import asyncio
from typing import Dict, Any

async def main() -> Dict[str, Any]:
    data = {"key": "value"}
    result = json.dumps(data)
    return {"success": True, "data": result}
"""
        result = self.validator.validate(code)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert "json" in result.imports_found
        assert "asyncio" in result.imports_found

    def test_disallowed_import_fails(self):
        """Test that disallowed imports are caught."""
        code = """
import os
import json

def get_files():
    return os.listdir(".")
"""
        result = self.validator.validate(code)

        assert result.is_valid is False
        assert any("os" in err for err in result.errors)
        assert "os" in result.imports_found

    def test_subprocess_blocked(self):
        """Test that subprocess import is blocked."""
        code = """
import subprocess

def run_cmd():
    subprocess.run(["ls", "-la"])
"""
        result = self.validator.validate(code)

        assert result.is_valid is False
        assert any("subprocess" in err for err in result.errors)

    def test_eval_blocked(self):
        """Test that eval() calls are blocked."""
        code = """
import json

def dangerous():
    user_input = "2 + 2"
    result = eval(user_input)
    return result
"""
        result = self.validator.validate(code)

        assert result.is_valid is False
        assert any("eval" in err for err in result.errors)

    def test_exec_blocked(self):
        """Test that exec() calls are blocked."""
        code = """
def run_code(code_str):
    exec(code_str)
"""
        result = self.validator.validate(code)

        assert result.is_valid is False
        assert any("exec" in err for err in result.errors)

    def test_open_blocked(self):
        """Test that open() calls are blocked."""
        code = """
def read_file():
    with open("/etc/passwd") as f:
        return f.read()
"""
        result = self.validator.validate(code)

        assert result.is_valid is False
        assert any("open" in err for err in result.errors)

    def test_syntax_error_caught(self):
        """Test that syntax errors are caught."""
        code = """
def broken(
    # Missing closing parenthesis
"""
        result = self.validator.validate(code)

        assert result.is_valid is False
        assert any("Syntax error" in err for err in result.errors)

    def test_from_import_checked(self):
        """Test that from...import statements are checked."""
        code = """
from os.path import join

def get_path():
    return join("a", "b")
"""
        result = self.validator.validate(code)

        assert result.is_valid is False
        assert "os" in result.imports_found

    def test_aiohttp_allowed(self):
        """Test that aiohttp is allowed for HTTP calls."""
        code = """
import aiohttp
import asyncio

async def fetch_data(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
"""
        result = self.validator.validate(code)

        assert result.is_valid is True
        assert "aiohttp" in result.imports_found

    def test_datetime_allowed(self):
        """Test that datetime is allowed."""
        code = """
from datetime import datetime, timezone

def get_timestamp():
    return datetime.now(timezone.utc).isoformat()
"""
        result = self.validator.validate(code)

        assert result.is_valid is True
        assert "datetime" in result.imports_found

    def test_dunder_import_blocked(self):
        """Test that __import__ is blocked."""
        code = """
def dynamic_import(name):
    module = __import__(name)
    return module
"""
        result = self.validator.validate(code)

        assert result.is_valid is False
        assert any("__import__" in err for err in result.errors)

    def test_complex_valid_code(self):
        """Test validation of complex but valid code."""
        code = """
import asyncio
import json
from typing import Dict, Any, List
import aiohttp
from datetime import datetime

async def call_affordance(
    session: aiohttp.ClientSession,
    url: str,
    method: str = "GET",
    data: Dict[str, Any] = None
) -> Dict[str, Any]:
    try:
        if method.upper() == "GET":
            async with session.get(url) as response:
                return {"success": response.status < 400, "data": await response.json()}
        elif method.upper() == "POST":
            async with session.post(url, json=data) as response:
                return {"success": response.status < 400, "data": await response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def main() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        result = await call_affordance(session, "http://device/action", "POST", {"value": 50})
        results.append(result)

    return {
        "success": all(r.get("success", False) for r in results),
        "timestamp": datetime.now().isoformat(),
        "results": results
    }

if __name__ == "__main__":
    result = asyncio.run(main())
    print(json.dumps(result, indent=2))
"""
        result = self.validator.validate(code)

        assert result.is_valid is True
        assert len(result.errors) == 0


class TestValidateCodeFunction:
    """Test the convenience validate_code function."""

    def test_validate_code_convenience(self):
        """Test the module-level validate_code function."""
        safe_code = """
import json
data = {"test": True}
"""
        result = validate_code(safe_code)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_defaults(self):
        """Test ValidationResult default values."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.imports_found == set()
        assert result.blocked_patterns_found == set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
