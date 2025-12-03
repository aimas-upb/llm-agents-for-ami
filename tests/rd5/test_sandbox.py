"""Tests for the sandbox execution module.

This module tests both Docker and Mock sandbox implementations.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rd5.execution.sandbox import (
    SandboxResult,
    MockSandbox,
    DockerSandbox,
    get_sandbox,
)


class TestSandboxResult:
    """Test cases for SandboxResult dataclass."""

    def test_sandbox_result_creation(self):
        """Test creating a SandboxResult."""
        result = SandboxResult(
            success=True,
            stdout="output",
            stderr="",
            return_value={"key": "value"},
            execution_time_ms=100,
            exit_code=0,
        )

        assert result.success is True
        assert result.stdout == "output"
        assert result.exit_code == 0
        assert result.return_value == {"key": "value"}


class TestMockSandbox:
    """Test cases for MockSandbox."""

    @pytest.mark.asyncio
    async def test_simple_print(self):
        """Test executing a simple print statement."""
        sandbox = MockSandbox(timeout=5)
        code = 'print("Hello, World!")'

        result = await sandbox.execute(code)

        assert result.success is True
        assert "Hello, World!" in result.stdout
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_json_output(self):
        """Test executing code that outputs JSON."""
        sandbox = MockSandbox(timeout=5)
        code = '''
import json
result = {"success": True, "message": "test"}
print(json.dumps(result))
'''

        result = await sandbox.execute(code)

        assert result.success is True
        assert result.return_value == {"success": True, "message": "test"}

    @pytest.mark.asyncio
    async def test_async_main(self):
        """Test executing async main function."""
        sandbox = MockSandbox(timeout=5)
        code = '''
import asyncio

async def main():
    await asyncio.sleep(0.01)
    return {"success": True, "async": True}
'''

        result = await sandbox.execute(code)

        assert result.success is True
        assert result.return_value == {"success": True, "async": True}

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        """Test handling syntax errors."""
        sandbox = MockSandbox(timeout=5)
        code = 'print("unclosed'

        result = await sandbox.execute(code)

        assert result.success is False
        assert result.error_type == "execution_error"
        assert "SyntaxError" in result.stderr

    @pytest.mark.asyncio
    async def test_runtime_error(self):
        """Test handling runtime errors."""
        sandbox = MockSandbox(timeout=5)
        code = '''
x = 1 / 0
'''

        result = await sandbox.execute(code)

        assert result.success is False
        assert result.error_type == "execution_error"
        assert "ZeroDivisionError" in result.stderr

    @pytest.mark.asyncio
    async def test_execution_time_recorded(self):
        """Test that execution time is recorded."""
        sandbox = MockSandbox(timeout=5)
        code = '''
import asyncio

async def main():
    await asyncio.sleep(0.1)
    return {"done": True}
'''

        result = await sandbox.execute(code)

        assert result.execution_time_ms >= 100

    @pytest.mark.asyncio
    async def test_restricted_builtins(self):
        """Test that dangerous builtins are restricted."""
        sandbox = MockSandbox(timeout=5)

        # open() should not be available
        code = 'f = open("/etc/passwd")'

        result = await sandbox.execute(code)

        assert result.success is False
        # Should fail because 'open' is not in restricted builtins

    @pytest.mark.asyncio
    async def test_allowed_operations(self):
        """Test that allowed operations work."""
        sandbox = MockSandbox(timeout=5)
        code = '''
import json

data = {"items": [1, 2, 3]}
total = sum(data["items"]) if hasattr(__builtins__, 'sum') else len(data["items"])
result = {"count": len(data["items"]), "success": True}
print(json.dumps(result))
'''

        result = await sandbox.execute(code)

        # Should succeed with basic operations
        assert result.success is True


class TestDockerSandbox:
    """Test cases for DockerSandbox."""

    def test_docker_sandbox_creation(self):
        """Test creating a DockerSandbox instance."""
        sandbox = DockerSandbox(
            timeout=30,
            memory_limit="128m",
            cpu_limit=0.5,
        )

        assert sandbox.timeout == 30
        assert sandbox.memory_limit == "128m"
        assert sandbox.cpu_limit == 0.5

    def test_get_sandbox_docker_available(self):
        """Test get_sandbox when Docker is available."""
        sandbox = get_sandbox(use_docker=True)

        # Should return either Docker or Mock based on availability
        assert sandbox is not None
        assert hasattr(sandbox, "execute")

    def test_get_sandbox_mock_forced(self):
        """Test get_sandbox with mock forced."""
        sandbox = get_sandbox(use_docker=False)

        assert isinstance(sandbox, MockSandbox)


class TestSandboxedExecutionNode:
    """Test cases for the sandboxed_execution node with real sandbox."""

    @pytest.mark.asyncio
    async def test_execution_with_mock_sandbox(self):
        """Test execution using mock sandbox."""
        # Set environment to use mock sandbox
        os.environ["RD5_USE_DOCKER_SANDBOX"] = "false"

        from rd5.workflow.nodes.sandboxed_execution import sandboxed_execution_node
        from rd5.workflow.state import create_initial_state, ValidationStatus

        state = create_initial_state(user_request="Test")
        state["generated_code"] = '''
import json
result = {"success": True, "message": "executed"}
print(json.dumps(result))
'''
        state["validation_result"] = {"is_valid": True}

        result = await sandboxed_execution_node(state)

        assert result["execution_result"]["success"] is True
        assert result["execution_result"]["return_value"]["success"] is True

    @pytest.mark.asyncio
    async def test_execution_with_error(self):
        """Test execution that produces an error."""
        os.environ["RD5_USE_DOCKER_SANDBOX"] = "false"

        from rd5.workflow.nodes.sandboxed_execution import sandboxed_execution_node
        from rd5.workflow.state import create_initial_state

        state = create_initial_state(user_request="Test")
        state["generated_code"] = '''
raise ValueError("Test error")
'''
        state["validation_result"] = {"is_valid": True}

        result = await sandboxed_execution_node(state)

        assert result["execution_result"]["success"] is False
        assert "ValueError" in result["execution_result"]["stderr"]


class TestSandboxIntegration:
    """Integration tests for sandbox with realistic code."""

    @pytest.mark.asyncio
    async def test_realistic_plan_code(self):
        """Test executing realistic plan code."""
        sandbox = MockSandbox(timeout=10)

        code = '''
import json
import asyncio
from typing import Dict, Any

async def call_affordance(url: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Simulate calling an affordance endpoint."""
    # In real execution, this would use aiohttp
    await asyncio.sleep(0.01)
    return {"success": True, "url": url, "method": method}

async def main() -> Dict[str, Any]:
    """Execute the plan."""
    results = []

    # Simulate turning on lights
    result1 = await call_affordance(
        "http://localhost:8090/lights/toggle",
        "POST",
        {"state": True}
    )
    results.append(result1)

    # Simulate setting brightness
    result2 = await call_affordance(
        "http://localhost:8090/lights/brightness",
        "PUT",
        {"level": 50}
    )
    results.append(result2)

    return {
        "success": all(r.get("success", False) for r in results),
        "results": results,
        "message": "Plan executed successfully"
    }
'''

        result = await sandbox.execute(code)

        assert result.success is True
        assert result.return_value is not None
        assert result.return_value["success"] is True
        assert len(result.return_value["results"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
