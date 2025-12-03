"""Docker sandbox executor for safe code execution.

This module provides isolated execution of generated Python code
using Docker containers with strict resource and network limits.
"""

import asyncio
import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import docker
from docker.errors import ContainerError, ImageNotFound, APIError

from rd5.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Result of sandbox code execution.

    Attributes:
        success: Whether execution completed without errors.
        stdout: Standard output from the container.
        stderr: Standard error from the container.
        return_value: Parsed JSON return value if available.
        execution_time_ms: Execution time in milliseconds.
        exit_code: Container exit code.
        error_type: Type of error if execution failed.
    """

    success: bool
    stdout: str
    stderr: str
    return_value: Optional[Dict[str, Any]]
    execution_time_ms: int
    exit_code: int
    error_type: Optional[str] = None


class DockerSandbox:
    """Docker-based sandbox for isolated code execution.

    Executes Python code in a restricted Docker container with:
    - Limited memory and CPU
    - No network access (except to allowed endpoints)
    - Read-only filesystem
    - Dropped capabilities
    """

    SANDBOX_IMAGE = "rd5-sandbox:latest"
    SANDBOX_IMAGE_FALLBACK = "python:3.11-slim"

    def __init__(
        self,
        timeout: Optional[int] = None,
        memory_limit: Optional[str] = None,
        cpu_limit: Optional[float] = None,
        network: Optional[str] = None,
    ):
        """Initialize the Docker sandbox.

        Args:
            timeout: Execution timeout in seconds.
            memory_limit: Memory limit (e.g., "128m").
            cpu_limit: CPU limit as fraction (e.g., 0.5 for 50%).
            network: Docker network name for isolation.
        """
        settings = get_settings()
        self.timeout = timeout or settings.sandbox_timeout
        self.memory_limit = memory_limit or settings.sandbox_memory_limit
        self.cpu_limit = cpu_limit or settings.sandbox_cpu_limit
        self.network = network or settings.sandbox_network
        self._client: Optional[docker.DockerClient] = None

    @property
    def client(self) -> docker.DockerClient:
        """Get or create Docker client.

        Returns:
            Docker client instance.
        """
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _get_image(self) -> str:
        """Get the sandbox image, falling back if not available.

        Returns:
            Image name to use.
        """
        try:
            self.client.images.get(self.SANDBOX_IMAGE)
            return self.SANDBOX_IMAGE
        except ImageNotFound:
            logger.warning(
                f"Sandbox image {self.SANDBOX_IMAGE} not found, "
                f"using fallback {self.SANDBOX_IMAGE_FALLBACK}"
            )
            return self.SANDBOX_IMAGE_FALLBACK

    def _get_network_mode(self) -> str:
        """Get network mode for container.

        Returns:
            Network mode string.
        """
        # Check if our isolated network exists
        try:
            self.client.networks.get(self.network)
            return self.network
        except docker.errors.NotFound:
            # Create the network if it doesn't exist
            try:
                self.client.networks.create(
                    self.network,
                    driver="bridge",
                    internal=True,  # No external access
                )
                logger.info(f"Created isolated network: {self.network}")
                return self.network
            except APIError:
                # Fall back to no network
                logger.warning("Could not create isolated network, using 'none'")
                return "none"

    async def execute(self, code: str) -> SandboxResult:
        """Execute Python code in the sandbox.

        Args:
            code: Python code to execute.

        Returns:
            SandboxResult with execution details.
        """
        start_time = time.time()

        # Write code to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "plan.py"
            code_path.write_text(code)

            try:
                result = await self._run_container(tmpdir, code_path.name)
                execution_time_ms = int((time.time() - start_time) * 1000)
                result.execution_time_ms = execution_time_ms
                return result

            except asyncio.TimeoutError:
                execution_time_ms = int((time.time() - start_time) * 1000)
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {self.timeout}s",
                    return_value=None,
                    execution_time_ms=execution_time_ms,
                    exit_code=-1,
                    error_type="timeout",
                )

            except Exception as e:
                execution_time_ms = int((time.time() - start_time) * 1000)
                logger.error(f"Sandbox execution error: {e}")
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr=str(e),
                    return_value=None,
                    execution_time_ms=execution_time_ms,
                    exit_code=-1,
                    error_type="sandbox_error",
                )

    async def _run_container(
        self, host_dir: str, script_name: str
    ) -> SandboxResult:
        """Run the container with the code.

        Args:
            host_dir: Host directory containing the code.
            script_name: Name of the script file.

        Returns:
            SandboxResult with execution output.
        """
        image = self._get_image()
        network_mode = self._get_network_mode()

        # Container configuration
        container_config = {
            "image": image,
            "command": ["python", f"/sandbox/{script_name}"],
            "volumes": {
                host_dir: {"bind": "/sandbox", "mode": "ro"},
            },
            "network_mode": network_mode,
            "mem_limit": self.memory_limit,
            "nano_cpus": int(self.cpu_limit * 1e9),  # Convert to nanoseconds
            "read_only": True,
            "security_opt": ["no-new-privileges"],
            "cap_drop": ["ALL"],
            "detach": True,
            "remove": False,  # We'll remove after getting logs
        }

        # Run in thread pool to not block
        loop = asyncio.get_event_loop()
        container = await loop.run_in_executor(
            None,
            lambda: self.client.containers.run(**container_config),
        )

        try:
            # Wait for container with timeout
            exit_result = await asyncio.wait_for(
                loop.run_in_executor(None, container.wait),
                timeout=self.timeout,
            )

            exit_code = exit_result.get("StatusCode", -1)

            # Get logs
            stdout = await loop.run_in_executor(
                None,
                lambda: container.logs(stdout=True, stderr=False).decode("utf-8"),
            )
            stderr = await loop.run_in_executor(
                None,
                lambda: container.logs(stdout=False, stderr=True).decode("utf-8"),
            )

            # Try to parse return value from stdout
            return_value = self._parse_return_value(stdout)

            success = exit_code == 0 and (
                return_value is None or return_value.get("success", True)
            )

            return SandboxResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                return_value=return_value,
                execution_time_ms=0,  # Will be set by caller
                exit_code=exit_code,
                error_type=None if success else "execution_failed",
            )

        except asyncio.TimeoutError:
            # Kill the container
            await loop.run_in_executor(None, container.kill)
            raise

        finally:
            # Clean up container
            await loop.run_in_executor(None, container.remove)

    def _parse_return_value(self, stdout: str) -> Optional[Dict[str, Any]]:
        """Try to parse JSON return value from stdout.

        Args:
            stdout: Standard output from container.

        Returns:
            Parsed JSON dict or None.
        """
        if not stdout.strip():
            return None

        # Try to find JSON in the output
        lines = stdout.strip().split("\n")

        # Check last line first (most likely to be the result)
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        # Try to parse entire output as JSON
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return None

    def close(self) -> None:
        """Close the Docker client."""
        if self._client:
            self._client.close()
            self._client = None


class MockSandbox:
    """Mock sandbox for testing without Docker.

    Executes code using exec() with allowed imports pre-loaded.
    WARNING: Not secure for production use! Use Docker sandbox in production.
    """

    # Modules allowed to be imported
    ALLOWED_MODULES = {
        "json",
        "asyncio",
        "datetime",
        "math",
        "re",
        "collections",
        "itertools",
        "functools",
        "typing",
    }

    def __init__(self, timeout: int = 30):
        """Initialize mock sandbox.

        Args:
            timeout: Execution timeout in seconds.
        """
        self.timeout = timeout

    def _safe_import(self, name: str, globals_dict=None, locals_dict=None, fromlist=(), level=0):
        """Restricted import function that only allows whitelisted modules."""
        base_module = name.split(".")[0]
        if base_module not in self.ALLOWED_MODULES:
            raise ImportError(f"Import of '{name}' is not allowed")
        return __builtins__["__import__"](name, globals_dict, locals_dict, fromlist, level)

    async def execute(self, code: str) -> SandboxResult:
        """Execute code in mock sandbox.

        Args:
            code: Python code to execute.

        Returns:
            SandboxResult with execution details.
        """
        import io
        import builtins
        from contextlib import redirect_stdout, redirect_stderr

        start_time = time.time()

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Create a copy of builtins with restricted import
        safe_builtins = dict(vars(builtins))
        safe_builtins["__import__"] = self._safe_import
        # Remove dangerous builtins
        for dangerous in ["open", "exec", "eval", "compile", "input", "breakpoint"]:
            safe_builtins.pop(dangerous, None)

        # Namespace with safe builtins
        namespace = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
        }

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)

                # If there's an async main, run it
                if "main" in namespace and asyncio.iscoroutinefunction(namespace["main"]):
                    result = await asyncio.wait_for(
                        namespace["main"](),
                        timeout=self.timeout,
                    )
                    if result:
                        import json as json_module
                        print(json_module.dumps(result))

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Parse return value
            return_value = None
            if stdout.strip():
                try:
                    import json as json_module
                    return_value = json_module.loads(stdout.strip().split("\n")[-1])
                except:
                    pass

            return SandboxResult(
                success=True,
                stdout=stdout,
                stderr=stderr,
                return_value=return_value,
                execution_time_ms=execution_time_ms,
                exit_code=0,
            )

        except asyncio.TimeoutError:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return SandboxResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=f"Execution timed out after {self.timeout}s",
                return_value=None,
                execution_time_ms=execution_time_ms,
                exit_code=-1,
                error_type="timeout",
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return SandboxResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=f"{type(e).__name__}: {str(e)}",
                return_value=None,
                execution_time_ms=execution_time_ms,
                exit_code=1,
                error_type="execution_error",
            )


def get_sandbox(use_docker: bool = True) -> DockerSandbox | MockSandbox:
    """Get appropriate sandbox implementation.

    Args:
        use_docker: Whether to use Docker sandbox.

    Returns:
        Sandbox instance.
    """
    if use_docker:
        try:
            client = docker.from_env()
            client.ping()
            return DockerSandbox()
        except Exception as e:
            logger.warning(f"Docker not available: {e}, using mock sandbox")
            return MockSandbox()
    return MockSandbox()
