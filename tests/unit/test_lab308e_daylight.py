"""Offline checks for daylight benchmark starts, real-time progression, and CLI wiring.

A trial started at 23:59 on the host must expose noon daylight, then keep
advancing from noon rather than jumping back when the host crosses midnight.
No tests start Home Assistant, a simulator process, or an LLM request.
"""
import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from ami_agents.environment.integration.HomeAssistant import simulate_lab308e as dynamics
from tests.hasp.bench import harness, run
from tests.hasp.bench.backend import Lab308eBackend
from tests.simuhome.bench.models import load_variant


class DaylightClockTests(unittest.IsolatedAsyncioTestCase):
    """Check the clock and light published by a trial that begins overnight."""

    async def test_nighttime_host_gets_noon_sensors_and_real_elapsed_time(self):
        """Publish noon daylight and keep time advancing from that same anchor."""
        # Set a nighttime host clock and supply an actuator snapshot without HA.
        host = dt.datetime(2026, 10, 29, 23, 59, tzinfo=dt.timezone(dt.timedelta(hours=2)))
        simulator = dynamics.Lab308eSimulator(None, 5, daylight=True)
        simulator._read_world = AsyncMock(return_value={
            key: {"state": "off", "attributes": {}} for key in dynamics.ENTITY})
        simulator._set_entity = AsyncMock()

        # Cross midnight on the host's elapsed clock without resetting simulated time.
        with patch.object(dynamics.dt, "datetime", wraps=dt.datetime) as clock, \
             patch.object(dynamics.time, "monotonic", side_effect=[100, 100, 165]):
            clock.now.return_value = host
            first = await simulator.tick_once()
            second = await simulator.tick_once()
        noon = host.astimezone().replace(hour=12, minute=0)
        self.assertEqual(dt.datetime.fromisoformat(first["clock"]), noon)
        self.assertEqual(dt.datetime.fromisoformat(second["clock"]), noon + dt.timedelta(seconds=65))
        self.assertGreater(first["external_lux"], 6000)
        simulator._set_entity.assert_any_await(dynamics.ENTITY["external_lux"], first["external_lux"])

    def test_default_uses_host_time_and_each_daylight_instance_restarts_at_noon(self):
        """Keep default wall-clock behavior while independently anchoring new trials."""
        # Compare all modes against the same nighttime local timestamp.
        host = dt.datetime(2026, 10, 29, 23, 59, tzinfo=dt.timezone.utc)
        with patch.object(dynamics.dt, "datetime", wraps=dt.datetime) as clock:
            clock.now.return_value = host
            self.assertEqual(dynamics.Lab308eSimulator(None, 5).now(), host)
            for _ in range(2):
                self.assertEqual(dynamics.Lab308eSimulator(None, 5, daylight=True).now(),
                                 host.astimezone().replace(hour=12, minute=0))


class DaylightHarnessTests(unittest.TestCase):
    """Exercise variant overrides, the worker CLI, and historical reporting."""

    def test_yaml_and_cli_forward_daylight_as_boolean_flags(self):
        """Allow --no-daylight to override YAML without passing a string boolean."""
        # Exercise both YAML inheritance and an explicit command-line override.
        for override, expected in ((None, "--daylight"), (False, "--no-daylight")):
            args = argparse.Namespace(daylight=override)
            options = harness.run_options({"run": {"daylight": True}}, args)
            commands = harness.commands(Path("variant.yaml"), Path("out/v"), Path("out"),
                                        options, False, run.DEFAULT_BENCHMARK)
            self.assertIn(expected, commands[0])
            self.assertNotIn("True", commands[0])
            self.assertNotIn("False", commands[0])

    def test_worker_cli_accepts_daylight_and_defaults_to_wall_clock(self):
        """Parse daylight settings without executing the live benchmark worker."""
        # Stop at the async worker boundary so parsing cannot contact services.
        for flags, expected in (([], False), (["--daylight"], True), (["--no-daylight"], False)):
            with patch.object(run, "run", new=AsyncMock(return_value=0)) as worker:
                self.assertEqual(run.main(["--variant", "unused.yaml", *flags]), 0)
                self.assertIs(worker.call_args.args[0].daylight, expected)

    def test_report_uses_saved_daylight_condition(self):
        """Make historical report conditions independent of today's CLI defaults."""
        # Save just the run setting; model selection can use the ordinary defaults.
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp)
            (records / "run.json").write_text(json.dumps({"settings": {"daylight": True}}))
            self.assertIn("local noon (daylight) for every trial",
                          harness.report_legend(records, load_variant(None)))


class DaylightStartupTests(unittest.IsolatedAsyncioTestCase):
    """Check that the harness waits for fresh sensors rather than stale readiness."""

    async def test_daylight_waits_for_new_readiness_and_restarts_each_trial(self):
        """Remove prior readiness before starting each daylight simulator process."""
        # Emulate a simulator completing its first update, without a subprocess.
        with tempfile.TemporaryDirectory() as tmp:
            backend = object.__new__(Lab308eBackend)
            backend.args = SimpleNamespace(daylight=True, startup_timeout=1)
            backend.output = Path(tmp)
            ready = backend.output / ".daylight-ready"
            ready.write_text("stale")

            def start(name, command):
                """Only mark ready after confirming the old marker was removed."""
                self.assertFalse(ready.exists())
                self.assertIn("--daylight", command)
                self.assertEqual(command[-2:], ["--ready-file", str(ready)])
                ready.write_text("fresh")
                return SimpleNamespace(poll=lambda: None)

            backend._start = Mock(side_effect=start)
            await backend.start_dynamics()
            await backend.start_dynamics()
            self.assertEqual(backend._start.call_count, 2)

    async def test_startup_exit_is_reported_before_discovery(self):
        """Fail promptly if daylight dynamics exit before publishing sensor values."""
        # A failed child process must never leave the harness waiting indefinitely.
        with tempfile.TemporaryDirectory() as tmp:
            backend = object.__new__(Lab308eBackend)
            backend.args = SimpleNamespace(daylight=True, startup_timeout=1)
            backend.output = Path(tmp)
            backend._start = Mock(return_value=SimpleNamespace(poll=lambda: 1))
            with self.assertRaisesRegex(RuntimeError, "process exited"):
                await backend.start_dynamics()

    async def test_startup_blocks_until_the_first_sensor_update_is_ready(self):
        """Prevent discovery while the new simulator has not finished updating HA."""
        # Hold readiness after process launch to emulate a slow first sensor update.
        with tempfile.TemporaryDirectory() as tmp:
            backend = object.__new__(Lab308eBackend)
            backend.args = SimpleNamespace(daylight=True, startup_timeout=1)
            backend.output = Path(tmp)
            backend._start = Mock(return_value=SimpleNamespace(poll=lambda: None))
            task = asyncio.create_task(backend.start_dynamics())
            try:
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                (backend.output / ".daylight-ready").write_text("fresh")
                await asyncio.wait_for(task, 2)
            finally:
                # Avoid leaking a task if a startup assertion fails.
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
