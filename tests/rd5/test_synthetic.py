"""Tests for synthetic data generation and benchmarking.

This module tests the data generator and benchmark framework.
"""

import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Ensure mock sandbox is used
os.environ["RD5_USE_DOCKER_SANDBOX"] = "false"


class TestDataGenerator:
    """Test the synthetic data generator."""

    def test_generate_single_scenario(self):
        """Test generating a single scenario."""
        from rd5.synthetic.data_generator import generate_smart_home_request

        scenario = generate_smart_home_request()

        assert scenario is not None
        assert scenario.user_request
        assert scenario.expected_action_verb
        assert scenario.expected_target_device
        assert scenario.scenario_id

    def test_generate_scenario_with_device_type(self):
        """Test generating scenario for specific device."""
        from rd5.synthetic.data_generator import (
            generate_smart_home_request,
            DeviceType,
        )

        scenario = generate_smart_home_request(device_type=DeviceType.LIGHTS)

        assert scenario.expected_target_device == "lights"
        assert "light" in scenario.user_request.lower()

    def test_generate_scenario_with_parameters(self):
        """Test generating scenario with parameters."""
        from rd5.synthetic.data_generator import (
            generate_smart_home_request,
            DeviceType,
        )

        scenario = generate_smart_home_request(
            device_type=DeviceType.THERMOSTAT,
            include_parameter=True,
        )

        assert scenario.expected_target_device == "thermostat"
        # May or may not have parameters depending on template

    def test_generate_multiple_scenarios(self):
        """Test generating multiple scenarios."""
        from rd5.synthetic.data_generator import generate_test_scenarios

        scenarios = generate_test_scenarios(count=20, seed=42)

        assert len(scenarios) == 20
        # Check variety
        categories = set(s.category for s in scenarios)
        assert len(categories) > 1

    def test_generate_scenarios_reproducible(self):
        """Test that seed produces reproducible results."""
        from rd5.synthetic.data_generator import generate_test_scenarios

        scenarios1 = generate_test_scenarios(count=10, seed=123)
        scenarios2 = generate_test_scenarios(count=10, seed=123)

        for s1, s2 in zip(scenarios1, scenarios2):
            assert s1.user_request == s2.user_request
            assert s1.scenario_id == s2.scenario_id

    def test_generate_complex_scenarios(self):
        """Test complex scenario generation."""
        from rd5.synthetic.data_generator import generate_complex_scenario

        scenario = generate_complex_scenario()

        assert scenario.category == "complex"
        assert scenario.difficulty == "hard"
        assert scenario.expected_affordance_count > 1

    def test_generate_edge_cases(self):
        """Test edge case generation."""
        from rd5.synthetic.data_generator import generate_edge_case_scenario

        scenario = generate_edge_case_scenario()

        assert scenario.category in ["edge_case", "voice_prefix"]
        assert scenario.difficulty in ["edge", "medium"]

    def test_scenario_to_dict(self):
        """Test scenario serialization."""
        from rd5.synthetic.data_generator import generate_smart_home_request

        scenario = generate_smart_home_request()
        data = scenario.to_dict()

        assert "user_request" in data
        assert "expected_action_verb" in data
        assert "expected_target_device" in data
        assert "should_succeed" in data

    def test_generate_all_device_types(self):
        """Test scenarios can be generated for all device types."""
        from rd5.synthetic.data_generator import (
            generate_smart_home_request,
            DeviceType,
        )

        for device_type in DeviceType:
            scenario = generate_smart_home_request(device_type=device_type)
            assert scenario.expected_target_device == device_type.value


class TestBenchmarkResult:
    """Test benchmark result dataclass."""

    def test_create_benchmark_result(self):
        """Test creating a benchmark result."""
        from rd5.synthetic.benchmark import BenchmarkResult

        result = BenchmarkResult(
            scenario_id="test-001",
            user_request="Turn on the lights",
            category="lights",
            difficulty="easy",
            workflow_success=True,
            intent_extracted=True,
            code_generated=True,
            code_valid=True,
            execution_success=True,
            plan_stored=True,
            action_verb_match=True,
            target_device_match=True,
            location_match=True,
            total_time_ms=100,
        )

        assert result.scenario_id == "test-001"
        assert result.workflow_success is True

    def test_benchmark_result_to_dict(self):
        """Test benchmark result serialization."""
        from rd5.synthetic.benchmark import BenchmarkResult

        result = BenchmarkResult(
            scenario_id="test-002",
            user_request="Turn off the lights",
            category="lights",
            difficulty="easy",
            workflow_success=False,
            intent_extracted=True,
            code_generated=False,
            code_valid=False,
            execution_success=False,
            plan_stored=False,
            action_verb_match=True,
            target_device_match=True,
            location_match=True,
            total_time_ms=50,
            errors=["Code generation failed"],
        )

        data = result.to_dict()
        assert data["scenario_id"] == "test-002"
        assert data["errors"] == ["Code generation failed"]


class TestBenchmarkSummary:
    """Test benchmark summary calculations."""

    def test_create_benchmark_summary(self):
        """Test creating a benchmark summary."""
        from rd5.synthetic.benchmark import BenchmarkSummary

        summary = BenchmarkSummary(
            run_id="test-run",
            timestamp="2025-01-01T00:00:00",
            total_scenarios=100,
            duration_seconds=60.0,
            workflow_success_rate=0.9,
            intent_extraction_rate=0.95,
            code_generation_rate=0.92,
            validation_pass_rate=0.88,
            execution_success_rate=0.85,
            plan_storage_rate=0.85,
            action_verb_accuracy=0.9,
            target_device_accuracy=0.88,
            location_accuracy=0.85,
            avg_total_time_ms=600,
            avg_intent_time_ms=100,
            avg_generation_time_ms=300,
            avg_validation_time_ms=50,
            avg_execution_time_ms=150,
        )

        assert summary.total_scenarios == 100
        assert summary.workflow_success_rate == 0.9


class TestWorkflowBenchmark:
    """Test the workflow benchmark runner."""

    @pytest.mark.asyncio
    async def test_run_single_scenario_success(self):
        """Test running a single scenario successfully."""
        from rd5.synthetic.benchmark import WorkflowBenchmark
        from rd5.synthetic.data_generator import TestScenario

        scenario = TestScenario(
            user_request="Turn on the living room lights",
            expected_action_verb="turn on",
            expected_target_device="lights",
            expected_location="living room",
            scenario_id="bench-001",
        )

        # Mock the workflow
        mock_state = {
            "extracted_intent": {
                "intent": "turn on lights",
                "action_verb": "turn on",
                "target_objects": ["lights"],
                "location": "living room",
            },
            "generated_code": "print('hello')",
            "validation_result": {"is_valid": True},
            "execution_result": {"success": True, "execution_time_ms": 10},
            "plan_id": "plan-001",
            "workflow_status": MagicMock(value="completed"),
        }

        from rd5.workflow.state import WorkflowStatus
        mock_state["workflow_status"] = WorkflowStatus.COMPLETED

        with patch(
            "rd5.workflow.graph.run_plan_generation",
            new_callable=AsyncMock,
            return_value=mock_state,
        ):
            benchmark = WorkflowBenchmark()
            result = await benchmark.run_single_scenario(scenario)

            assert result.scenario_id == "bench-001"
            assert result.workflow_success is True
            assert result.action_verb_match is True
            assert result.target_device_match is True
            assert result.location_match is True

    @pytest.mark.asyncio
    async def test_run_single_scenario_failure(self):
        """Test handling scenario failure."""
        from rd5.synthetic.benchmark import WorkflowBenchmark
        from rd5.synthetic.data_generator import TestScenario

        scenario = TestScenario(
            user_request="",  # Empty request
            expected_action_verb="unknown",
            expected_target_device="unknown",
            should_succeed=False,
            scenario_id="bench-002",
        )

        from rd5.workflow.state import WorkflowStatus
        mock_state = {
            "extracted_intent": {},
            "generated_code": None,
            "validation_result": {"is_valid": False},
            "execution_result": {"success": False, "execution_time_ms": 0},
            "workflow_status": WorkflowStatus.FAILED,
            "errors": ["No user request provided"],
        }

        with patch(
            "rd5.workflow.graph.run_plan_generation",
            new_callable=AsyncMock,
            return_value=mock_state,
        ):
            benchmark = WorkflowBenchmark()
            result = await benchmark.run_single_scenario(scenario)

            assert result.workflow_success is False
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_run_benchmark_with_scenarios(self):
        """Test running benchmark with multiple scenarios."""
        from rd5.synthetic.benchmark import WorkflowBenchmark
        from rd5.synthetic.data_generator import generate_test_scenarios
        from rd5.workflow.state import WorkflowStatus

        scenarios = generate_test_scenarios(count=5, seed=42)

        # Mock successful workflow
        mock_state = {
            "extracted_intent": {
                "intent": "test",
                "action_verb": "turn on",
                "target_objects": ["lights"],
            },
            "generated_code": "print('test')",
            "validation_result": {"is_valid": True},
            "execution_result": {"success": True, "execution_time_ms": 10},
            "plan_id": "plan-test",
            "workflow_status": WorkflowStatus.COMPLETED,
        }

        with patch(
            "rd5.workflow.graph.run_plan_generation",
            new_callable=AsyncMock,
            return_value=mock_state,
        ):
            benchmark = WorkflowBenchmark()
            summary = await benchmark.run_benchmark(scenarios)

            assert summary.total_scenarios == 5
            assert len(benchmark.results) == 5

    def test_action_verb_matching(self):
        """Test action verb matching logic."""
        from rd5.synthetic.benchmark import WorkflowBenchmark

        benchmark = WorkflowBenchmark()

        # Exact match
        assert benchmark._check_action_match("turn on", "turn on") is True

        # Partial match
        assert benchmark._check_action_match("turn on", "turn") is True

        # Case insensitive
        assert benchmark._check_action_match("Turn On", "turn on") is True

        # No match
        assert benchmark._check_action_match("turn on", "dim") is False

        # Unknown expected (always match)
        assert benchmark._check_action_match("anything", "unknown") is True

    def test_device_matching(self):
        """Test device matching logic."""
        from rd5.synthetic.benchmark import WorkflowBenchmark

        benchmark = WorkflowBenchmark()

        # Match in list
        assert benchmark._check_device_match(["lights", "lamp"], "lights") is True

        # Partial match
        assert benchmark._check_device_match(["living room lights"], "lights") is True

        # No match
        assert benchmark._check_device_match(["thermostat"], "lights") is False

        # Unknown expected
        assert benchmark._check_device_match(["anything"], "unknown") is True

    def test_location_matching(self):
        """Test location matching logic."""
        from rd5.synthetic.benchmark import WorkflowBenchmark

        benchmark = WorkflowBenchmark()

        # Both None
        assert benchmark._check_location_match(None, None) is True

        # Expected None
        assert benchmark._check_location_match("living room", None) is True

        # Extracted None but expected something
        assert benchmark._check_location_match(None, "living room") is False

        # Match
        assert benchmark._check_location_match("living room", "living room") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
