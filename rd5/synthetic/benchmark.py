"""Benchmark framework for RD5 workflow evaluation.

This module provides tools to benchmark the plan generation workflow
against synthetic test scenarios.
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from statistics import mean, stdev

from rd5.synthetic.data_generator import TestScenario, generate_test_scenarios


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark scenario."""

    scenario_id: str
    user_request: str
    category: str
    difficulty: str

    # Workflow results
    workflow_success: bool
    intent_extracted: bool
    code_generated: bool
    code_valid: bool
    execution_success: bool
    plan_stored: bool

    # Accuracy metrics
    action_verb_match: bool
    target_device_match: bool
    location_match: bool

    # Timing
    total_time_ms: int
    intent_time_ms: int = 0
    generation_time_ms: int = 0
    validation_time_ms: int = 0
    execution_time_ms: int = 0

    # Details
    extracted_intent: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Summary statistics from a benchmark run."""

    # Run info
    run_id: str
    timestamp: str
    total_scenarios: int
    duration_seconds: float

    # Success rates
    workflow_success_rate: float
    intent_extraction_rate: float
    code_generation_rate: float
    validation_pass_rate: float
    execution_success_rate: float
    plan_storage_rate: float

    # Accuracy rates
    action_verb_accuracy: float
    target_device_accuracy: float
    location_accuracy: float

    # Timing statistics
    avg_total_time_ms: float
    avg_intent_time_ms: float
    avg_generation_time_ms: float
    avg_validation_time_ms: float
    avg_execution_time_ms: float

    # By category
    success_by_category: Dict[str, float] = field(default_factory=dict)
    success_by_difficulty: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def print_summary(self) -> None:
        """Print a formatted summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Run ID: {self.run_id}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Total scenarios: {self.total_scenarios}")
        print(f"Duration: {self.duration_seconds:.2f}s")
        print()
        print("SUCCESS RATES:")
        print(f"  Workflow success:     {self.workflow_success_rate:.1%}")
        print(f"  Intent extraction:    {self.intent_extraction_rate:.1%}")
        print(f"  Code generation:      {self.code_generation_rate:.1%}")
        print(f"  Validation pass:      {self.validation_pass_rate:.1%}")
        print(f"  Execution success:    {self.execution_success_rate:.1%}")
        print(f"  Plan storage:         {self.plan_storage_rate:.1%}")
        print()
        print("ACCURACY RATES:")
        print(f"  Action verb:          {self.action_verb_accuracy:.1%}")
        print(f"  Target device:        {self.target_device_accuracy:.1%}")
        print(f"  Location:             {self.location_accuracy:.1%}")
        print()
        print("TIMING (avg):")
        print(f"  Total:                {self.avg_total_time_ms:.0f}ms")
        print(f"  Intent extraction:    {self.avg_intent_time_ms:.0f}ms")
        print(f"  Code generation:      {self.avg_generation_time_ms:.0f}ms")
        print(f"  Validation:           {self.avg_validation_time_ms:.0f}ms")
        print(f"  Execution:            {self.avg_execution_time_ms:.0f}ms")
        print()
        if self.success_by_category:
            print("BY CATEGORY:")
            for cat, rate in sorted(self.success_by_category.items()):
                print(f"  {cat}: {rate:.1%}")
        if self.success_by_difficulty:
            print("BY DIFFICULTY:")
            for diff, rate in sorted(self.success_by_difficulty.items()):
                print(f"  {diff}: {rate:.1%}")
        print("=" * 60)


class WorkflowBenchmark:
    """Benchmark runner for RD5 workflow."""

    def __init__(
        self,
        use_docker: bool = False,
        max_concurrency: int = 1,
        timeout_per_scenario: int = 60,
    ):
        """Initialize benchmark runner.

        Args:
            use_docker: Whether to use Docker sandbox.
            max_concurrency: Max concurrent workflow runs.
            timeout_per_scenario: Timeout per scenario in seconds.
        """
        self.use_docker = use_docker
        self.max_concurrency = max_concurrency
        self.timeout_per_scenario = timeout_per_scenario
        self.results: List[BenchmarkResult] = []

    async def run_single_scenario(
        self,
        scenario: TestScenario,
    ) -> BenchmarkResult:
        """Run a single benchmark scenario.

        Args:
            scenario: Test scenario to run.

        Returns:
            BenchmarkResult with metrics.
        """
        from rd5.workflow.graph import run_plan_generation

        start_time = time.time()
        errors = []

        try:
            # Run workflow
            final_state = await asyncio.wait_for(
                run_plan_generation(
                    user_request=scenario.user_request,
                    request_id=scenario.scenario_id,
                ),
                timeout=self.timeout_per_scenario,
            )

            total_time_ms = int((time.time() - start_time) * 1000)

            # Extract results
            extracted_intent = final_state.get("extracted_intent", {})
            validation_result = final_state.get("validation_result", {})
            execution_result = final_state.get("execution_result", {})

            # Determine success metrics
            intent_extracted = bool(extracted_intent.get("intent"))
            code_generated = bool(final_state.get("generated_code"))
            code_valid = validation_result.get("is_valid", False)
            execution_success = execution_result.get("success", False)
            plan_stored = bool(final_state.get("plan_id"))

            from rd5.workflow.state import WorkflowStatus
            workflow_success = final_state.get("workflow_status") == WorkflowStatus.COMPLETED

            # Check accuracy against expected values
            action_verb_match = self._check_action_match(
                extracted_intent.get("action_verb", ""),
                scenario.expected_action_verb,
            )
            target_device_match = self._check_device_match(
                extracted_intent.get("target_objects", []),
                scenario.expected_target_device,
            )
            location_match = self._check_location_match(
                extracted_intent.get("location"),
                scenario.expected_location,
            )

            errors.extend(final_state.get("errors", []))

            return BenchmarkResult(
                scenario_id=scenario.scenario_id,
                user_request=scenario.user_request,
                category=scenario.category,
                difficulty=scenario.difficulty,
                workflow_success=workflow_success,
                intent_extracted=intent_extracted,
                code_generated=code_generated,
                code_valid=code_valid,
                execution_success=execution_success,
                plan_stored=plan_stored,
                action_verb_match=action_verb_match,
                target_device_match=target_device_match,
                location_match=location_match,
                total_time_ms=total_time_ms,
                execution_time_ms=execution_result.get("execution_time_ms", 0),
                extracted_intent=extracted_intent,
                errors=errors,
            )

        except asyncio.TimeoutError:
            total_time_ms = int((time.time() - start_time) * 1000)
            return BenchmarkResult(
                scenario_id=scenario.scenario_id,
                user_request=scenario.user_request,
                category=scenario.category,
                difficulty=scenario.difficulty,
                workflow_success=False,
                intent_extracted=False,
                code_generated=False,
                code_valid=False,
                execution_success=False,
                plan_stored=False,
                action_verb_match=False,
                target_device_match=False,
                location_match=False,
                total_time_ms=total_time_ms,
                errors=["Scenario timed out"],
            )

        except Exception as e:
            total_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Benchmark error for {scenario.scenario_id}: {e}")
            return BenchmarkResult(
                scenario_id=scenario.scenario_id,
                user_request=scenario.user_request,
                category=scenario.category,
                difficulty=scenario.difficulty,
                workflow_success=False,
                intent_extracted=False,
                code_generated=False,
                code_valid=False,
                execution_success=False,
                plan_stored=False,
                action_verb_match=False,
                target_device_match=False,
                location_match=False,
                total_time_ms=total_time_ms,
                errors=[str(e)],
            )

    def _check_action_match(self, extracted: str, expected: str) -> bool:
        """Check if action verbs match."""
        if not expected or expected == "unknown":
            return True  # No expected value to match

        extracted_lower = extracted.lower()
        expected_lower = expected.lower()

        # Direct match
        if extracted_lower == expected_lower:
            return True

        # Partial match (e.g., "turn on" matches "turn")
        if expected_lower in extracted_lower or extracted_lower in expected_lower:
            return True

        return False

    def _check_device_match(self, extracted: List[str], expected: str) -> bool:
        """Check if target device matches."""
        if not expected or expected == "unknown":
            return True

        expected_lower = expected.lower()

        for device in extracted:
            if expected_lower in device.lower() or device.lower() in expected_lower:
                return True

        return False

    def _check_location_match(
        self,
        extracted: Optional[str],
        expected: Optional[str],
    ) -> bool:
        """Check if location matches."""
        # Both None is a match
        if expected is None and extracted is None:
            return True

        # Expected None but extracted something - still OK
        if expected is None:
            return True

        # Expected something but got None
        if extracted is None:
            return False

        return expected.lower() in extracted.lower()

    async def run_benchmark(
        self,
        scenarios: List[TestScenario],
        progress_callback: Optional[callable] = None,
    ) -> BenchmarkSummary:
        """Run benchmark on a set of scenarios.

        Args:
            scenarios: List of test scenarios.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            BenchmarkSummary with aggregated results.
        """
        import uuid

        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        self.results = []

        logger.info(f"Starting benchmark {run_id} with {len(scenarios)} scenarios")

        # Run scenarios
        for i, scenario in enumerate(scenarios):
            result = await self.run_single_scenario(scenario)
            self.results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(scenarios))

            logger.debug(
                f"[{i+1}/{len(scenarios)}] {scenario.scenario_id}: "
                f"{'OK' if result.workflow_success else 'FAIL'}"
            )

        duration = time.time() - start_time

        # Calculate summary
        summary = self._calculate_summary(run_id, duration)

        logger.info(f"Benchmark {run_id} completed in {duration:.2f}s")

        return summary

    def _calculate_summary(
        self,
        run_id: str,
        duration: float,
    ) -> BenchmarkSummary:
        """Calculate summary statistics from results."""

        total = len(self.results)
        if total == 0:
            return BenchmarkSummary(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                total_scenarios=0,
                duration_seconds=duration,
                workflow_success_rate=0,
                intent_extraction_rate=0,
                code_generation_rate=0,
                validation_pass_rate=0,
                execution_success_rate=0,
                plan_storage_rate=0,
                action_verb_accuracy=0,
                target_device_accuracy=0,
                location_accuracy=0,
                avg_total_time_ms=0,
                avg_intent_time_ms=0,
                avg_generation_time_ms=0,
                avg_validation_time_ms=0,
                avg_execution_time_ms=0,
            )

        # Success rates
        workflow_success = sum(1 for r in self.results if r.workflow_success) / total
        intent_extraction = sum(1 for r in self.results if r.intent_extracted) / total
        code_generation = sum(1 for r in self.results if r.code_generated) / total
        validation_pass = sum(1 for r in self.results if r.code_valid) / total
        execution_success = sum(1 for r in self.results if r.execution_success) / total
        plan_storage = sum(1 for r in self.results if r.plan_stored) / total

        # Accuracy rates
        action_accuracy = sum(1 for r in self.results if r.action_verb_match) / total
        device_accuracy = sum(1 for r in self.results if r.target_device_match) / total
        location_accuracy = sum(1 for r in self.results if r.location_match) / total

        # Timing averages
        avg_total = mean(r.total_time_ms for r in self.results)
        avg_execution = mean(r.execution_time_ms for r in self.results)

        # By category
        categories = set(r.category for r in self.results)
        success_by_category = {}
        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            if cat_results:
                success_by_category[cat] = sum(
                    1 for r in cat_results if r.workflow_success
                ) / len(cat_results)

        # By difficulty
        difficulties = set(r.difficulty for r in self.results)
        success_by_difficulty = {}
        for diff in difficulties:
            diff_results = [r for r in self.results if r.difficulty == diff]
            if diff_results:
                success_by_difficulty[diff] = sum(
                    1 for r in diff_results if r.workflow_success
                ) / len(diff_results)

        return BenchmarkSummary(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            total_scenarios=total,
            duration_seconds=duration,
            workflow_success_rate=workflow_success,
            intent_extraction_rate=intent_extraction,
            code_generation_rate=code_generation,
            validation_pass_rate=validation_pass,
            execution_success_rate=execution_success,
            plan_storage_rate=plan_storage,
            action_verb_accuracy=action_accuracy,
            target_device_accuracy=device_accuracy,
            location_accuracy=location_accuracy,
            avg_total_time_ms=avg_total,
            avg_intent_time_ms=0,  # Would need workflow timing hooks
            avg_generation_time_ms=0,
            avg_validation_time_ms=0,
            avg_execution_time_ms=avg_execution,
            success_by_category=success_by_category,
            success_by_difficulty=success_by_difficulty,
        )

    def save_results(
        self,
        filepath: str,
        include_summary: bool = True,
    ) -> None:
        """Save results to JSON file.

        Args:
            filepath: Output file path.
            include_summary: Whether to include summary in output.
        """
        data = {
            "results": [r.to_dict() for r in self.results],
        }

        if include_summary and self.results:
            summary = self._calculate_summary("saved", 0)
            data["summary"] = summary.to_dict()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


async def run_quick_benchmark(
    num_scenarios: int = 10,
    seed: int = 42,
) -> BenchmarkSummary:
    """Run a quick benchmark with synthetic data.

    Args:
        num_scenarios: Number of scenarios to generate.
        seed: Random seed for reproducibility.

    Returns:
        BenchmarkSummary with results.
    """
    # Generate scenarios
    scenarios = generate_test_scenarios(
        count=num_scenarios,
        include_complex=True,
        include_edge_cases=True,
        seed=seed,
    )

    # Run benchmark
    benchmark = WorkflowBenchmark(use_docker=False)
    summary = await benchmark.run_benchmark(scenarios)

    summary.print_summary()

    return summary


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Run quick benchmark
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    asyncio.run(run_quick_benchmark(num_scenarios=num))
