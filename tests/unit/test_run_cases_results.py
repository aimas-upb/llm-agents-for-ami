"""
Unit tests for the results-CSV generation-mode columns in
tests/e2e-lab308e/run_cases.py (loaded via importlib — it is a script,
not a package module).
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_RUN_CASES_PATH = Path(__file__).resolve().parents[1] / "e2e-lab308e" / "run_cases.py"


@pytest.fixture(scope="module")
def run_cases():
    spec = importlib.util.spec_from_file_location("run_cases", _RUN_CASES_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_cases"] = module
    spec.loader.exec_module(module)
    return module


def _code_mode_plan() -> str:
    return json.dumps(
        {
            "plan_type": "behavior_tree",
            "plan_mode": "python_code",
            "generated_code": "tree = action('light308/turnOn')",
            "tree": {"name": "N", "type": "action", "action_url": "http://h/a"},
            "signifier_reuse": False,
        }
    )


class TestExtractGenerationFields:
    def test_reads_envelope_fields(self, run_cases):
        fields = run_cases._extract_generation_fields(_code_mode_plan())
        assert fields == {
            "generation_mode": "python_code",
            "generated_code": "tree = action('light308/turnOn')",
        }

    def test_falls_back_to_harness_mode(self, run_cases):
        fields = run_cases._extract_generation_fields(None, "behavior_tree")
        assert fields == {"generation_mode": "behavior_tree", "generated_code": ""}

    def test_signifier_reuse_row_has_mode_but_no_code(self, run_cases):
        plan = json.dumps(
            {"plan_mode": "python_code", "signifier_reuse": True, "tree": {"type": "action"}}
        )
        fields = run_cases._extract_generation_fields(plan, "python_code")
        assert fields["generation_mode"] == "python_code"
        assert fields["generated_code"] == ""


class TestResultColumns:
    def test_new_columns_are_last(self, run_cases):
        assert run_cases.RESULT_COLUMNS[-2:] == ["generation_mode", "generated_code"]

    def test_append_writes_generation_columns(self, run_cases, tmp_path):
        result = run_cases.CaseResult(
            path=Path("simuhome_qt2_feasible_seed_1.json"),
            passed=True,
            details={
                "case": {"name": "simuhome_qt2_feasible_seed_1"},
                "model_name": "qwen2.5-coder:3b",
                "plan": _code_mode_plan(),
            },
        )
        out = tmp_path / "results.csv"
        run_cases._append_results_csv([result], out, default_generation_mode="python_code")

        with out.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["generation_mode"] == "python_code"
        assert rows[0]["generated_code"] == "tree = action('light308/turnOn')"

    def test_old_csv_is_migrated_on_append(self, run_cases, tmp_path):
        out = tmp_path / "results.csv"
        old_columns = [c for c in run_cases.RESULT_COLUMNS if c not in ("generation_mode", "generated_code")]
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=old_columns)
            writer.writeheader()
            writer.writerow({c: "" for c in old_columns} | {"test_name": "legacy_row"})

        run_cases._ensure_results_csv_header(out)

        with out.open(newline="") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == run_cases.RESULT_COLUMNS
            rows = list(reader)
        assert rows[0]["test_name"] == "legacy_row"
        assert rows[0]["generation_mode"] == ""
