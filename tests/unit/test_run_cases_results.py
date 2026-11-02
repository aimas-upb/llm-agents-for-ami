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
        assert run_cases.RESULT_COLUMNS[-3:] == ["generation_mode", "generated_code", "planner_model_name"]

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
        # Single-model run: the planner column mirrors model_name.
        assert rows[0]["planner_model_name"] == "qwen2.5-coder:3b"

    def test_append_writes_planner_model_for_role_split(self, run_cases, tmp_path):
        result = run_cases.CaseResult(
            path=Path("simuhome_qt2_feasible_seed_1.json"),
            passed=False,
            details={
                "case": {"name": "simuhome_qt2_feasible_seed_1"},
                "model_name": "gpt-5-mini",
                "planner_model_name": "qwen2.5-coder:3b",
                "plan": _code_mode_plan(),
            },
        )
        out = tmp_path / "results.csv"
        run_cases._append_results_csv([result], out)

        with out.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["model_name"] == "gpt-5-mini"
        assert rows[0]["planner_model_name"] == "qwen2.5-coder:3b"

    def test_old_csv_is_migrated_on_append(self, run_cases, tmp_path):
        out = tmp_path / "results.csv"
        new_columns = ("generation_mode", "generated_code", "planner_model_name")
        old_columns = [c for c in run_cases.RESULT_COLUMNS if c not in new_columns]
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=old_columns)
            writer.writeheader()
            writer.writerow(
                {c: "" for c in old_columns} | {"test_name": "legacy_row", "model_name": "codegemma:7b-instruct"}
            )

        run_cases._ensure_results_csv_header(out)

        with out.open(newline="") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == run_cases.RESULT_COLUMNS
            rows = list(reader)
        assert rows[0]["test_name"] == "legacy_row"
        assert rows[0]["generation_mode"] == ""
        # Pre-role-split rows were single-model runs, so the planner was model_name.
        assert rows[0]["planner_model_name"] == "codegemma:7b-instruct"


class TestPlannerOverride:
    def test_no_env_means_single_model_protocol(self, run_cases, monkeypatch):
        monkeypatch.delenv("PLANNER_MODEL", raising=False)
        assert run_cases._planner_override_from_env() is None

    def test_env_is_read(self, run_cases, monkeypatch):
        monkeypatch.setenv("PLANNER_MODEL", "qwen2.5-coder:3b")
        monkeypatch.setenv("PLANNER_BASE_URL", "http://127.0.0.1:11500/v1")
        monkeypatch.setenv("PLANNER_API_KEY", "dummy")
        monkeypatch.delenv("PLANNER_REASONING_EFFORT", raising=False)
        assert run_cases._planner_override_from_env() == {
            "model": "qwen2.5-coder:3b",
            "base_url": "http://127.0.0.1:11500/v1",
            "api_key": "dummy",
            "reasoning_effort": "",
        }

    def test_override_lands_in_planning_sections_only(self, run_cases):
        config = {
            "llm": {"providers": {"openai": {"model": "gpt-5-mini", "base_url": "https://api.openai.com/v1"}}},
            "planning": {"llm_planning": {"output_format": "py_trees_code"}},
        }
        run_cases._apply_planner_override(
            config,
            model="qwen2.5-coder:3b",
            base_url="http://127.0.0.1:11500/v1",
            api_key="dummy",
        )
        for section in (config["planning"], config["interaction_solver"]["planning"]):
            planning = section["llm_planning"]
            assert planning["model"] == "qwen2.5-coder:3b"
            assert planning["base_url"] == "http://127.0.0.1:11500/v1"
            assert planning["api_key"] == "dummy"
            # Present-but-empty: the ISA factory must not fall back to the
            # shared provider's reasoning effort for a local planner.
            assert planning["reasoning_effort"] == ""
            assert "temperature" not in planning
        # Existing planning keys and the shared (UserAssistant) provider are untouched.
        assert config["planning"]["llm_planning"]["output_format"] == "py_trees_code"
        assert config["llm"]["providers"]["openai"]["model"] == "gpt-5-mini"

    def test_override_temperature_is_optional(self, run_cases):
        config = run_cases._apply_planner_override(
            {}, model="gpt-5-mini", base_url="https://api.openai.com/v1", api_key="sk", temperature=1.0
        )
        assert config["planning"]["llm_planning"]["temperature"] == 1.0

    def test_default_base_url_follows_tag_shape(self, run_cases):
        assert run_cases._default_base_url("gpt-4o") == "https://api.openai.com/v1"
        assert run_cases._default_base_url("qwen2.5-coder:3b") == "https://openrouter.ai/api/v1"
