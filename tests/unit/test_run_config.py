"""The configuration a run ran with, stated in every report made from it.

`tests/simuhome/bench/run_config.py` reads a run's `run.json` and the reports (`stats.py`
-> `summary.md`, `report.py` -> `report.md`, `prompt_report.py` -> README / pages /
index) open with what it gives. No LLM and no simulator: run.json files are written by
hand into a temp directory, and the reports are rendered from them.

* the record: `Variant.as_dict` writes every axis of the prompt, the default ones too;
* the section: the prompt axes with the option's description, the models, the settings;
  a run.json from before the selection was recorded is read with today's defaults and
  says so; a run without a run.json says that it has none;
* the reports: each one carries the section (or, for the prompt report, the label).
"""

import json

from ami_agents.shared.utils import prompt_variants as pv
from tests.homebench import prompt_report
from tests.simuhome.bench import models, run_config, stats

MODEL = {"model": "ds-r1-7b-16k", "base_url": "http://localhost:11434/v1",
         "temperature": 0.0, "api": "auto"}


def _variant(prompts=None):
    """A variant with one local model on every step, and the given prompt variants."""
    raw = {"models": {"structuring": {"model": "m", "temperature": 0}}}
    if prompts is not None:
        raw["prompt_variants"] = prompts
    return models.variant_from_dict(raw, "arm")


def _run_json(prompts=None, **variant_extra):
    """The content of a run.json for a variant with ``prompts``."""
    variant = _variant(prompts).as_dict()
    variant.update(variant_extra)
    return {"variant": variant, "episodes": ["hb1_a", "hb1_b"], "commit": "abc123",
            "started": "2026-11-09 12:00:00",
            "settings": {"variant": "tests/homebench/variants/arm.yaml", "suite": "homebench",
                         "guidelines": "homebench", "environment": "on", "memory": "off",
                         "context": "none", "stop_after": None, "workers": 1}}


class TestTheRecord:
    def test_a_default_run_records_every_axis_as_the_default(self):
        selection = _variant().as_dict()["prompt_selection"]
        assert selection == {"structuring": {"context": "text", "prompt": "full",
                                             "constraint": "none", "decoding": "free",
                                             "repair": "none"}}

    def test_a_changed_axis_is_recorded_beside_the_defaults(self):
        selection = _variant({"structuring": {"context": "json_v2"}}
                             ).as_dict()["prompt_selection"]
        assert selection["structuring"]["context"] == "json_v2"
        assert selection["structuring"]["prompt"] == "full"

    def test_the_record_does_not_change_what_the_cache_hashes(self):
        """`prompt_selection` is for reading; the step cache key still follows the
        non-default choices only, so earlier caches match."""
        assert _variant({"structuring": {"context": "text"}}).key("structuring") == \
            _variant().key("structuring")
        assert _variant({"structuring": {"context": "json_v2"}}).key("structuring") != \
            _variant({"structuring": {"context": "json"}}).key("structuring")


class TestTheSection:
    def test_it_says_which_version_of_the_json_prompt_ran(self):
        config = run_config.configuration(_run_json({"structuring": {"context": "json_v2"}}))
        assert run_config.prompt_label(config) == "structuring: context=json_v2"
        text = run_config.markdown(config)
        assert "`json_v2`" in text and "**no**" in text
        # The option's own description comes with it, from the code.
        assert "rules reworded" in text or "reworded in the JSON's keys" in text
        # The axes left alone are stated as defaults, not left out.
        assert "`full`" in text and "`none`" in text and "yes" in text

    def test_the_two_json_versions_are_told_apart(self):
        v1 = run_config.prompt_label(run_config.configuration(
            _run_json({"structuring": {"context": "json"}})))
        v2 = run_config.prompt_label(run_config.configuration(
            _run_json({"structuring": {"context": "json_v2"}})))
        assert v1 != v2 and "json_v2" not in v1

    def test_a_default_run_is_labelled_as_such(self):
        config = run_config.configuration(_run_json())
        assert run_config.prompt_label(config) == "default (every axis at its default)"

    def test_models_and_settings_are_stated(self):
        run = _run_json()
        run["variant"]["steps"]["structuring"] = dict(MODEL)
        text = run_config.markdown(run_config.configuration(run))
        assert "`ds-r1-7b-16k`" in text and "http://localhost:11434/v1" in text
        assert "temperature=0.0" in text
        for line in ("- `suite`: homebench", "- `guidelines`: homebench",
                     "- `environment`: on", "- `memory`: off"):
            assert line in text
        assert "commit `abc123`" in text and "2 episode(s)" in text
        # A setting the run did not record (stop_after is None) is not shown.
        assert "stop_after" not in text

    def test_a_run_json_from_before_the_selection_was_recorded_says_so(self):
        run = _run_json({"structuring": {"context": "json"}})
        del run["variant"]["prompt_selection"]
        config = run_config.configuration(run)
        assert config["prompt_reconstructed"] is True
        assert run_config.prompt_label(config) == "structuring: context=json"
        assert "predates the recorded selection" in run_config.markdown(config)
        # A run that has it is not reconstructed.
        assert run_config.configuration(_run_json())["prompt_reconstructed"] is False

    def test_an_option_the_code_no_longer_has_still_renders(self):
        run = _run_json()
        run["variant"]["prompt_selection"]["structuring"]["context"] = "retired"
        config = run_config.configuration(run)
        row = next(r for r in config["prompt"] if r["axis"] == "context")
        assert row["option"] == "retired" and row["default"] is False and row["summary"] == ""
        assert "retired" in run_config.markdown(config)

    def test_a_cell_with_a_pipe_does_not_split_the_table(self):
        lines = run_config._table(["a", "b"], [["x|y", "z"]])
        assert lines[2] == "| x\\|y | z |"

    def test_a_run_without_run_json_says_it_has_none(self, tmp_path):
        assert run_config.load(tmp_path) is None
        text = run_config.markdown_of(tmp_path)
        assert "Not recorded" in text and "no `run.json`" in text
        assert run_config.prompt_label(None) == "unknown (no run.json)"

    def test_load_reads_the_run_json_beside_the_records(self, tmp_path):
        (tmp_path / "run.json").write_text(json.dumps(
            _run_json({"structuring": {"context": "json_v2"}})), encoding="utf-8")
        assert run_config.prompt_label(run_config.load(tmp_path)) == \
            "structuring: context=json_v2"


class TestTheReports:
    def _folder(self, tmp_path, prompts):
        (tmp_path / "run.json").write_text(json.dumps(_run_json(prompts)), encoding="utf-8")
        return tmp_path

    def test_summary_md_opens_with_it(self, tmp_path):
        folder = self._folder(tmp_path, {"structuring": {"context": "json_v2"}})
        stats.write(folder, [], [], "Title")
        summary = (folder / "stats" / "summary.md").read_text(encoding="utf-8")
        assert "## Experiment configuration" in summary and "`json_v2`" in summary
        assert summary.index("## Experiment configuration") < summary.index("## Score")

    def test_summary_md_of_a_run_without_run_json_says_so(self, tmp_path):
        stats.write(tmp_path, [], [], "Title")
        assert "Not recorded" in (tmp_path / "stats" / "summary.md").read_text(encoding="utf-8")

    def test_the_prompt_report_names_the_variants_everywhere(self, tmp_path):
        config = run_config.configuration(_run_json({"structuring": {"context": "json_v2"}}))
        call = {"seq": 1, "step": "structuring.intent", "model": "m", "base_url": None,
                "api": "chat", "started": 0.0, "seconds": 1, "ok": True, "error": None,
                "site": {"file": "f.py", "line": 1, "function": "_structure"},
                "request": {"messages": [{"role": "system", "content": "rules"},
                                         {"role": "user", "content": "request"}],
                            "params": {"temperature": 0}},
                "response": {"content": "{}", "reasoning": None, "tool_calls": [],
                             "finish_reason": "stop", "usage": {}}}
        meta = {"episode": "hb1_x_seed_1", "config": "c.yaml", "config_name": "c",
                "started": "now", "commit": "abc", "input": "request", "outcome": "stopped",
                "models": {"structuring": "m"}, "configuration": config}
        prompt_report.write_report(tmp_path, meta, [call])
        readme = (tmp_path / "README.md").read_text(encoding="utf-8")
        assert "Prompt variants: structuring: context=json_v2" in readme
        assert "## Experiment configuration" in readme and "`json_v2`" in readme
        page = next((tmp_path / "calls").glob("*.md")).read_text(encoding="utf-8")
        assert "structuring: context=json_v2" in page
        assert "context=json_v2" in (tmp_path / "abstract.md").read_text(encoding="utf-8")
        # The report can be written again from its capture, configuration included.
        (tmp_path / "README.md").unlink()
        prompt_report.render_only(tmp_path)
        assert "context=json_v2" in (tmp_path / "README.md").read_text(encoding="utf-8")

    def test_a_capture_without_configuration_says_it_is_not_recorded(self, tmp_path):
        meta = {"episode": "hb1_x_seed_1", "config": "c.yaml", "config_name": "c",
                "started": "now", "commit": "abc", "input": "request", "outcome": "stopped",
                "models": {"structuring": "m"}}
        prompt_report.write_report(tmp_path, meta, [])
        assert "not recorded" in (tmp_path / "README.md").read_text(encoding="utf-8")

    def test_the_index_states_the_variants(self):
        index = prompt_report.index_markdown("M3.pv-json-v2", [],
                                             "structuring: context=json_v2")
        assert "structuring: context=json_v2" in index
        # Callers that pass none still get a line saying so.
        assert "not recorded" in prompt_report.index_markdown("M3", [])


class TestDescribeSelection:
    def test_option_summary_is_the_first_paragraph_of_its_docstring(self):
        summary = pv.option_summary("structuring", "context", "json_v2")
        assert summary.startswith("Version 2 of the JSON prompt")
        assert "\n" not in summary

    def test_an_unknown_option_has_no_summary(self):
        assert pv.option_summary("structuring", "context", "nope") == ""
        assert pv.option_summary("nostep", "context", "text") == ""
