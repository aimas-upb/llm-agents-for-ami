"""HomeBench error report: the taxonomy of one episode and the report built on it.

No LLM is called and no server is started. Records and evaluations are built by hand in
the shape the bench stores (`.json` and `.eval.json`), small enough to read, and pass
through the real `analyze_episode` and `build_report`. The groups, and what each pins:

* normalization: values, room and device-class names to the keys of the tables;
* a part's first failing stage and code, one test per code family that carries a rule:
  format (bare class names, filled `parameter_name`, null `artifact_name`), slot
  confusion, wrong value (a relative goal excepted), a planner refusal on a right
  goal, a wrong value executed, dead structures (timeout, question), a lost part;
* infeasible parts: refusal pooled over parts, substitution, accepted by the home
  (label suspect);
* the roll-up: the earliest stage of any failing part is the episode's;
* the files: `errors.md` / `.json` / two CSVs written for a hand-made run folder, and the
  deterministic reconstruction of a missing `.eval.json` against a stored one when the
  M3 results are present.
"""

import json
from pathlib import Path

import pytest

from tests.homebench import error_report as report
from tests.homebench import error_taxonomy as tax

ROOT = Path(__file__).resolve().parents[2]
M3 = ROOT / "tests" / "homebench" / "results" / "harness" / "homebench.ecai.M3"
TOOLS = tax.Tools.load()

GOLD_BRIGHTNESS = "''' living_room.light.set_brightness(70)'''"
TOKEN = "living_room.light.set_brightness(70)"


# -- builders --------------------------------------------------------------------------


def goal(**overrides):
    """A well-formed structured goal for the living-room light, brightness 70."""
    base = {"goal_specificity": "explicit", "goal_kind": "achievement",
            "location_class": "homeont:LivingRoom", "artifact_class": "homeont:ColorLight",
            "artifact_name": "living_room_light", "affordance_class": "saref:Command",
            "affordance_name": "brightness", "parameter_name": "NA",
            "goal_effect": "set", "target_value_determined": "70"}
    base.update(overrides)
    return base


def structure(text="set the brightness", **overrides):
    """A structuring result holding one goal."""
    return {"text": text, "error": None, "question": None,
            "structure": {"goals": {"G1": goal(**overrides)}}}


def plan(text="set the brightness", outcome="plan", **extra):
    """A plan entry for the structure whose text is ``text``."""
    return {"intent": {"text_intent": text}, "outcome": outcome, "source":
            "explicit_deterministic", "route": "explicit", **extra}


def record(gold=GOLD_BRIGHTNESS, htype="normal", structures=None, plans=None,
           actions=None, query="Set the brightness to 70 in the living room.", name="ep1"):
    """An episode record with the fields the taxonomy reads."""
    return {"episode": name, "case": "feasible", "query": query,
            "meta": {"id": "home5_one_1", "homebench_type": htype, "gold": gold},
            "intents": [{"text": "x", "type": "GOAL_REQUEST"}],
            "structures": structures if structures is not None else [structure()],
            "plans": plans if plans is not None else [plan()],
            "execution": {"mode": "immediate", "plans": [], "actions": actions or []},
            "final_home_state": {"rooms": {"living_room": {"devices": [
                {"device_id": "living_room_light"}]}}},
            "meter": {"calls": []}, "wall_seconds": 1.0}


def evaluation(score, checks, refused=0, clarified=0):
    """An evaluation with the given checks (one per feasible gold part)."""
    return {"score": score, "score_extra": score,
            "breakdown": {"checks": checks},
            "homebench": {"bucket": "normal_single", "refused": refused,
                          "clarified": clarified}}


def check(klass="missing", index=0, **extra):
    """One stored check of a gold call."""
    out = {"goal": index, "device": "living_room_light", "expected": TOKEN,
           "class": klass}
    if klass == "ok":
        out["matched"] = "action"
    return {**out, **extra}


def action(url_device="living_room_light", name="brightness", value=70, ok=True,
           room="living_room"):
    """An executed action request."""
    return {"url": f"http://x/workspaces/home5/{room}/artifacts/{url_device}/actions/"
                   f"{name}", "parameters": {"value": value}, "success": ok,
            "status_code": 200 if ok else 400}


def analyze(rec, ev, source=None):
    """`analyze_episode` with the repo's tables."""
    return tax.analyze_episode(rec, ev, source, TOOLS)


# -- normalization ---------------------------------------------------------------------


def test_norm_value_reads_numbers_booleans_and_words():
    assert tax.norm_value(50.0) == "50"
    assert tax.norm_value("50%") == "50"
    assert tax.norm_value("On") == "true" and tax.norm_value(False) == "false"
    assert tax.norm_value("Fan Only") == "fan_only"
    assert tax.norm_value(None) is None


def test_vocabulary_resolves_prefixed_bare_and_titled_names():
    vocab = tax.Vocabulary()
    assert vocab.room("homeont:DiningRoom") == "ding_room"
    assert vocab.room("Dining Room") == "ding_room" and vocab.room("Foyer") == "foyer"
    assert vocab.room("Aromatherapy") is None
    assert vocab.device_type("homeont:DimmableLight") == "light"
    assert vocab.device_type("AirPurifier") == "air_purifiers"
    assert vocab.device_type("Aromatherapy") == "aromatherapy"


def test_bare_class_fields_ignores_prefixed_and_na():
    assert tax.bare_class_fields(goal()) == []
    assert tax.bare_class_fields(goal(location_class="Foyer", affordance_class="NA")
                                 ) == ["location_class"]


# -- one part: stage and code ----------------------------------------------------------


def test_a_passing_episode_has_no_failing_part():
    out = analyze(record(), evaluation(1, [check("ok")]))
    assert out.verdict == "pass" and out.primary_code is None
    assert [p.ok for p in out.parts] == [True] and out.parts[0].linked


def test_bare_class_names_are_blamed_on_structuring_not_the_planner():
    rec = record(structures=[structure(location_class="LivingRoom",
                                       artifact_class="ColorLight")],
                 plans=[plan(outcome="impossible", explanation="No single device 'X' "
                             "matches the goal's class and room.")])
    out = analyze(rec, evaluation(0, [check()]))
    assert (out.primary_stage, out.primary_code) == ("structuring", "struct_class_prefix")
    assert out.parts[0].detail["bare"] == ["location_class", "artifact_class"]


def test_a_prefixed_goal_the_planner_refuses_is_the_planners():
    rec = record(plans=[plan(outcome="impossible", explanation="No action 'brightness' "
                             "on 'living_room_light'.")])
    out = analyze(rec, evaluation(0, [check()]))
    assert (out.primary_stage, out.primary_code) == ("planning", "plan_no_action")


def test_a_filled_parameter_name_refusal_is_structurings():
    explanation = ("Cannot invoke living_room_light.brightness with '70': the action "
                   "takes a direct value, not parameter 'brightness'.")
    rec = record(structures=[structure(parameter_name="brightness")],
                 plans=[plan(outcome="impossible", explanation=explanation)])
    assert analyze(rec, evaluation(0, [check()])).primary_code == "struct_parameter_name"


def test_an_unnamed_device_that_ends_unrouted_is_structurings():
    rec = record(structures=[structure(artifact_name=None)],
                 plans=[plan(outcome="error", error="unrouted")])
    assert analyze(rec, evaluation(0, [check()])).primary_code == "struct_device_unnamed"


def test_a_device_class_in_the_room_field_is_a_slot_confusion():
    rec = record(structures=[structure(location_class="Curtain")])
    out = analyze(rec, evaluation(0, [check()]))
    assert out.primary_code == "struct_slot_confusion"
    assert out.parts[0].detail["diffs"]["slot_confusion"][1] == "Curtain"


def test_wrong_value_is_structurings_but_a_relative_goal_is_not_compared():
    wrong = analyze(record(structures=[structure(target_value_determined="40")]),
                    evaluation(0, [check()]))
    assert wrong.primary_code == "struct_wrong_value"
    relative = analyze(record(structures=[structure(target_value_determined="40",
                                                    goal_effect="modify")]),
                       evaluation(0, [check()]))
    assert relative.primary_code == "exec_no_effect"


def test_a_right_goal_and_plan_with_a_wrong_value_sent_is_execution():
    rec = record(actions=[action(value=100)])
    out = analyze(rec, evaluation(0, [check("wrong_value", values=[
        "living_room.light.set_brightness(100)"])]))
    assert (out.primary_stage, out.primary_code) == ("execution", "exec_wrong_value")


def test_a_rejected_action_is_named():
    rec = record(actions=[action(ok=False)])
    assert analyze(rec, evaluation(0, [check()])).primary_code == "exec_action_rejected"


def test_a_part_with_no_goal_is_explained_by_the_dead_structure():
    timed_out = {"text": "set the brightness in the living room", "error":
                 "Request timed out.", "structure": None}
    question = {"text": "set the brightness in the living room", "structure": None,
                "question": "I can't find the light in the living room. What now?"}
    json_bad = {"text": "x", "structure": None, "error": "expected a JSON object, got"}
    for dead, code in ((timed_out, "infra_llm_timeout"),
                       (question, "struct_false_clarify_device"),
                       (json_bad, "struct_json_invalid")):
        out = analyze(record(structures=[dead], plans=[]), evaluation(0, [check()]))
        assert out.primary_code == code, code


def test_a_part_with_no_goal_and_no_dead_structure_is_lost_in_segmentation():
    out = analyze(record(structures=[], plans=[]), evaluation(0, [check()]))
    assert (out.primary_stage, out.primary_code) == ("segmentation", "seg_part_lost")


def test_a_lenient_only_pass_is_an_evaluation_code():
    lenient = check("ok")
    lenient["matched"] = "turn_off"
    out = analyze(record(), evaluation(0, [lenient]))
    assert (out.primary_stage, out.primary_code) == ("evaluation", "eval_turn_off_reading")


def test_infrastructure_verdict_carries_its_code():
    ev = {"score": -1, "error_type": "Infrastructure",
          "detail": "LLM calls failed in ['segmentation']"}
    out = analyze(record(), ev)
    assert out.verdict == "infra" and out.primary_code == "infra_llm_timeout"


def test_no_evaluation_is_unscored_and_nothing_is_diagnosed():
    out = analyze(record(), None)
    assert out.verdict == "unscored" and out.parts == []


# -- infeasible parts ------------------------------------------------------------------

SOURCE = {"eval": {"goals": [{"room_id": "kitchen", "device_id": "kitchen_heating",
                              "device_type": "heating", "feasibility": False,
                              "reason": "unexist_device", "resolved": True}]}}


def infeasible_record(actions):
    """A single infeasible request with the given executed actions."""
    return record(gold="'''error_input'''", htype="unexist_device", structures=[],
                  plans=[], actions=actions)


def test_a_refusal_covers_the_infeasible_part():
    out = analyze(infeasible_record([]), evaluation(1, [], refused=1), SOURCE)
    assert out.verdict == "pass" and out.parts[0].ok


def test_operating_another_device_in_the_room_is_a_substitution():
    acts = [action("kitchen_air_conditioner", "mode", "heat", room="kitchen")]
    out = analyze(infeasible_record(acts), evaluation(0, []), SOURCE)
    assert out.primary_code == "infeas_substituted"
    assert out.extra_actions == ["kitchen.air_conditioner.set_mode(heat)"]


def test_the_home_accepting_an_action_on_the_forbidden_device_is_a_label_suspect():
    src = {"eval": {"goals": [{**SOURCE["eval"]["goals"][0], "device_id": "kitchen_light",
                               "device_type": "light", "reason": "unexist_attribute"}]}}
    acts = [action("kitchen_light", "brightness", 50, room="kitchen")]
    out = analyze(infeasible_record(acts), evaluation(0, []), src)
    assert out.primary_code == "infeas_accepted_on_target"
    assert out.parts[0].label_suspect
    assert report.blame([out]) == {"label_or_metric": 1}


# -- the roll-up -----------------------------------------------------------------------


def part(code, ok=False):
    """A diagnosed part with the given code."""
    return tax.PartDiagnosis(0, "t", True, None, "light", "kitchen", "turn_on", ok=ok,
                             stage=tax.CODES[code][0] if code else None, code=code)


def test_the_earliest_stage_of_any_failing_part_is_the_episodes():
    parts = [part("exec_wrong_value"), part("struct_wrong_room"), part("struct_wrong_room")]
    assert tax.rollup(parts)[:2] == ("structuring", "struct_wrong_room")
    assert tax.rollup([part(None, ok=True)]) == (None, None, [])


def test_every_code_names_a_known_stage():
    assert {stage for stage, _ in tax.CODES.values()} <= set(tax.STAGES)


def test_parts_of_one_request_link_to_their_own_goals():
    two = record(gold="'''living_room.light.set_brightness(70),kitchen.light.turn_off(),'''",
                 htype="multi2_normal")
    parts = tax.build_parts(two, None, TOOLS)
    kitchen = goal(location_class="homeont:Kitchen", artifact_name="kitchen_light",
                   affordance_name="onOff", target_value_determined="false")
    refs = [tax.GoalRef(0, "G1", kitchen, "a"), tax.GoalRef(1, "G1", goal(), "b")]
    links = tax.link_parts(parts, refs, TOOLS)
    assert links[0].goal["artifact_name"] == "living_room_light"
    assert links[1].goal["artifact_name"] == "kitchen_light"


# -- the report and its files ----------------------------------------------------------


def write_run(folder: Path, episodes):
    """A run folder with run.json and one record + evaluation per ``(rec, ev)``."""
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({
        "variant": {"name": folder.name, "steps": {"structuring": {"model": "m"}}},
        "episodes": [rec["episode"] for rec, _ in episodes] + ["gone"]}), encoding="utf-8")
    for rec, ev in episodes:
        stem = folder / rec["episode"]
        stem.with_suffix(".json").write_text(json.dumps(rec), encoding="utf-8")
        if ev is not None:
            (folder / f"{rec['episode']}.eval.json").write_text(json.dumps(ev),
                                                               encoding="utf-8")


def test_the_cli_writes_the_four_files_for_a_hand_made_run(tmp_path):
    bare = record(name="hb1_feasible_seed_1", structures=[structure(
        location_class="LivingRoom")], plans=[plan(outcome="impossible", explanation=
                                                    "No single device 'X' matches.")])
    good = record(name="hb1_feasible_seed_2")
    run_dir = tmp_path / "run"
    write_run(run_dir, [(bare, evaluation(0, [check()])),
                        (good, evaluation(1, [check("ok")]))])
    assert report.main([str(run_dir), "--episodes", str(tmp_path / "none")]) == 0
    stats = run_dir / "stats"
    data = json.loads((stats / "errors.json").read_text(encoding="utf-8"))
    assert data["overview"]["verdicts"] == {"fail": 1, "pass": 1}
    assert data["overview"]["missing"] == ["gone"]
    assert data["stage_codes"]["episode_code"] == {"struct_class_prefix": 1}
    text = (stats / "errors.md").read_text(encoding="utf-8")
    assert "## 2. Where the pipeline loses requests" in text
    assert "struct_class_prefix" in text
    episodes = (stats / "errors_episodes.csv").read_text(encoding="utf-8").splitlines()
    parts = (stats / "errors_parts.csv").read_text(encoding="utf-8").splitlines()
    assert len(episodes) == 3 and len(parts) == 3


def test_goal_format_shows_the_planner_outcome_by_class_format(tmp_path):
    bare = record(name="hb1_feasible_seed_1", structures=[structure(
        location_class="LivingRoom")], plans=[plan(outcome="impossible")])
    good = record(name="hb1_feasible_seed_2")
    run_dir = tmp_path / "run"
    write_run(run_dir, [(bare, evaluation(0, [check()])), (good, evaluation(1, [check("ok")]))])
    rows = {r["group"]: r for r in report.goal_format(
        report.load_run(run_dir, tmp_path / "none"))["by_format"]}
    assert rows["class format: bare class field"]["plan"] == 0
    assert rows["class format: all prefixed / NA"]["plan_rate"] == 100.0


def test_an_unreadable_folder_has_no_records(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        report.main([str(tmp_path / "empty")])


@pytest.mark.skipif(not M3.is_dir(), reason="M3 results not present")
def test_reconstructed_evaluation_equals_the_stored_one():
    names = sorted(p.name[:-5] for p in M3.glob("hb*.json")
                   if not p.name.endswith(".eval.json"))[:25]
    run = report.load_run(M3, report.DEFAULT_BENCHMARK)
    for name in names:
        stored = json.loads((M3 / f"{name}.eval.json").read_text(encoding="utf-8"))
        rebuilt = report.reconstruct_evaluation(run.records[name], run.sources[name])
        assert (rebuilt["score"], rebuilt.get("score_extra")) == (
            stored["score"], stored.get("score_extra")), name
