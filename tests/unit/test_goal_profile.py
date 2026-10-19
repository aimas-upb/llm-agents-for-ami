"""The profiling checklist and the resolution of the model's answers."""

import pytest

from ami_agents.agents.interaction_solver.utils.goal_profile import (
    InconsistentStructure,
    ProfileOutputError,
    build_checklist,
    fallback_profile,
    render_goal_profile_prompt,
    resolve_profile,
)
from ami_agents.shared.models.goal_structure import GoalStructure


def _structure(goals, predicates=None):
    return GoalStructure.from_dict({"goals": goals, "predicates": predicates or {}},
                                   intent_text="x", structure="dependency")


ACHIEVE = {"goal_specificity": "explicit", "goal_kind": "achievement"}
MAINTAIN = {"goal_specificity": "explicit", "goal_kind": "maintenance"}


class TestChecklist:
    def test_each_rule(self):
        checklist = build_checklist(_structure(
            {"G1": {**MAINTAIN, "timing": {"starts": "now", "ends": {"kind": "duration",
                                                                     "text": "1 hour"}}},
             "G2": {**ACHIEVE, "timing": {"starts": "now"}},
             "G3": {**ACHIEVE, "timing": {"starts": "after_goal", "ref_goal": "G2"}},
             "G4": {**ACHIEVE, "timing": {"completes_with": "G2"}},
             "G5": {**ACHIEVE, "timing": {"starts": "after_goal", "ref_goal": "G1"}},
             "G6": {**ACHIEVE, "timing": {"starts": "on_trigger"}}},
            {"P1": {"predicate_subject": "device_property", "role": "trigger",
                    "applies_to": ["G6"], "offset_direction": "before",
                    "offset_text": "1 minute"}}))
        assert [(c["question"], c["target"], c.get("ref")) for c in checklist] == [
            ("maintenance_profile", "G1", None),
            ("dependency_anchor", "G3", "G2"),
            ("completion_timing", "G4", "G2"),
            ("time_to_event", "P1", None)]
        assert [c["id"] for c in checklist] == ["Q1", "Q2", "Q3", "Q4"]

    def test_completing_with_an_event_with_a_start_given(self):
        """Seed 46: "start dryer 1 now so it and dishwasher 1 finish
        together". The start is the user's, but the run length and the
        event's time left are still asked, for the check before planning."""
        checklist = build_checklist(_structure(
            {"G1": {**ACHIEVE, "timing": {"starts": "now", "completes_with": "P1"}}},
            {"P1": {"predicate_subject": "device_property", "role": "trigger",
                    "applies_to": ["G1"]}}))
        assert [(c["question"], c["target"], c.get("ref")) for c in checklist] == [
            ("completion_timing", "G1", "P1"), ("time_to_event", "P1", None)]

    def test_the_next_instruction_to_the_same_device_asks_nothing(self):
        same = {**ACHIEVE, "artifact_name": "ac_1"}
        checklist = build_checklist(_structure(
            {"G1": {**same, "timing": {"starts": "now"}},
             "G2": {**same, "timing": {"starts": "after_goal", "ref_goal": "G1"}},
             "G3": {**same, "timing": {"starts": "after_goal", "ref_goal": "G2",
                                       "offset_text": "5 minutes"}},
             "G4": {**ACHIEVE, "artifact_name": "fan_1",
                    "timing": {"starts": "after_goal", "ref_goal": "G1"}}}))
        # A wait, or another device, still needs the anchor.
        assert [(c["question"], c["target"]) for c in checklist] == [
            ("dependency_anchor", "G3"), ("dependency_anchor", "G4")]

    def test_waiting_for_a_maintenance_goal_that_never_ends_is_inconsistent(self):
        with pytest.raises(InconsistentStructure, match="never|no scope"):
            build_checklist(_structure(
                {"G1": {**MAINTAIN, "timing": {"starts": "now"}},
                 "G2": {**ACHIEVE, "timing": {"starts": "after_goal", "ref_goal": "G1"}}}))

    def test_a_scope_ends_the_maintenance_goal(self):
        checklist = build_checklist(_structure(
            {"G1": {**MAINTAIN, "timing": {"starts": "now"}},
             "G2": {**ACHIEVE, "timing": {"starts": "after_goal", "ref_goal": "G1"}}},
            {"P1": {"role": "scope", "applies_to": ["G1"]}}))
        assert [c["question"] for c in checklist] == ["maintenance_profile"]

    def test_the_prompt_takes_its_three_inputs(self):
        prompt = render_goal_profile_prompt("STRUCT", "CTX", "LIST")
        assert prompt.rstrip().endswith("checklist:\nLIST")
        assert "goal_structure:\nSTRUCT" in prompt and "{" in prompt   # JSON kept


CHECKLIST = [{"id": "Q1", "question": "dependency_anchor", "target": "G2", "ref": "G1"},
             {"id": "Q2", "question": "time_to_event", "target": "P1"}]
REF = {"artifact_name": "washer_1", "property_name": "operationalState"}


class TestResolve:
    def test_answers_are_kept_and_missing_ones_fall_back(self):
        resolved, log = resolve_profile(CHECKLIST, {"answers": [
            {"id": "Q1", "answer": {"anchor": "condition", "condition": {"ref": REF}}}]})
        assert resolved["Q1"]["source"] == "model"
        assert resolved["Q2"]["answer"] == {"fallback": "at_event"}
        assert "Q2 time_to_event P1: fallback applied (not returned by the model)" in log

    def test_an_unresolved_question_falls_back_with_its_reason(self):
        resolved, _ = resolve_profile(CHECKLIST, {"unresolved": [
            {"id": "Q1", "reason": "no end state"}]})
        assert resolved["Q1"]["answer"] == {"anchor": "goal_success", "basis": "fallback"}
        assert resolved["Q1"]["reason"] == "no end state"

    def test_a_property_not_in_the_context_falls_back(self):
        known = {"G2": set(), "G1": {("washer_1", "countdownTime")}, "P1": set()}
        resolved, _ = resolve_profile(CHECKLIST, {"answers": [
            {"id": "Q1", "answer": {"anchor": "condition", "condition": {"ref": REF}}}]}, known)
        assert resolved["Q1"]["source"] == "fallback"
        assert "washer_1.operationalState is not in the context" in resolved["Q1"]["reason"]

    def test_an_inconsistency_stops_everything(self):
        with pytest.raises(InconsistentStructure):
            resolve_profile(CHECKLIST, {"unresolved": [
                {"id": "Q1", "question": "inconsistency", "target": "G2",
                 "reason": "waits for an event that never happens"}]})

    @pytest.mark.parametrize("output", [
        {"answers": [{"id": "Q9", "answer": {}}]},
        {"answers": [{"id": "Q1", "answer": {}}, {"id": "Q1", "answer": {}}]},
        {"answers": [{"id": "Q1", "answer": {}}], "unresolved": [{"id": "Q1"}]}])
    def test_ids_must_match_the_checklist(self, output):
        with pytest.raises(ProfileOutputError):
            resolve_profile(CHECKLIST, output)

    def test_every_question_on_its_fallback(self):
        resolved, log = fallback_profile(CHECKLIST, "profiling failed")
        assert {r["source"] for r in resolved.values()} == {"fallback"}
        assert len(log) == 2

    @pytest.mark.parametrize("answer, reason", [
        ({"ref_remaining": {"ref": REF, "reads_as": "duration"},
          "target_duration": {"reads_as": "duration"}}, "target_duration"),
        ({"ref_remaining": None, "ref_duration": None,
          "target_duration": {"ref": REF, "reads_as": "duration"}}, "neither"),
        ({"ref_remaining": {"ref": REF}, "target_duration": {"ref": REF, "reads_as": "duration"}},
         "ref_remaining"),
    ])
    def test_a_malformed_answer_falls_back(self, answer, reason):
        """completion_timing has no fallback start: its answer is empty, and
        the planner asks the user instead."""
        checklist = [{"id": "Q1", "question": "completion_timing", "target": "G1", "ref": "P1"}]
        resolved, log = resolve_profile(checklist, {"answers": [{"id": "Q1", "answer": answer}]})
        assert resolved["Q1"]["source"] == "fallback" and reason in resolved["Q1"]["reason"]
        assert resolved["Q1"]["answer"] == {}

