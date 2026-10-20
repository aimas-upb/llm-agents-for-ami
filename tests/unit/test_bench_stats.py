"""The harness statistics (`tests/simuhome/bench/stats.py`) on synthetic
records: what each episode row holds, how groups are formed and averaged, and
the files written. No LLM, no simulator.

The records are cut down to the fields the statistics read: the meter's
stages, the plan entries, the executed plans, the structures, and the
evaluation's score, breakdown, execution check and judge.
"""

import json

from tests.simuhome.bench.stats import (
    PHASES,
    aggregate,
    compute,
    episode_metrics,
    load,
)


# --------------------------------------------------------------------------
# Synthetic records
# --------------------------------------------------------------------------

def _stages(seconds=1.0, tokens=100):
    """Meter stages with the same cost in every phase (one call each)."""
    return {phase: {"seconds": seconds, "calls": 1, "llm_seconds": seconds,
                    "tokens_in": tokens, "tokens_out": tokens // 10} for phase in PHASES}


def _record(name, plans=(), executed=(), structures=None, stages=True, outcome="executed"):
    """A record: its episode name gives its type and case."""
    qt, case = name.split("_")[:2]
    return {"episode": name, "query_type": qt, "case": case, "outcome": outcome,
            "error_step": None, "wall_seconds": 9.0,
            "structures": structures if structures is not None else [{} for _ in plans],
            "plans": [{"outcome": o} for o in plans],
            "execution": {"plans": [{"status": s} for s in executed]},
            "meter": {"stages": _stages()} if stages else {}}


def _feasible_eval(score, goal_checks):
    """A feasible evaluation: ``goal_checks`` lists, per eval goal, whether
    each of its checks passed."""
    checks = [{"goal": g, "class": "ok" if ok else "wrong_value"}
              for g, oks in enumerate(goal_checks) for ok in oks]
    passed = sum(c["class"] == "ok" for c in checks)
    return {"score": score, "breakdown": {"checks": checks, "total": len(checks),
                                          "ok": passed, "success": passed / len(checks)}}


def _infeasible_eval(check, judge):
    return {"score": int(check == 1 and judge == 1), "execution_check": {"score": check},
            "judge": {"score": judge}}


# Two goals, the second with one of its two checks wrong: SimuHome fails it.
MULTI_ONE_WRONG = (_record("qt3_feasible_seed_1", ["plan", "plan"], ["SUCCESS", "SUCCESS"]),
                   _feasible_eval(0, [[True], [True, False]]))
SINGLE_OK = (_record("qt3_feasible_seed_2", ["plan"], ["SUCCESS"]),
             _feasible_eval(1, [[True, True]]))
REFUSED = (_record("qt3_infeasible_seed_1", ["impossible", "plan"], ["SUCCESS"],
                   outcome="impossible"),
           _infeasible_eval(1, 1))
CLARIFIED = (_record("qt4-1_infeasible_seed_1", [], [], structures=[{"clarification": "when?"}],
                     outcome="clarify"),
             _infeasible_eval(1, 0))
INFRA = (_record("qt4-1_feasible_seed_1", ["plan"], ["FAILURE"], outcome="error"),
         {"score": -1, "error_type": "Infrastructure"})


def _rows(*pairs):
    return [episode_metrics(record, evaluation) for record, evaluation in pairs]


def _group(summary, qt, case):
    (row,) = [r for r in summary if (r["type"], r["case"]) == (qt, case)]
    return row


# --------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------

class TestEpisodeMetrics:
    def test_cost_per_phase_and_total(self):
        (row,) = _rows(SINGLE_OK)
        assert row["segmentation_s"] == 1.0 and row["guarding_tok_in"] == 100
        assert row["total_s"] == len(PHASES) and row["total_calls"] == len(PHASES)
        assert row["total_tok_out"] == 10 * len(PHASES) and row["wall_s"] == 9.0

    def test_a_record_without_stages_has_no_cost(self):
        record, evaluation = SINGLE_OK
        (row,) = _rows((dict(record, meter={}), evaluation))
        assert row["planning_s"] is None and row["total_s"] is None

    def test_a_goal_is_verified_only_when_all_its_checks_pass(self):
        (row,) = _rows(MULTI_ONE_WRONG)
        assert (row["goals_total"], row["goals_verified"], row["multi_goal"]) == (2, 1, True)
        assert row["goals_verified_frac"] == 0.5 and row["check_rate"] == 2 / 3
        # Our executor ran both plans to SUCCESS: the two views disagree here.
        assert (row["plans_run"], row["plans_success"]) == (2, 2)

    def test_refused_goals_count_impossible_entries_and_clarifications(self):
        refused, clarified = _rows(REFUSED, CLARIFIED)
        assert refused["goals_refused"] == 1 and refused["goals_total"] is None
        assert clarified["goals_refused"] == 1 and clarified["plan_entries"] == 0
        assert (clarified["execution_check"], clarified["judge"]) == (1, 0)

    def test_a_judge_failure_is_no_verdict(self):
        record, _ = REFUSED
        (row,) = _rows((record, _infeasible_eval(1, -1)))
        assert row["judge"] is None


# --------------------------------------------------------------------------
# Groups
# --------------------------------------------------------------------------

class TestAggregate:
    def test_groups_by_type_and_case_then_type_case_and_overall(self):
        summary = aggregate(_rows(MULTI_ONE_WRONG, SINGLE_OK, REFUSED, CLARIFIED, INFRA))
        assert [(r["type"], r["case"]) for r in summary] == [
            ("qt3", "feasible"), ("qt3", "infeasible"), ("qt4-1", "feasible"),
            ("qt4-1", "infeasible"), ("qt3", "all"), ("qt4-1", "all"),
            ("all", "feasible"), ("all", "infeasible"), ("all", "all")]

    def test_infrastructure_failures_are_left_out_of_rates_and_costs(self):
        summary = aggregate(_rows(MULTI_ONE_WRONG, SINGLE_OK, REFUSED, CLARIFIED, INFRA))
        overall = _group(summary, "all", "all")
        assert (overall["n"], overall["n_scored"], overall["n_infra"]) == (5, 4, 1)
        assert overall["n_error"] == 1 and overall["n_costed"] == 4
        assert overall["success_rate"] == 0.5                  # SINGLE_OK and REFUSED
        # The infra episode's FAILURE plan is not counted either.
        assert overall["plans_success_rate"] == 1.0
        infra_only = _group(summary, "qt4-1", "feasible")
        assert infra_only["success_rate"] is None and infra_only["total_s"] is None

    def test_goals_are_pooled_and_reported_for_multi_goal_requests(self):
        feasible = _group(aggregate(_rows(MULTI_ONE_WRONG, SINGLE_OK)), "qt3", "feasible")
        assert (feasible["goals_total"], feasible["goals_verified"]) == (3, 2)
        assert feasible["goals_verified_rate"] == round(2 / 3, 3)
        assert feasible["goals_verified_mean"] == 1.0 and feasible["goals_per_request"] == 1.5
        assert feasible["multi_n"] == 1 and feasible["multi_goals_verified_rate"] == 0.5

    def test_infeasible_rows_report_the_execution_check_and_the_judge(self):
        summary = aggregate(_rows(SINGLE_OK, REFUSED, CLARIFIED))
        infeasible = _group(summary, "all", "infeasible")
        assert (infeasible["execution_check_rate"], infeasible["judge_rate"]) == (1.0, 0.5)
        assert _group(summary, "qt3", "feasible")["execution_check_rate"] is None

    def test_cost_means_and_spread(self):
        overall = _group(aggregate(_rows(MULTI_ONE_WRONG, SINGLE_OK)), "all", "all")
        assert overall["execution_s"] == 1.0 and overall["total_calls"] == len(PHASES)
        assert overall["total_s_median"] == overall["total_s_p95"] == len(PHASES)


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

class TestFiles:
    def test_compute_reads_a_records_directory_and_writes_the_stats(self, tmp_path):
        # 1. Two records, one scored, plus files that are not records.
        for record, evaluation in (SINGLE_OK, REFUSED):
            (tmp_path / f"{record['episode']}.json").write_text(json.dumps(record))
        (tmp_path / "qt3_feasible_seed_2.eval.json").write_text(json.dumps(SINGLE_OK[1]))
        (tmp_path / "run.json").write_text(json.dumps({"variant": {}}))
        # 2. Loaded as two pairs, the unscored one without an evaluation.
        pairs = load(tmp_path)
        assert [(r["episode"], e is not None) for r, e in pairs] == [
            ("qt3_feasible_seed_2", True), ("qt3_infeasible_seed_1", False)]
        # 3. The four files, the report with its legend and three tables.
        out = compute(tmp_path, "test")
        assert sorted(p.name for p in out.iterdir()) == [
            "episodes.csv", "summary.csv", "summary.json", "summary.md"]
        report = (out / "summary.md").read_text()
        assert report.startswith("# test") and "| goals ok |" in report
        assert "## Seconds per phase" in report and "## Tokens per phase" in report
        assert json.loads((out / "summary.json").read_text())[-1]["type"] == "all"
