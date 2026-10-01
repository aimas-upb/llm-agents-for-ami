"""Selecting a named subset of benchmark episodes.

A prompt change is judged on the episodes it was meant to fix, so the eval has
to be able to replay exactly those. Without this the smallest unit of
measurement is a whole family, and a change that fixes 60 episodes while
breaking 30 looks like a small aggregate gain.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO.parent / "SimuHome" / "data" / "benchmark"
FAMILIES = ["qt1", "qt2", "qt3", "qt4-1", "qt4-2", "qt4-3"]
CASES = ["feasible", "infeasible"]

pytestmark = pytest.mark.skipif(
    not BENCHMARK.is_dir(), reason="SimuHome benchmark corpus not available")


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location(
        "eval_atomic_segmentation",
        REPO / "tests" / "simuhome" / "eval_atomic_segmentation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def names(episodes):
    return sorted(e["path"].name for e in episodes)


class TestReadEpisodeList:
    def test_comments_and_blank_lines_are_ignored(self, ev, tmp_path):
        listing = tmp_path / "eps.txt"
        listing.write_text("# a comment\nqt1_feasible_seed_1.json\n\n"
                           "qt2_feasible_seed_3.json  # trailing\n")
        assert ev.read_episode_list(listing) == [
            "qt1_feasible_seed_1.json", "qt2_feasible_seed_3.json"]

    def test_a_path_is_reduced_to_its_filename(self, ev, tmp_path):
        """Lists are often pasted from a results directory."""
        listing = tmp_path / "eps.txt"
        listing.write_text("some/dir/qt1_feasible_seed_1.json\n")
        assert ev.read_episode_list(listing) == ["qt1_feasible_seed_1.json"]

    def test_a_missing_file_stops_the_run(self, ev, tmp_path):
        with pytest.raises(SystemExit):
            ev.read_episode_list(tmp_path / "nope.txt")

    def test_an_empty_list_stops_the_run(self, ev, tmp_path):
        """Silently running zero episodes would read as a result."""
        listing = tmp_path / "eps.txt"
        listing.write_text("# nothing but comments\n")
        with pytest.raises(SystemExit):
            ev.read_episode_list(listing)


class TestLoadEpisodes:
    def test_only_the_named_episodes_load(self, ev):
        wanted = ["qt1_feasible_seed_1.json", "qt4-2_feasible_seed_1.json",
                  "qt4-1_infeasible_seed_38.json"]
        assert names(ev.load_episodes(BENCHMARK, FAMILIES, CASES, None,
                                      wanted)) == sorted(wanted)

    def test_it_composes_with_families(self, ev, capsys):
        wanted = ["qt1_feasible_seed_1.json", "qt4-2_feasible_seed_1.json"]
        loaded = ev.load_episodes(BENCHMARK, ["qt1"], CASES, None, wanted)
        assert names(loaded) == ["qt1_feasible_seed_1.json"]
        assert "not loaded" in capsys.readouterr().out

    def test_an_unknown_name_is_reported_not_swallowed(self, ev, capsys):
        loaded = ev.load_episodes(
            BENCHMARK, FAMILIES, CASES, None,
            ["qt1_feasible_seed_1.json", "nosuch_episode.json"])
        assert names(loaded) == ["qt1_feasible_seed_1.json"]
        assert "nosuch_episode.json" in capsys.readouterr().out

    def test_no_list_means_the_whole_corpus(self, ev):
        assert len(ev.load_episodes(BENCHMARK, FAMILIES, CASES, None)) == 600
