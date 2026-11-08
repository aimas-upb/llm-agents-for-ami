"""event_lexicon: appliance-event predicates completed from their words."""

import pytest
from rdflib import Graph

from ami_agents.agents.interaction_solver.utils.dependency_context import resolve_predicate
from ami_agents.agents.interaction_solver.utils.event_lexicon import complete_event_predicate
from ami_agents.shared.models.goal_structure import PredicateSpec
from ami_agents.shared.utils.vocabulary import close_types
from tests.unit.test_dependency_compiler import HOME, _device, _prop, PREFIXES, _room


@pytest.fixture(scope="module")
def graph():
    # The compiler's home plus a second washer and a dryer in the bathroom.
    extra = "".join([
        _room("bath", "Bathroom", "Bathroom", ["washer_2", "dryer_1"]),
        _device("washer_2", "LaundryWasher"),
        _prop("washer_2", "operationalState", "OperationalStateOperationalState", "js:IntegerSchema"),
        _device("dryer_1", "LaundryDryer"),
        _prop("dryer_1", "operationalState", "OperationalStateOperationalState", "js:IntegerSchema"),
    ])
    g = Graph().parse(data=HOME + extra, format="turtle")
    close_types(g)
    return g


def _pred(text, **fields):
    return PredicateSpec.from_dict({"predicate_subject": "device_property",
                                    "predicate_text": text, "role": "trigger", **fields})


class TestEmptyPredicates:
    def test_the_washer_in_the_utility_room(self, graph):
        p = _pred("washer 1 in the utility room finishes")
        note = complete_event_predicate(graph, p)
        assert (p.artifact_class, p.location_class, p.artifact_name) == \
            ("homeont:LaundryWasher", "homeont:UtilityRoom", "washer_1")
        assert (p.property_name, p.comparison, p.target_value_determined, p.target_value_text) == \
            ("operationalState", "==", "0", "finished")
        assert p.predicate_specificity == "explicit"
        assert "appliance-event lexicon" in note
        target = resolve_predicate(graph, p)
        assert [i.prop for i in target.properties] == ["operationalState"]

    def test_the_index_picks_among_several_washers(self, graph):
        p = _pred("when washer 2 finishes")
        complete_event_predicate(graph, p)
        assert p.artifact_name == "washer_2"

    @pytest.mark.parametrize("text", ["dryer 1 in the bathroom is finished",
                                      "the dryer 1 finished drying",
                                      "dryer 1's cycle ends", "dryer 1 is done"])
    def test_event_wordings(self, graph, text):
        p = _pred(text)
        complete_event_predicate(graph, p)
        assert (p.artifact_name, p.target_value_determined) == ("dryer_1", "0")

    def test_starting_is_running(self, graph):
        p = _pred("dryer 1 starts")
        complete_event_predicate(graph, p)
        assert (p.target_value_determined, p.target_value_text) == ("1", "running")

    def test_a_missing_device_keeps_the_class_for_the_not_found_path(self, graph):
        p = _pred("dishwasher 1 in the kitchen finishes")
        complete_event_predicate(graph, p)
        assert (p.artifact_class, p.artifact_name, p.target_value_determined) == \
            ("homeont:Dishwasher", None, "0")
        assert resolve_predicate(graph, p).startswith("no device has the property")

    def test_an_ambiguous_instance_is_left_unnamed(self, graph):
        p = _pred("the washer finishes")          # washer_1 and washer_2
        complete_event_predicate(graph, p)
        assert p.artifact_name is None and p.artifact_class == "homeont:LaundryWasher"

    def test_not_an_appliance_event(self, graph):
        for text in ("the hallway is lit", "17 minutes after the previous action",
                     "light 1 finishes"):
            p = _pred(text)
            assert complete_event_predicate(graph, p) is None
            assert p.target_value_determined is None

    def test_environment_predicates_are_not_touched(self, graph):
        p = _pred("washer 1 finishes", predicate_subject="environment_property")
        assert complete_event_predicate(graph, p) is None


class TestResolvedPredicates:
    def test_finished_valued_as_running_is_corrected(self, graph):
        p = _pred("washer 1 finishes", artifact_name="washer_1", property_name="operationalState",
                  comparison="==", target_value_text="finished", target_value_determined="1")
        note = complete_event_predicate(graph, p)
        assert p.target_value_determined == "0" and "stopped" in note

    def test_a_consistent_predicate_is_untouched(self, graph):
        p = _pred("washer 1 finishes", artifact_name="washer_1", property_name="operationalState",
                  comparison="==", target_value_text="Stopped", target_value_determined="0")
        assert complete_event_predicate(graph, p) is None

    def test_a_deliberate_running_value_is_kept(self, graph):
        p = _pred("washer 1 finishes", artifact_name="washer_1", property_name="operationalState",
                  comparison="==", target_value_text="Running", target_value_determined="1")
        assert complete_event_predicate(graph, p) is None
