"""The execution-order walk shared by both guards (`utils/plan_order.py`).

The trees here use two made-up leaves, so the walk's rules are tested on
their own, with no TD graph:

    {"type": "set", "key": k, "value": v}   writes k = v
    {"type": "probe", "id": i}              records the facts it starts with

    values       same_value / satisfies: how a fact decides a check
    sequence     carries facts from child to child
    selector, parallel, decorators
                 each child from the facts before; afterwards nothing a child
                 writes is certain
    repeat       its body starts without the keys it writes
    waits        a timer passes the facts through (the plan is trusted)
    rewriting    a leaf may replace its node; the input tree is not modified
"""

from ami_agents.agents.interaction_solver.utils.plan_order import (
    same_value,
    satisfies,
    walk_in_order,
)


def _set(key, value):
    return {"type": "set", "key": key, "value": value}


def _probe(probe_id):
    return {"type": "probe", "id": probe_id}


def _node(kind, *children, **fields):
    return {"type": kind, "name": kind, "children": list(children), **fields}


def _walk(tree, facts=None):
    """(the facts each probe saw, the facts after the tree)."""
    seen = {}

    def leaf(node, current, path):
        """`set` writes its key; `probe` records; anything else: not mine."""
        if node.get("type") == "set":
            return node, {**current, node["key"]: ("==", node["value"])}
        if node.get("type") == "probe":
            seen[node["id"]] = dict(current)
            return node, current
        return None

    def writes(node):
        return {node["key"]} if node.get("type") == "set" else None

    _, after = walk_in_order(tree, leaf, writes, facts)
    return seen, after


class TestValues:
    def test_same_value_by_number_boolean_and_text(self):
        assert same_value(1, 1.0) and same_value("2", 2)
        assert same_value(True, "true") and not same_value(True, 1)
        assert same_value("Paused", "paused") and not same_value("Paused", "Running")

    def test_a_known_value_decides_every_comparison(self):
        assert satisfies(("==", 2), "!=", 1)
        assert satisfies(("==", 2), "==", 2) and not satisfies(("==", 2), "==", 1)
        assert satisfies(("==", 22), ">", 20) and not satisfies(("==", 22), "<", 20)

    def test_a_known_not_only_decides_the_same_not(self):
        """"!= 1" says nothing about which value is there."""
        assert satisfies(("!=", 1), "!=", 1)
        assert not satisfies(("!=", 1), "==", 2)
        assert not satisfies(None, "!=", 1)


class TestOrder:
    def test_a_sequence_carries_facts_from_child_to_child(self):
        seen, after = _walk(_node("sequence", _probe("a"), _set("x", 2), _probe("b")))
        assert seen == {"a": {}, "b": {"x": ("==", 2)}}
        assert after == {"x": ("==", 2)}

    def test_a_later_write_replaces_an_earlier_one(self):
        """Pause, resume, then the step: the fact is the resume's."""
        seen, _ = _walk(_node("sequence", _set("x", 2), _set("x", 1), _probe("p")))
        assert seen["p"] == {"x": ("==", 1)}

    def test_a_selector_branch_does_not_count_after_it(self):
        """A branch may not have run: x is unknown after the selector -- and a
        fact x held before is no longer certain either."""
        seen, after = _walk(_node("sequence", _set("x", 0), _node(
            "selector", _set("x", 2), _probe("second")), _probe("after")))
        assert seen["second"] == {"x": ("==", 0)}     # each branch from before
        assert seen["after"] == {}

    def test_a_parallel_child_does_not_run_before_its_sibling(self):
        seen, after = _walk(_node("parallel", _set("x", 2), _probe("sibling")))
        assert seen["sibling"] == {} and after == {}

    def test_a_decorated_child_may_have_stopped_partway(self):
        _, after = _walk(_node("ignore_failure", _node("sequence", _set("x", 2))))
        assert after == {}

    def test_a_fact_survives_a_timer(self):
        """The plan is trusted while it waits."""
        seen, _ = _walk(_node("sequence", _set("x", 2), {"type": "timer", "seconds": 600},
                              _probe("p")))
        assert seen["p"] == {"x": ("==", 2)}

    def test_a_repeat_body_starts_without_what_it_writes(self):
        """From the second pass on, x holds the previous pass's value."""
        seen, _ = _walk(_node("sequence", _set("x", 0), _set("y", 5), _node(
            "repeat", _node("sequence", _probe("body"), _set("x", 1)))))
        assert seen["body"] == {"y": ("==", 5)}

    def test_starting_facts_are_used(self):
        seen, _ = _walk(_probe("p"), facts={"x": ("!=", 1)})
        assert seen["p"] == {"x": ("!=", 1)}


class TestRewriting:
    def test_a_leaf_rewrites_its_node_and_the_input_is_kept(self):
        tree = _node("sequence", _set("x", 1))

        def leaf(node, facts, path):
            """Every `set` replaced by a probe, with its path."""
            if node.get("type") == "set":
                return {"type": "probe", "path": path}, facts
            return None

        out, _ = walk_in_order(tree, leaf, lambda node: None)
        assert out["children"] == [{"type": "probe", "path": (0,)}]
        assert tree["children"] == [_set("x", 1)]
