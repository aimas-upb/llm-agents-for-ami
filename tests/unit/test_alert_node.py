"""The `alert` action: the node raises, the UA delivers.

Covered:
  * the IR: an alert needs a non-empty text; it compiles to an AlertNode;
  * the node: one alert per entry, one per pass of a `repeat`, none twice;
    two trees never share alerts;
  * AlertBehaviour: the alert goes to stderr always, then to every handler
    the agent has; a failing handler is logged and the others still run;
  * a running plan delivers what its tree raised, one AlertBehaviour per alert.
"""

import logging

import py_trees
import pytest
from py_trees.common import Status
from unittest.mock import MagicMock

from ami_agents.agents.user_assistant.behaviours.alert import (
    ALERT_LOGGER_NAME,
    AlertBehaviour,
)
from ami_agents.agents.user_assistant.behaviours.plan_execution import (
    AchievementPlanBehaviour,
)
from ami_agents.bt_planning.nodes.alert_node import Alert, AlertNode
from ami_agents.bt_planning.nodes.registry import compile_node, validate_node

from tests.unit.test_plan_execution import make_agent, register


def _compile(spec):
    """An IR spec as a py_trees tree, as the UA compiles it."""
    def child(s):
        return compile_node(s, child)
    return child(spec)


def _alerts(tree):
    """Everything the tree's alert nodes have raised, drained."""
    return [a.text for n in tree.iterate() if isinstance(n, AlertNode) for a in n.drain()]


# -- the IR -------------------------------------------------------------------------

class TestIR:
    def test_an_alert_says_something(self):
        assert validate_node({"type": "alert", "text": "Grid import above 3 kW"}) == []
        assert validate_node({"type": "alert", "text": "  "})
        assert validate_node({"type": "alert"})

    def test_it_compiles_to_an_alert_node(self):
        node = _compile({"type": "alert", "name": "G1: alert", "text": "hello"})
        assert isinstance(node, AlertNode)
        assert (node.name, node.text) == ("G1: alert", "hello")


# -- the node -----------------------------------------------------------------------

class TestNode:
    def test_one_alert_per_entry_and_never_twice(self):
        """Entered once, it raises once; a drained alert is gone."""
        # 1. One entry, one alert, SUCCESS.
        node = _compile({"type": "alert", "text": "hello"})
        node.tick_once()
        assert node.status is Status.SUCCESS
        assert [a.text for a in node.drain()] == ["hello"]
        # 2. Drained means delivered: nothing left.
        assert node.drain() == []

    def test_every_pass_of_a_repeat_raises_a_new_alert(self):
        """"Whenever": each pass of the loop enters the node again."""
        tree = _compile({"type": "repeat", "name": "whenever", "children": [
            {"type": "alert", "text": "again"}]})
        for _ in range(3):
            tree.tick_once()
        assert _alerts(tree) == ["again"] * 3

    def test_two_trees_never_share_alerts(self):
        """Alerts wait on their own node, not on the process-wide blackboard."""
        first = _compile({"type": "alert", "text": "first"})
        second = _compile({"type": "alert", "text": "second"})
        first.tick_once()
        assert _alerts(second) == []
        assert _alerts(first) == ["first"]


# -- the UA behaviour ---------------------------------------------------------------

def _behaviour(agent, text="Grid import above 3 kW"):
    """An AlertBehaviour bound to a fake agent, as add_behaviour would."""
    behaviour = AlertBehaviour("plan-1", "thread-a", Alert(text, MagicMock()))
    behaviour.agent = agent
    return behaviour


class TestBehaviour:
    @pytest.mark.asyncio
    async def test_the_default_action_writes_to_stderr(self, capsys):
        """No handler hooked in: the alert still reaches the console."""
        agent = make_agent()
        agent.alert_handlers = []
        await _behaviour(agent).run()
        err = capsys.readouterr().err
        assert "ALERT plan=plan-1 Grid import above 3 kW" in err

    @pytest.mark.asyncio
    async def test_every_handler_gets_the_alert(self):
        """The hook: sync and async handlers alike."""
        agent = make_agent()
        got = []

        async def collector(behaviour, alert):
            got.append(("async", behaviour.plan_id, alert.text))

        agent.alert_handlers = [collector,
                                lambda behaviour, alert: got.append(("sync", alert.text))]
        await _behaviour(agent).run()
        assert got == [("async", "plan-1", "Grid import above 3 kW"),
                       ("sync", "Grid import above 3 kW")]

    @pytest.mark.asyncio
    async def test_a_failing_handler_is_logged_and_the_others_still_run(self):
        agent = make_agent()
        got = []

        def broken(behaviour, alert):
            raise RuntimeError("push service down")

        agent.alert_handlers = [broken, lambda b, a: got.append(a.text)]
        behaviour = _behaviour(agent)
        behaviour.logger = MagicMock()
        await behaviour.run()
        assert got == ["Grid import above 3 kW"]
        assert "push service down" in behaviour.logger.error.call_args[0][0]

    def test_the_alert_logger_has_one_stderr_handler(self):
        """However many alerts are delivered, each prints once."""
        from ami_agents.agents.user_assistant.behaviours.alert import alert_logger
        alert_logger(); alert_logger()
        handlers = logging.getLogger(ALERT_LOGGER_NAME).handlers
        assert sum(getattr(h, "_ami_alert_stderr", False) for h in handlers) == 1


# -- a running plan delivers --------------------------------------------------------

class TestDelivery:
    @pytest.mark.asyncio
    async def test_a_tick_that_raises_spawns_one_behaviour_per_alert(self):
        """The plan's tick drains its alert nodes into AlertBehaviours."""
        # 1. A real tree: two alerts in one sequence.
        agent = make_agent(); register(agent)
        spec = {"type": "sequence", "name": "two", "children": [
            {"type": "alert", "text": "one"}, {"type": "alert", "text": "two"}]}
        behaviour = AchievementPlanBehaviour("plan-1", spec, max_ticks=5, tick_period=0.0)
        behaviour.agent = agent
        behaviour.receive = MagicMock(return_value=None)
        behaviour.tree = _compile(spec)
        behaviour._setup_done = True
        # 2. One tick: both raised, each handed to its own behaviour.
        behaviour._tick()
        spawned = [c.args[0] for c in agent.add_behaviour.call_args_list]
        assert [(b.text, b.conversation_id) for b in spawned] == [
            ("one", "thread-a"), ("two", "thread-a")]
        # 3. Delivered means drained: the nodes hold nothing to deliver twice.
        assert _alerts(behaviour.tree) == []
