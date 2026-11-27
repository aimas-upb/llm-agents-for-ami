"""No behaviour of ours redefines a method SPADE's behaviours rely on.

SPADE calls its own methods on every behaviour: `_step()` from its run loop,
`match(message)` to route each incoming message. A behaviour that defines a
method of the same name for its own purpose replaces SPADE's silently:
`DependencyGoalPlanningBehaviour._step(graph, gid, goal)` made every dependency
request crash, and `SignifierMatchBehaviour.match(payload)` (async, so always
truthy) made the EnvExplorer queue every message to it, and
`IncompleteGoalPlanningBehaviour._run(behaviour)` replaced the `_run()` SPADE's
run loop calls. This test lists every
behaviour class under `ami_agents/agents` and fails on any such name.
"""

import importlib
import inspect
import pkgutil

import spade.behaviour as sb

import ami_agents.agents as agents

# What a behaviour is meant to define, or what Python adds to every class.
MEANT_TO_OVERRIDE = {"run", "on_start", "on_end", "__init__"}
PYTHON_ADDS = {"__doc__", "__module__", "__qualname__", "__dict__", "__weakref__",
               "__annotations__", "__firstlineno__", "__static_attributes__",
               "__abstractmethods__", "_abc_impl", "__parameters__", "__orig_bases__"}


def _spade_names():
    """Every attribute SPADE's behaviour classes define."""
    names = set()
    for cls in (sb.CyclicBehaviour, sb.OneShotBehaviour, sb.PeriodicBehaviour,
                sb.TimeoutBehaviour, sb.FSMBehaviour, sb.State):
        for klass in cls.__mro__:
            if klass.__module__.startswith("spade"):
                names |= set(vars(klass))
    return names - MEANT_TO_OVERRIDE - PYTHON_ADDS


def _our_behaviours():
    """Every behaviour class defined in an `ami_agents.agents` module."""
    for module in pkgutil.walk_packages(agents.__path__, agents.__name__ + "."):
        loaded = importlib.import_module(module.name)
        for _, cls in inspect.getmembers(loaded, inspect.isclass):
            # Every SPADE behaviour kind derives from CyclicBehaviour.
            if cls.__module__ == loaded.__name__ and issubclass(cls, sb.CyclicBehaviour):
                yield cls


def test_no_behaviour_shadows_spade():
    spade = _spade_names()
    clashes = {f"{cls.__module__}.{cls.__name__}": sorted(set(vars(cls)) & spade)
               for cls in _our_behaviours()}
    assert not {k: v for k, v in clashes.items() if v}, clashes
