"""Behaviour: bootstrap the Experience Engine at agent startup.

Per Alex's chat feedback: the Experience Engine startup should be an
explicit Behaviour visible in ``EnvExplorerAgent.setup()`` rather than
hidden in a utility function called lazily on first signifier query.

This makes engine readiness a discrete, observable lifecycle event
(loggable, testable, time-bounded) instead of a side-effect of the
first incoming SIGNIFIER_MATCH_REQUEST.

The legacy ``ensure_experience_engine_ready`` helper is kept as a
defensive backstop: behaviours that call it after this OneShot has
already completed will short-circuit on the ``_experience_engine_ready``
flag. If the OneShot ever fails (e.g. missing dependency), the lazy
path can still recover when the dependency comes back.
"""

from spade.behaviour import OneShotBehaviour

from ....shared.utils.demo_log import demo
from ..experience.engine_utils import ensure_experience_engine_ready


class InitializeExperienceEngineBehaviour(OneShotBehaviour):
    """Initialise the Experience Engine (SignifierRegistry, IntentMatcherRegistry,
    ContextGraphBuilder, SHACLValidator) once during agent startup."""

    async def run(self) -> None:
        self.agent.logger.info(demo("Initialising Experience Engine..."))
        ok = await ensure_experience_engine_ready(self.agent)
        if ok:
            self.agent.logger.info(
                demo(
                    f"Experience Engine startup complete "
                    f"(matcher_default={self.agent._experience_engine_default_matcher_version}, storage_dir={self.agent._experience_engine_storage_dir})"
                )
            )
        else:
            self.agent.logger.error(
                "Experience Engine startup FAILED — falling back to lazy init "
                "on first signifier request."
            )
