"""Read one property's current value while planning, over the TD's read form.

The planners mostly reason over the TD graph alone; a few decisions need the value a
device holds now. The ambiguous planner shows its model the current state, and the
modify planner checks that an ordered value ("fan speed auto") is on its scale before
it plans "up by one level". Both read the same way, here:

    value = await read_property_value("http://.../properties/fanSpeed")
    # "auto" -- or None when the read failed: unknown, never a guess

The HTTP call blocks, so it runs off the event loop. `off_scale_check` is the modify
planner's use of it, shared by the deterministic and the small-model explicit planners.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from ....bt_planning.nodes.http_client import HTTPClient, HTTPClientConfig
from .modify_goal_plan import ModifyRecipe, off_scale_entry


async def read_property_value(url: str) -> Optional[Any]:
    """The body of a GET on ``url`` -- the property's value, as the TD's read
    form serves it -- or None when the request fails or answers an error. A
    failed read is unknown, not wrong, so it never raises."""
    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(
            None, HTTPClient(config=HTTPClientConfig()).get, url)
    except Exception:  # noqa: BLE001 - unknown, not wrong
        return None
    return response.body if getattr(response, "is_success", False) else None


async def off_scale_check(recipe: ModifyRecipe, source: str) -> Optional[Dict[str, Any]]:
    """For a recipe that steps along an ordered scale, the question to ask instead of
    planning when the device's current value is not on the scale (`off_scale_entry`);
    None when the recipe is not ordered, the value is on the scale, or it could not be
    read -- the recipe's tree is then the plan."""
    if not recipe.order:
        return None
    return off_scale_entry(recipe, await read_property_value(recipe.property_url), source)
