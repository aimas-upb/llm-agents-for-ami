"""Where an artifact's state is read from, given a snapshot entry.

EnvExplorer's snapshot carries `state_property_urls` per artifact, taken from
its TD: HASP names one state property per entity (`lightState`,
`climateState`), so the URL cannot be built from the artifact alone. An entry
without them falls back to `{artifact}/properties/state`, the name HASP still
serves as a deprecated alias.
"""

from __future__ import annotations

from typing import Any, Dict, List


def state_urls(artifact_id: str, artifact_info: Dict[str, Any]) -> List[str]:
    """The artifact's state property URLs, or the legacy one."""
    urls = artifact_info.get("state_property_urls") if isinstance(artifact_info, dict) else None
    if urls:
        return [str(u) for u in urls]
    artifact_url = str(artifact_id or "")
    if artifact_url.endswith("#artifact"):
        artifact_url = artifact_url[: -len("#artifact")]
    return [f"{artifact_url}/properties/state"] if artifact_url else []
