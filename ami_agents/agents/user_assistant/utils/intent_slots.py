"""Normalising the class slots a structuring parser returns.

Both the ENV_STATE and the ENV_CAPABILITIES parser answer with ontology class
identifiers, some bare (`location_class`) and some as `{class, ...}` objects
(`device_property`). These coerce the LLM's object into the shape callers rely
on: an identifier or None, never a sentinel string.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# The parser must answer with a class specific enough to identify something.
# These are the property taxonomy's roots: they match every property beneath
# them, so an answer naming one resolves to everything and therefore to nothing.
BRANCH_ROOTS = frozenset({
    "homeont:ActuatableDeviceProperty",
    "homeont:DeviceStateProperty",
    "homeont:DeviceCapabilityProperty",
    "sosa:ObservableProperty",
})


def clean_class(value: Any) -> Optional[str]:
    """A class identifier, or None.

    Older prompts used "NA" as a sentinel; treat any such leftover as absent so
    a caller never has to test for a magic string.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NONE", "NULL", "UNKNOWN"}:
        return None
    return text


def clean_nested(value: Any, *keys: str) -> Optional[Dict[str, str]]:
    """A `{class, ...}` object, or None if it carries no usable class."""
    if not isinstance(value, dict):
        return None
    cls = clean_class(value.get("class"))
    if cls is None:
        return None
    out: Dict[str, str] = {"class": cls}
    for key in keys:
        extra = clean_class(value.get(key))
        if extra is not None:
            out[key] = extra
    return out
