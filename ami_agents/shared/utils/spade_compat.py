"""
Compatibility helpers for SPADE / slixmpp differences.

Older slixmpp releases (≤1.8.1) do not accept the ``address`` keyword
argument on ``ClientXMPP.connect``. SPADE ≥4.1 passes ``address`` as a kwarg,
which raises ``TypeError`` on those older builds. This module patches the
method at import time to accept the kwarg and forward it positionally.
"""

from __future__ import annotations

import inspect
from typing import Any, Tuple


def _patch_slixmpp_connect() -> None:
    try:
        from slixmpp import ClientXMPP
    except Exception:
        return

    signature = inspect.signature(ClientXMPP.connect)
    if "address" in signature.parameters:
        return

    original_connect = ClientXMPP.connect

    def _connect(self: ClientXMPP, *args: Any, **kwargs: Any) -> Any:
        address = kwargs.pop("address", None)
        new_args = args
        if address is not None:
            if not isinstance(address, tuple):
                address = tuple(address)  # type: ignore[arg-type]
            new_args = (address,) + args
        return original_connect(self, *new_args, **kwargs)

    ClientXMPP.connect = _connect  # type: ignore[assignment]


_patch_slixmpp_connect()

