"""The clock plans run on: the system time, or the environment's own.

A plan that waits ("at 1:39 PM", "in 13 minutes", "23 minutes before the washer
finishes") must wait in the time of the environment it acts on. In a real home
that is the system time; in the SimuHome simulator it is the simulator's
virtual time, which runs at its own pace. Which one is configuration, part of
the environment model -- the agents never decide it:

    # environment.yaml > environment.clock
    source: system | http          (default: system)
    http: {url, field, format}     e.g. SHTD's /_shtd/time, which proxies the
                                   simulator's GET /api/time

    make_clock(environment_config) -> Clock      Clock.now() -> naive datetime

Times are naive wall-clock datetimes (the simulator speaks
"YYYY-MM-DD HH:MM:SS", with no zone). For arithmetic a time is encoded as
seconds by `to_seconds` -- the wall clock read as if it were UTC -- so
differences are exact and a time of day can be recovered from the number.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

DEFAULT_FIELD = "now"
DEFAULT_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_TIMEOUT = 5.0


class ClockError(RuntimeError):
    """The clock could not be read."""


class Clock:
    """Where "now" comes from."""

    def now(self) -> datetime:  # pragma: no cover - interface
        raise NotImplementedError


class SystemClock(Clock):
    """The machine's local wall-clock time."""

    def now(self) -> datetime:
        return datetime.now()


class HttpClock(Clock):
    """A clock read over HTTP: `GET url` answers JSON whose `field` is a time in
    `fmt` (the simulator's virtual time, through SHTD)."""

    def __init__(self, url: str, field: str = DEFAULT_FIELD, fmt: str = DEFAULT_FORMAT,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        if not url:
            raise ValueError("an http clock needs a url")
        self.url, self.field, self.fmt, self.timeout = url, field, fmt, timeout

    def now(self) -> datetime:
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8") or "{}")
        except Exception as exc:  # noqa: BLE001 - one error type for callers
            raise ClockError(f"GET {self.url}: {exc}") from exc
        value = body.get(self.field) if isinstance(body, dict) else None
        if value is None and isinstance(body, dict) and isinstance(body.get("data"), dict):
            value = body["data"].get(self.field)     # the simulator's own envelope
        if value is None:
            raise ClockError(f"GET {self.url}: no {self.field!r} in {body!r}")
        try:
            return datetime.strptime(str(value), self.fmt)
        except ValueError as exc:
            raise ClockError(f"GET {self.url}: {value!r} is not {self.fmt!r}") from exc


def make_clock(environment_config: Optional[Dict[str, Any]]) -> Clock:
    """The clock the environment configuration names; the system clock when it
    names none."""
    cfg = ((environment_config or {}).get("clock") or {})
    source = str(cfg.get("source") or "system").lower()
    if source == "http":
        http = cfg.get("http") or {}
        return HttpClock(url=str(http.get("url") or ""),
                         field=str(http.get("field") or DEFAULT_FIELD),
                         fmt=str(http.get("format") or DEFAULT_FORMAT))
    return SystemClock()


# --------------------------------------------------------------------------
# Times as numbers
# --------------------------------------------------------------------------

def to_seconds(moment: datetime) -> float:
    """A wall-clock time as seconds (read as if UTC): what the blackboard holds."""
    if moment.tzinfo is not None:
        moment = moment.replace(tzinfo=None)
    return moment.replace(tzinfo=timezone.utc).timestamp()


def from_seconds(seconds: float) -> datetime:
    """The wall-clock time back from `to_seconds`."""
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


def parse_moment(value: Union[str, int, float, datetime]) -> Optional[float]:
    """A timestamp as a property reports it -> seconds, or None. Numbers are
    taken as seconds already; strings in the clock format or ISO 8601 are parsed."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return to_seconds(value)
    if isinstance(value, str):
        for fmt in (DEFAULT_FORMAT, None):
            try:
                moment = (datetime.strptime(value, fmt) if fmt
                          else datetime.fromisoformat(value))
                return to_seconds(moment)
            except ValueError:
                continue
    return None
