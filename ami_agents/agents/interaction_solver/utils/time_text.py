"""The structurer's time words, as numbers the dependency templates can use.

The structurer copies time words verbatim (`timing.offset_text` "13 minutes",
`timing.time_text` "1:39 PM", `ends.text` "for the next 5 hours"). The compiler
needs a span in seconds or a time of day; it never needs a date or the current
time -- a time of day is resolved against the plan's clock when the plan runs
(`bt_planning/nodes/dependency_nodes.WaitUntilTime`).

    span_seconds("1 hour and 30 minutes")   -> 5400.0
    time_of_day("1:39 PM")                  -> "13:39"

Both raise TimeTextError on words they cannot read; the compiler then reports
the words instead of guessing.
"""

from __future__ import annotations

import re

_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}
_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
          "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
          "half an": 0.5, "half a": 0.5}
# A number may touch its unit ("90s"); a number word must be followed by a space,
# or "and" would read as "an d(ay)".
_SPAN = re.compile(r"\b(?:(\d+(?:\.\d+)?)\s*|(half an?|an?|one|two|three|four|five|six|seven"
                   r"|eight|nine|ten|fifteen|twenty|thirty|forty|fifty|sixty)\s+)([a-z]+)\b")
_CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?![\d:])")


class TimeTextError(ValueError):
    """Time words that cannot be read."""


def span_seconds(text: str) -> float:
    """A span in seconds: the sum of every "<amount> <unit>" in the words
    ("1 hour and 30 minutes", "half an hour", "90 s")."""
    words = str(text or "").lower()
    total, found = 0.0, False
    for digits, word, unit in _SPAN.findall(words):
        if unit not in _UNITS:
            continue
        number = float(digits) if digits else float(_WORDS[word])
        total += number * _UNITS[unit]
        found = True
    if not found:
        raise TimeTextError(f"no span of time in {text!r}")
    return total


def time_of_day(text: str) -> str:
    """A clock time "HH:MM[:SS]" (24 h) from "1:39 PM", "13:39", "7 am", "noon"."""
    words = str(text or "").lower().strip()
    if "noon" in words or "midday" in words:
        return "12:00"
    if "midnight" in words:
        return "00:00"
    for hour, minute, second, half in _CLOCK.findall(words):
        if not minute and not half:
            continue                  # a bare number is not a time ("in 5")
        h, m = int(hour), int(minute or 0)
        half = half.replace(".", "")
        if half == "pm" and h < 12:
            h += 12
        elif half == "am" and h == 12:
            h = 0
        if h > 23 or m > 59:
            continue
        return f"{h:02d}:{m:02d}" + (f":{int(second):02d}" if second else "")
    raise TimeTextError(f"no time of day in {text!r}")
