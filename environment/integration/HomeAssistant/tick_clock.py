#!/usr/bin/env python3
"""
Continuously update the time on the "clock_308" artifact every second
using the existing set-property.py script.

Usage:
  python tick_clock.py            # updates clock_308 every second
  python tick_clock.py <name>     # updates the given artifact name instead

Requires env: HA_URL, HA_TOKEN (used by set-property.py).
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path


def iso_utc_now() -> str:
    s = dt.datetime.now(dt.timezone.utc).isoformat()
    # Convert "+00:00" to "Z" for nicer RFC3339
    return s.replace("+00:00", "Z")


def main():
    artifact = sys.argv[1] if len(sys.argv) > 1 else "clock_308"

    # Ensure the helper exists; prefer local file in the same directory
    here = Path(__file__).resolve().parent
    setter = here / "set-property.py"
    if not setter.exists():
        print("set-property.py not found next to this script.")
        print("Run from repo root or pass a correct path.")
        sys.exit(1)

    # Check required env passed through to child
    for var in ("HA_URL", "HA_TOKEN"):
        if not os.getenv(var):
            print(f"Missing environment variable: {var}")
            sys.exit(2)

    print(f"Ticking {artifact} every second. Ctrl+C to stop.")
    try:
        while True:
            now = iso_utc_now()
            # Invoke the setter: set state to the current timestamp
            # Use sys.executable for consistent interpreter
            cmd = [sys.executable, str(setter), artifact, "state", now]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    print("set-property failed:", res.stderr.strip() or res.stdout.strip())
                else:
                    # Optional: brief confirmation
                    print(f"{artifact} ← {now}")
            except Exception as e:
                print("Error running set-property:", e)
            # Sleep roughly 1 second
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

