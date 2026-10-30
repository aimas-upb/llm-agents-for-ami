#!/usr/bin/env python3
"""Run the HASP reasoning harness with timestamped output.

From the repository root: python tests/hasp/hasp_lab308e_harness.py --dry-run.
The module form, python -m tests.hasp.hasp_lab308e_harness, works as well.
"""
from pathlib import Path
import sys

# Direct script execution puts tests/hasp on sys.path; add the repository root
# so package imports work just as they do with python -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.hasp.bench.harness import main

if __name__ == "__main__":
    raise SystemExit(main())
