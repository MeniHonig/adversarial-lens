#!/usr/bin/env python3
"""
Top-level launcher: a thin shim that forwards to ui/run.py.

So you can do any of:

    python run.py                          # from repo root
    python ui/run.py                       # from anywhere
    python /abs/path/to/adversarial-lens/run.py
    ./run.py                               # if executable bit is set

All flags are forwarded verbatim. See ui/run.py --help for the full surface.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UI_RUN = ROOT / "ui" / "run.py"

if not UI_RUN.exists():
    sys.stderr.write(f"error: cannot find {UI_RUN}\n")
    sys.exit(2)

sys.argv[0] = str(UI_RUN)
runpy.run_path(str(UI_RUN), run_name="__main__")
