#!/usr/bin/env python3
"""
One-click launcher for the HW1 Adversarial Playground.

Press the Run button in your IDE (or `python ui/run.py`) and this will:
  1. Make sure streamlit / plotly / pillow are installed.
  2. Launch `streamlit run ui/app.py` from the hw1-release/ directory
     (so dataset.npz and trained-models/ resolve correctly).
  3. Open the UI in your browser at http://localhost:8501.

You can pass extra streamlit flags after the script, e.g.:
    python ui/run.py --server.port 8600
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
HW_DIR = UI_DIR.parent
APP = UI_DIR / "app.py"
REQS = UI_DIR / "requirements.txt"

REQUIRED = ["streamlit", "plotly", "PIL"]


def _ensure_deps() -> None:
    missing = []
    for mod in REQUIRED:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if not missing:
        return
    print(f"[run.py] Installing UI dependencies (missing: {missing})...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(REQS)]
    )


def main() -> int:
    _ensure_deps()

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
        *sys.argv[1:],
    ]
    print(f"[run.py] cwd      = {HW_DIR}")
    print(f"[run.py] launching: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(HW_DIR))


if __name__ == "__main__":
    sys.exit(main())
