#!/usr/bin/env python3
"""
Robust one-click launcher for the adversarial-lens Streamlit UI.

Usage (works from anywhere — repo root, ui/, an absolute path, your IDE):

    python ui/run.py                       # auto-pick port, open browser
    python ui/run.py --port 8600           # force port 8600
    python ui/run.py --no-browser          # don't open browser
    python ui/run.py --check-only          # verify env, don't launch
    python ui/run.py -- --server.maxUploadSize 50   # passthrough streamlit flags

What it does:
    1. Detects the repo root (regardless of cwd) and resolves `ui/app.py`.
    2. Verifies the trained checkpoints + dataset exist (clear error if not).
    3. Imports every UI module to catch syntax / import errors *before*
       Streamlit swallows them.
    4. Checks installed dependency versions, prompts to install missing or
       too-old ones (auto-installs in non-interactive runs).
    5. Picks a free TCP port if the default is busy.
    6. Sets `PYTHONPATH` so `import attacks / utils / models / pipeline`
       resolve consistently across platforms.
    7. Launches Streamlit, prints the URL, and tears down cleanly on Ctrl-C.

Exit codes:
    0  Streamlit exited normally
    1  Generic launcher error
    2  Missing repo files (data / checkpoints)
    3  Dependency install failed or user declined
    4  Imports failed (bug in the UI modules)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import signal
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

# --------- repo discovery (works no matter the cwd) ------------------------

UI_DIR = Path(__file__).resolve().parent
REPO = UI_DIR.parent
APP = UI_DIR / "app.py"
REQS_REPO = REPO / "requirements.txt"
REQS_UI = UI_DIR / "requirements.txt"


# --------- coloured print --------------------------------------------------

_ANSI = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str) -> str:
    return code if _ANSI else ""


GREY = _c("\033[90m")
GREEN = _c("\033[32m")
YELLOW = _c("\033[33m")
RED = _c("\033[31m")
BOLD = _c("\033[1m")
RESET = _c("\033[0m")


def info(msg: str) -> None:
    print(f"{GREY}[run]{RESET} {msg}")


def ok(msg: str) -> None:
    print(f"{GREEN}[run]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[run]{RESET} {msg}")


def err(msg: str) -> None:
    print(f"{RED}[run]{RESET} {msg}", file=sys.stderr)


# --------- environment / file checks --------------------------------------

REQUIRED_FILES = [
    REPO / "dataset.npz",
    REPO / "trained-models" / "simple-cnn-0",
    REPO / "trained-models" / "simple-cnn-1",
    REPO / "trained-models" / "simple-cnn-2",
    UI_DIR / "app.py",
    UI_DIR / "pipeline.py",
    UI_DIR / "embeddings.py",
    UI_DIR / "embedding_reducers.py",
    UI_DIR / "embedding_viz.py",
]


def check_repo_files() -> bool:
    missing = [p for p in REQUIRED_FILES if not p.exists()]
    if not missing:
        ok(f"repo files: all present under {REPO}")
        return True
    err("missing required files:")
    for p in missing:
        err(f"  - {p}")
    err("are you sure you're inside the adversarial-lens repo?")
    return False


# --------- dependencies ---------------------------------------------------
# (module-to-import, distribution-name-for-pip, minimum-version-or-None)
CORE_DEPS = [
    ("numpy", "numpy", "1.24"),
    ("torch", "torch", "2.0"),
    ("torchvision", "torchvision", "0.15"),
    ("streamlit", "streamlit", "1.32"),
    ("plotly", "plotly", "5.18"),
    ("PIL", "pillow", "9.0"),
    ("sklearn", "scikit-learn", "1.3"),
]
OPTIONAL_DEPS = [
    ("umap", "umap-learn", "0.5"),
]


def _version_ok(installed: str, minimum: str) -> bool:
    """Lazy semver-ish comparison: split on dots, compare numeric prefixes."""
    def parse(v):
        out = []
        for part in v.split(".")[:3]:
            num = ""
            for ch in part:
                if ch.isdigit():
                    num += ch
                else:
                    break
            out.append(int(num) if num else 0)
        while len(out) < 3:
            out.append(0)
        return tuple(out)
    return parse(installed) >= parse(minimum)


def check_dependencies(install: bool = True,
                       include_optional: bool = False) -> bool:
    """Verify (and optionally install) the python deps."""
    missing_core: list[str] = []
    outdated_core: list[tuple[str, str, str]] = []
    for mod, pkg, min_v in CORE_DEPS:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing_core.append(pkg)
            continue
        if min_v is None:
            continue
        try:
            ver = md.version(pkg)
        except md.PackageNotFoundError:
            continue
        if not _version_ok(ver, min_v):
            outdated_core.append((pkg, ver, min_v))

    if outdated_core:
        warn("outdated dependencies:")
        for pkg, ver, minv in outdated_core:
            warn(f"  - {pkg} {ver} < {minv}")

    if include_optional:
        for mod, pkg, _ in OPTIONAL_DEPS:
            try:
                importlib.import_module(mod)
            except ImportError:
                warn(f"optional dep {pkg!r} not installed (reducer "
                     "needing it will be greyed-out in the UI)")

    if not missing_core and not outdated_core:
        ok("dependencies: ok")
        return True

    if not install:
        err("missing/outdated deps but --no-install was passed")
        return False

    targets = (missing_core
               + [pkg for pkg, _, _ in outdated_core])
    info(f"installing/upgrading: {', '.join(targets)}")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *targets]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        err(f"pip install failed (exit {e.returncode}). "
            f"Try: pip install -r {REQS_REPO.relative_to(Path.cwd())}")
        return False
    ok("dependencies installed")
    return True


# --------- import smoke test ---------------------------------------------

def check_imports() -> bool:
    """Import every UI module so syntax errors surface here, not later."""
    sys.path.insert(0, str(UI_DIR))
    sys.path.insert(0, str(REPO))
    for mod_name in ("attacks", "utils", "models", "pipeline",
                     "viz", "embeddings", "embedding_reducers",
                     "embedding_viz"):
        try:
            importlib.import_module(mod_name)
        except Exception as e:  # noqa: BLE001
            err(f"failed to import {mod_name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    ok("module imports: ok")
    return True


# --------- port handling -------------------------------------------------

def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(start: int = 8501, max_tries: int = 30) -> int | None:
    for p in range(start, start + max_tries):
        if port_free(p):
            return p
    return None


# --------- launch --------------------------------------------------------

def launch_streamlit(port: int, open_browser: bool,
                     passthrough: list[str]) -> int:
    env = os.environ.copy()
    py_path = [str(UI_DIR), str(REPO), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(p for p in py_path if p)
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "true")

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.port", str(port),
        "--server.headless", "false" if open_browser else "true",
        "--browser.gatherUsageStats", "false",
        *passthrough,
    ]
    info(f"cwd       = {REPO}")
    info(f"app       = {APP.relative_to(REPO)}")
    info(f"command   = {' '.join(cmd)}")
    ok(f"open      → http://localhost:{port}")

    try:
        return subprocess.call(cmd, cwd=str(REPO), env=env)
    except KeyboardInterrupt:
        info("interrupted by user; shutting down")
        return 0


# --------- CLI -----------------------------------------------------------

def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        prog="run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Robust launcher for the adversarial-lens Streamlit UI.

            All unknown args after `--` are forwarded to streamlit, e.g.
                python ui/run.py -- --server.maxUploadSize 50
        """),
    )
    p.add_argument("--port", type=int, default=8501,
                   help="preferred port (auto-bump if busy)")
    p.add_argument("--no-browser", action="store_true",
                   help="don't open the system browser")
    p.add_argument("--no-install", action="store_true",
                   help="don't auto-install missing dependencies")
    p.add_argument("--include-optional", action="store_true",
                   help="also check optional deps (umap-learn, ...)")
    p.add_argument("--check-only", action="store_true",
                   help="run preflight checks and exit (no launch)")
    if "--" in argv:
        i = argv.index("--")
        ours, passthrough = argv[:i], argv[i + 1:]
    else:
        ours, passthrough = argv, []
    return p.parse_args(ours), passthrough


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(sys.argv[1:] if argv is None else argv)

    print(f"{BOLD}adversarial-lens launcher{RESET}  "
          f"{GREY}(python {sys.version.split()[0]}, "
          f"{sys.platform}){RESET}")

    if not check_repo_files():
        return 2
    if not check_dependencies(install=not args.no_install,
                              include_optional=args.include_optional):
        return 3
    if not check_imports():
        return 4
    if args.check_only:
        ok("preflight passed — exiting (--check-only)")
        return 0

    port = args.port
    if not port_free(port):
        new = find_free_port(port + 1)
        if new is None:
            err(f"no free port in {port}..{port + 30}")
            return 1
        warn(f"port {port} busy → using {new}")
        port = new

    # Graceful Ctrl-C handling on Unix.
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal.default_int_handler)

    rc = launch_streamlit(port, open_browser=not args.no_browser,
                          passthrough=passthrough)
    if rc != 0:
        warn(f"streamlit exited with code {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
