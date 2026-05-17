#!/usr/bin/env bash
# Convenience launcher for the HW1 UI.
# Usage:  bash ui/run.sh        (run from hw1-release/)
#     or  bash run.sh           (run from hw1-release/ui/)

set -e

# Resolve repo root (hw1-release/) regardless of CWD.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

cd "$ROOT"

if ! python -c "import streamlit" 2>/dev/null; then
  echo "Installing UI dependencies (streamlit, plotly, pillow)..."
  python -m pip install -r ui/requirements.txt
fi

exec python -m streamlit run ui/app.py "$@"
