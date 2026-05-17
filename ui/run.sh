#!/usr/bin/env bash
# Convenience launcher — delegates to ui/run.py for the full preflight.
# Usage:
#   bash run.sh                  # from repo root or ui/
#   bash run.sh --port 8600 -- --server.maxUploadSize 50

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
exec python3 ui/run.py "$@"
