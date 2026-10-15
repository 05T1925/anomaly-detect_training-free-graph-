#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$PROJECT/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/.venvs/cola/bin/python}"

"$PYTHON" "$PROJECT/scripts/run_neighbordiv.py" \
  --data "$ROOT/data/Cora/cora-freegad.mat" \
  --rank 8 \
  --random-state 0 \
  --output-dir "$ROOT/results/NeighborDiv/cora-table16-audit-full"
