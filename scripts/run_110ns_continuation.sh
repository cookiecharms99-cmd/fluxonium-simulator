#!/usr/bin/env bash

# Continue the saved 120-ns fixed-charge solution at 110 ns. This is a
# full-physics duration-continuation experiment, not a smoke run.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PARENT_RUN="data/output/cmaes/continuation_120ns_from_130ns_20261104T091143Z"
PARENT_WEIGHTS="${PARENT_RUN}/best_weights.npy"
RUN_LABEL="continuation_110ns_from_120ns_$(date -u +%Y%m%dT%H%M%SZ)"

if [[ ! -f "$PARENT_WEIGHTS" ]]; then
  echo "Missing warm-start weights: $PARENT_WEIGHTS" >&2
  exit 1
fi

systemd-inhibit --what=sleep --why="Fluxonium 110-ns continuation" \
  uv run python main/run_cmaes.py \
    --physics-config config/reference_prof.json \
    --control-protocol reference \
    --initialization warm_start \
    --warm-start "$PARENT_WEIGHTS" \
    --full-physics \
    --duration 110 \
    --generations 2000 \
    --popsize 16 \
    --workers 4 \
    --seed 1234 \
    --out-dir "data/output/cmaes/${RUN_LABEL}"
