#!/usr/bin/env bash

# Continue the saved 150-ns fixed-charge solution at 140 ns. This is a
# full-physics duration-continuation experiment, not a smoke run.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PARENT_RUN="data/output/cmaes/reference_6h_legacy_4776599"
PARENT_WEIGHTS="${PARENT_RUN}/best_weights.npy"
RUN_LABEL="continuation_140ns_from_reference_6h_legacy_$(date -u +%Y%m%dT%H%M%SZ)"

if [[ ! -f "$PARENT_WEIGHTS" ]]; then
  echo "Missing warm-start weights: $PARENT_WEIGHTS" >&2
  exit 1
fi

systemd-inhibit --what=sleep --why="Fluxonium 140-ns continuation" \
  uv run python main/run_cmaes.py \
    --physics-config config/reference_prof.json \
    --control-protocol reference \
    --initialization warm_start \
    --warm-start "$PARENT_WEIGHTS" \
    --full-physics \
    --duration 140 \
    --generations 10000 \
    --popsize 16 \
    --workers 4 \
    --seed 1234 \
    --out-dir "data/output/cmaes/${RUN_LABEL}"
