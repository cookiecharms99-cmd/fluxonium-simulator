#!/usr/bin/env bash

# Compare three CMA-ES initial centers under identical trusted reference physics.
# Run from the repository root with:
#   systemd-inhibit --what=sleep --why="Fluxonium reference comparison" \
#     bash scripts/run_reference_initialization_compare.sh

set -euo pipefail

RUN_LABEL="laptop_reference_init_$(date -u +%Y%m%dT%H%M%SZ)"
COMMON_ARGS=(
  --physics-config config/reference_prof.json
  --control-protocol reference
  --full-physics
  --duration 150
  --generations 80
  --popsize 16
  --workers 4
  --seed 1234
)

for INIT_SCHEME in zero random legacy_piecewise_lstsq; do
  uv run main/run_cmaes.py \
    "${COMMON_ARGS[@]}" \
    --initialization "$INIT_SCHEME" \
    --out-dir "data/output/cmaes/${RUN_LABEL}_${INIT_SCHEME}"
done
