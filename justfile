# Project wide tasks!

# Show list
default:
    @just --list
# Display project commands
desc:
    batcat commands.md

# Show current data files by recency
latest_data num:
    find data/output/cmaes -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 ls -dt | head -n {{ num }}

# Run test suite!
test:
    uv run pytest -q

# pull data from the cluster :D
pull-cluster-data:
    rsync -avz --progress jseepaul2023@athene-login.hpc.fau.edu:~/fluxonium-simulator/data/output/cmaes ./data/output/cmaes/

# Quick plot on a selected run directory. Example:
# just ez_plot --population-cutoff 13
ez_plot +args="":
    RUN_DIR="$(ls -td data/output/cmaes/*/ 2>/dev/null | fzf)" && \
    uv run python main/plot_run.py "$RUN_DIR" {{ args }}

# quick audit on a run data_directory
ez_cutoff_audit:
    uv run source/audit_controls.py --run "$(find data/output/cmaes -maxdepth 1 -type d | fzf)"
# quick benchmark on a run data directory
ez_benchmark_run:
    uv run main/benchmark_simulator.py --run "$(find data/output/cmaes -maxdepth 1 -type d | fzf)"
