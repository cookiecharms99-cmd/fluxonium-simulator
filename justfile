# Friendly project tasks. Run `just manual` for examples and advanced flags.

# Show the compact recipe list.
default:
    @just --list

# Display the complete manual. batcat is optional; plain cat works everywhere.
manual:
    @if command -v batcat >/dev/null 2>&1; then batcat commands.md; else cat commands.md; fi

# Historical friendly name for the manual.
desc:
    @just manual

# Show current data files by recency
latest_data num:
    find data/output/cmaes -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 ls -dt | head -n {{ num }}

# Run test suite!
test:
    uv run pytest -q

# Direct, scriptable entry points. Append any underlying script flags after
# the declared recipe arguments; see `just manual` for examples.
run +args="":
    @set -- {{ args }}; if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then shift; fi; uv run python main/run_cmaes.py "$@"

plot run +args="":
    @set -- {{ args }}; if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then shift; fi; uv run python main/plot_run.py "{{ run }}" "$@"

audit run +args="":
    @set -- {{ args }}; if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then shift; fi; uv run python source/audit_controls.py --run "{{ run }}" "$@"

benchmark run +args="":
    @set -- {{ args }}; if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then shift; fi; uv run python main/benchmark_simulator.py --run "{{ run }}" "$@"

analyze +runs="":
    uv run python main/analyze_run.py {{ runs }}

list-runs:
    find data/output/cmaes -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort

# pull data from the cluster :D
pull-cluster-data:
    rsync -avz --progress jseepaul2023@athene-login.hpc.fau.edu:~/fluxonium-simulator/data/output/cmaes/ ./data/output/cmaes/

# Interactive helpers for local exploration. They require fzf.
# Example: just ez_plot --population-cutoff 13
ez_plot +args="":
    RUN_DIR="$(ls -1td --time=birth data/output/cmaes/*/ 2>/dev/null | fzf)" && \
    uv run python main/plot_run.py "$RUN_DIR" {{ args }}

# Quick audit on a selected run directory.
ez_cutoff_audit:
    uv run source/audit_controls.py --run "$(ls -1td --time=birth data/output/cmaes/*/ 2>/dev/null | fzf)"

# Quick benchmark on a selected run directory.
ez_benchmark_run:
    uv run main/benchmark_simulator.py --run "$(ls -1td --time=birth data/output/cmaes/*/ 2>/dev/null | fzf)"
