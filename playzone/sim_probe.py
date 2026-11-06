import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import csv
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_RUNS_DIRECTORY = PROJECT_ROOT / "data" / "output" / "cmaes"

    sys.path.insert(0, str(PROJECT_ROOT))

    from source.simulator import (
        FluxoniumResetSimulator,  # noqa: E402
        sine_basis,
    )



    return (
        DEFAULT_RUNS_DIRECTORY,
        FluxoniumResetSimulator,
        PROJECT_ROOT,
        Path,
        json,
        np,
        plt,
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scratchwork probing

    ## **Experiments**

    1. Plot weight pasting to see if it works and check the fidelity
    2. idk :3
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Experiment 1

    **Plotting**

    first we handle the importing of data
    """)
    return


@app.cell
def _(DEFAULT_RUNS_DIRECTORY):
    # This is the directory for the 150 nanosecond run

    old_run_dir = "reference_6h_legacy_4776599"

    # This is the directory for the continued run from that

    new_run_dir = "continuation_140ns_from_reference_6h_legacy_20261029T062442Z"

    old_run_path = DEFAULT_RUNS_DIRECTORY / old_run_dir

    new_run_path = DEFAULT_RUNS_DIRECTORY / new_run_dir
    return new_run_path, old_run_path


@app.cell
def _(new_run_path, np, old_run_path):
    # With this fetch the best weights of old and initial weights of new (should be the same)! :D
    old_data_path = f"{old_run_path}/best_weights.npy"
    best_weights_150 = np.load(old_data_path)
    initial_data_path = f"{new_run_path}/initial_weights.npy"
    initial_weights_140  = np.load(initial_data_path)

    delta = initial_weights_140 - best_weights_150
    print("Summary of fetching!")
    print(f"Initial weights for 140ns-> {initial_weights_140}")
    print(f"Best weights from 150ns -> {best_weights_150}")
    print(f"Max difference -> {np.max(delta)}")
    return best_weights_150, initial_weights_140


@app.cell
def _(FluxoniumResetSimulator, PROJECT_ROOT, Path, json):
    # Next we also need to load the config which isn't too hard

    reference_config_path = PROJECT_ROOT / "config" / "reference_prof.json"

    def load_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    reference_config = load_json(reference_config_path)
    reference_config["model_settings"]["cutoff"] = 13

    print(reference_config)
    print("\n\nLoaded! :D")
    # Make the corresponding sim


    sim = FluxoniumResetSimulator(reference_config)
    print("Constructed!")
    return reference_config, sim


@app.cell
def _(best_weights_150, initial_weights_140, np, plt, reference_config, sim):
    num_t_steps = reference_config["model_settings"]["num_t_steps"]

    times_150_ns = np.linspace(0.0,150.0, num_t_steps)
    times_140_ns = np.linspace(0,140.0, num_t_steps)
    times_unit = times_150_ns / 150
    controls_150ns = sim.reference_controls(flux_weights = best_weights_150, times = times_150_ns)
    controls_140ns = sim.reference_controls(flux_weights = initial_weights_140, times = times_140_ns)

    flux_150ns = controls_150ns.external_phase
    flux_140ns = controls_140ns.external_phase

    _fig, _axes = plt.subplots(2)
    _ax1, _ax2 = _axes

    _ax1.plot(times_unit, flux_150ns)
    _ax1.set_title("150 ns best flux")
    _ax1.set_xlabel("time")
    _ax1.set_ylabel("flux phase")
    _ax2.plot(times_unit, flux_140ns)
    _ax2.set_title("140 ns best flux")
    _ax2.set_xlabel("time")
    _ax2.set_ylabel("flux phase")

    plt.tight_layout()
    plt.show()
    return times_140_ns, times_150_ns


@app.cell
def _(best_weights_150, initial_weights_140, sim, times_140_ns, times_150_ns):
    final_pop_150 = sim.reference_final_ground_population(flux_weights = best_weights_150, times = times_150_ns)
    final_pop_140 = sim.reference_final_ground_population(flux_weights = initial_weights_140, times = times_140_ns)

    print(f"140 (ns): {final_pop_140} | 150 : {final_pop_150}")
    return


app._unparsable_cell(
    r"""
    print(sim._reference_times - sim._times)\
    """,
    name="_"
)


@app.cell
def _():
    return


@app.cell
def _(P_e0, P_f0, P_g0, P_g1, controls, leakage, np, plt, times_150_ns):

    _fig, (ax_pop, ax_ctrl) = plt.subplots(
        2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )

    # ---------------- Top Panel: Pathway Populations ----------------
    ax_pop.plot(times_150_ns, P_g0, label=r"$|g,0\rangle$ (Target ground)", color="#2ca02c", lw=2.4)
    ax_pop.plot(times_150_ns, P_e0, label=r"$|e,0\rangle$ (Initial qubit)", color="#1f77b4", lw=1.8)
    ax_pop.plot(times_150_ns, P_f0, label=r"$|f,0\rangle$ (Drive intermediate)", color="#ff7f0e", lw=1.8)
    ax_pop.plot(times_150_ns, P_g1, label=r"$|g,1\rangle$ (Cavity photon)", color="#9467bd", lw=1.8, ls="--")
    ax_pop.plot(times_150_ns, leakage, label=r"Leakage ($1 - \sum P_i$)", color="#d62728", lw=1.5, ls=":")

    # Highlight Target Benchmark (0.999)
    ax_pop.axhline(0.999, color="black", ls="--", alpha=0.4, label="Target: 0.999")
    ax_pop.set_ylabel("State Population", fontsize=11, fontweight="semibold")
    ax_pop.set_ylim(-0.02, 1.04)
    ax_pop.grid(True, linestyle="--", alpha=0.4)
    ax_pop.legend(loc="upper left", framealpha=0.9, fontsize=9.5)
    ax_pop.set_title(
        f"150-ns Reference Reset Dynamics — Final $P(g,0) = {P_g0[-1]:.5f}$",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    # ---------------- Bottom Panel: Applied Controls ----------------
    # Plot flux trajectory in physical units (Phi / Phi_0)
    flux_in_units = controls.external_phase / (2.0 * np.pi)
    ax_ctrl.plot(times_150_ns, flux_in_units, color="#333333", lw=2.0, label=r"$\Phi_{\rm ext}(t) / \Phi_0$")
    ax_ctrl.axhline(0.25, color="crimson", ls=":", alpha=0.7, label=r"Calibrated $0.25\,\Phi_0$")
    ax_ctrl.axhline(0.50, color="gray", ls="--", alpha=0.5, label=r"Idle $0.50\,\Phi_0$")

    ax_ctrl.set_xlabel("Time (ns)", fontsize=11, fontweight="semibold")
    ax_ctrl.set_ylabel(r"Flux [$\Phi_0$]", fontsize=10, fontweight="semibold")
    ax_ctrl.grid(True, linestyle="--", alpha=0.4)
    ax_ctrl.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax_ctrl.set_ylim(0, 0.70)

    plt.tight_layout()
    plt.gca()
    plt.show()
    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
