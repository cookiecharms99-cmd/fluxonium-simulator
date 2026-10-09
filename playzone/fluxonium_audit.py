import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import qutip as qt
    from qutip import Qobj, mesolve
    from scqubits import Fluxonium

    return Fluxonium, Path, Qobj, json, np, plt, qt


@app.cell
def _(Path, json):
    # Load data!

    script_dir = Path(__file__).resolve().parent
    data_fp = script_dir.parent / "params.json"
    with open(data_fp, "r") as data_file:
        params = json.load(data_file)
    print(json.dumps(params, indent=4, sort_keys=True))
    return (params,)


@app.cell
def _():
    test_cutoff = 15
    print("Done")
    return (test_cutoff,)


@app.cell
def _(Fluxonium, params):
    # Load some of it into mem

    EJ, EC, EL = params["EJ"], params["EC"], params["EL"]
    cutoff = params["cutoff"]
    flux1, flux2 = params["flux1"], params["flux2"]

    fluxonium = Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=100, flux=flux1)
    fluxonium2 = Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=100, flux=flux2)
    energies, eigenvectors = fluxonium.eigensys(evals_count=6)
    energies2, eigenvectors2 = fluxonium2.eigensys(evals_count=6)
    print("Done")
    return (
        EC,
        EJ,
        EL,
        eigenvectors,
        eigenvectors2,
        energies,
        energies2,
        fluxonium,
        fluxonium2,
    )


@app.cell
def _(energies, energies2, params):
    labels = ["g", "e", "f", "h", "i", "j"]
    print(f"Key values!\n")

    print(f"freq_ef = {params['omega_ef']}")
    print(f"freq_r = {params['omega_r']}")
    for k in range(len(energies) - 1):
        transition = energies[k + 1] - energies[k]
        print(f"{labels[k]} -> {labels[k + 1]} | Transition: {transition:.6f}")

    for k in range(len(energies) - 1):
        transition = energies2[k + 1] - energies2[k]
        print(f"{labels[k]} -> {labels[k + 1]} | Transition 2: {transition:.6f}")

    # Okok flux = 0.25 seems to correspond to \omega stuff :0
    return


@app.cell
def _(eigenvectors, eigenvectors2, fluxonium, fluxonium2, np, params):
    n_op = fluxonium.n_operator(energy_esys=False)  # we are working in LC basis?
    n_op2 = fluxonium2.n_operator()

    def get_matrix_element(op, i, j, eigenvectors):
        bra = eigenvectors[:, i]
        ket = eigenvectors[:, j]
        return np.vdot(bra, op @ ket)

    n_ef = get_matrix_element(n_op, 1, 2, eigenvectors)
    n_gf = get_matrix_element(n_op, 0, 2, eigenvectors)

    n_ef2 = get_matrix_element(n_op2, 1, 2, eigenvectors2)
    n_gf2 = get_matrix_element(n_op2, 0, 2, eigenvectors2)

    n_ef = abs(n_ef)
    n_gf = abs(n_gf)

    n_ef2 = abs(n_ef2)
    n_gf2 = abs(n_gf2)
    print(f"Taken n_ef: {params['n_ef']}")
    print(f"Taken n_gf: {params['n_gf']}")

    print(f"Flux 0.25")
    print(f"Measured n_ef -> {n_ef}")
    print(f"Measured n_gf: {n_gf}")

    print("Flux 0.5")
    print(f"Measured n_ef -> {n_ef2}")
    print(f"Measured n_gf: {n_gf2}")
    return


@app.cell
def _(energies, params):
    print(f"freq_ef = {params['omega_ef']}")
    print(f"freq_r = {params['omega_r']}")
    f_gf = energies[2] - energies[0]
    f_cavity = params["omega_r"]
    detuning = f_cavity - f_gf
    print(f"f_gf -> {f_gf}")
    print(f"f_cavity -> {f_cavity}")
    print(f"Detuning: {detuning}")
    return


@app.cell
def _(EC, EJ, EL, Fluxonium, np, params, plt, test_cutoff):
    _flux_vals = np.linspace(0.1, 0.5, num=200)
    freq_gf_vals = []

    for _flux_frac in _flux_vals:
        test_fluxonium = Fluxonium(
            EJ=EJ, EC=EC, EL=EL, cutoff=test_cutoff, flux=_flux_frac
        )
        test_energies = test_fluxonium.eigenvals(evals_count=3)
        freq_gf = test_energies[2] - test_energies[0]
        freq_gf_vals.append(freq_gf)
    freq_gf_vals = np.array(freq_gf_vals)
    freq_cavity = params["omega_r"]
    plt.plot(_flux_vals, freq_gf_vals)
    plt.axhline(
        freq_cavity,
        color="purple",
        linestyle="--",
        label=rf"Cavity frequency = {freq_cavity:.4f} GHz",
    )
    plt.axvline(
        0.25, color="gray", linestyle=":", label="Calibration bias at 0.25 frac"
    )
    plt.xlabel("External Flux fraction $\Phi_{ext}/\Phi_0$")
    plt.ylabel("Frequency GHz")
    plt.title(r"Fluxonium $g \rightarrow f$ transition near interaction bias")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()
    return freq_cavity, freq_gf_vals


@app.cell
def _():
    return


@app.cell
def _(flux_vals, freq_cavity, freq_gf_vals, np):
    # Find the closest flux value to cavity frequency

    closest_flux_idx = np.argmin(abs(freq_gf_vals - freq_cavity))
    closest_flux = flux_vals[closest_flux_idx]
    best_freq = freq_gf_vals[closest_flux_idx]
    best_detuning = best_freq - freq_cavity
    print(f"Best Flux: {closest_flux}")
    print(f"Best Frequency: {best_freq}")
    print(f"Best detuning: {best_detuning}")
    return


@app.cell
def _(EC, EJ, EL, Fluxonium, freq_cavity, test_cutoff):
    # Optional optimizer, root finder :0

    from scipy.optimize import brentq

    def cavity_detuning(flux_fraction):
        test_fluxonium = Fluxonium(
            EJ=EJ, EC=EC, EL=EL, cutoff=test_cutoff, flux=flux_fraction
        )
        energies = test_fluxonium.eigenvals(evals_count=3)
        freq_gf = energies[2] - energies[0]
        detuning = freq_gf - freq_cavity
        return detuning

    resonant_flux = brentq(cavity_detuning, 0.24, 0.26)
    print(f"Bare g-f / cavity resonance: {resonant_flux:.9f} Phi0")
    print(f"Residual detuning: {cavity_detuning(resonant_flux) * 1e3:.6f} MHz")
    return (resonant_flux,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Next Mission! :D

    - Analyze the actual coupling with f0 and g1 near $\Phi_{ext}$ ~ 0.25
    - Coupled hamiltonian is
      $H = H_{flux}(\phi_{ext}) \otimes I_{cav} + I_{flux} \otimes H_{cav} + g\hat{n} \otimes \sigma_y$
    - hi
    - They aren't eigenstates but we attempt to diagonalize our tensor product hamiltonian against them for energy?
    """)
    return


@app.cell
def _(freq_cavity, np, test_cutoff):
    I_flux = np.eye(test_cutoff)
    sigma_y = np.array([[0, -1j], [1j, 0]])
    I_cavity = np.eye(2)
    H_cavity = freq_cavity * np.diag([0.0, 1.0])
    return H_cavity, I_cavity, I_flux, sigma_y


@app.cell
def _(
    EC,
    EJ,
    EL,
    Fluxonium,
    H_cavity,
    I_cavity,
    I_flux,
    np,
    params,
    sigma_y,
    test_cutoff,
):
    flux_vals = np.linspace(0.245, 0.265, 101)
    lower_branch = []
    upper_branch = []
    for flux_frac in flux_vals:
        flx = Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=test_cutoff, flux=flux_frac)
        H_flux = np.asarray(flx.hamiltonian())
        flux_energies, flux_states = flx.eigensys(
            evals_count=6
        )  # get the actual states :3

    

        n_op_iter = np.asarray(flx.n_operator(energy_esys=False))
        H_total = (
            np.kron(H_flux, I_cavity)
            + np.kron(I_flux, H_cavity)
            + params["g"] * np.kron(n_op_iter, sigma_y)
        )
        dressed_energies, dressed_states = np.linalg.eigh(H_total)
        _f0 = np.kron(flux_states[:, 2], np.array([1, 0]))
        _g1 = np.kron(flux_states[:, 0], np.array([0, 1]))

        f0_weight = np.abs(_f0.conj() @ dressed_states) ** 2
        g1_weight = np.abs(_g1.conj() @ dressed_states) ** 2
        subspace_weight = f0_weight + g1_weight
        branch_idxs = np.argsort(subspace_weight)[-2:]
        branch_energies = np.sort(dressed_energies[branch_idxs])

        lower_branch.append(branch_energies[0] - flux_energies[0])
        upper_branch.append(branch_energies[1] - flux_energies[0])

    lower_branch = np.array(lower_branch)
    upper_branch = np.array(upper_branch)
    print("Done")
    return flux_vals, lower_branch, upper_branch


@app.cell
def _(flux_vals, lower_branch, plt, upper_branch):
    plt.figure(figsize=(8, 5))

    plt.plot(flux_vals, lower_branch, label="Lower dressed branch")
    plt.plot(flux_vals, upper_branch, label="Upper dressed branch")

    plt.axvline(
        0.253980967,
        color="gray",
        linestyle=":",
        label="Bare resonance",
    )

    plt.xlabel(r"External flux $\Phi_{\rm ext}/\Phi_0$")
    plt.ylabel(r"Energy relative to $|g0\rangle$ (GHz)")
    plt.title(r"Dressed $|f0\rangle$-$|g1\rangle$ avoided crossing")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Indeed!**

    This shows that there is coupling otherwise they would have crossed!

    The basic idea seems to be is that without coupling at resonance these energies would have crossed. This supports the pathway for |f0> -> |g1>
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Next mission!!

    Track the populations over time at resonance of f0, g0, g1
    """)
    return


@app.cell
def _(np):
    N_t_res = 100
    T_max_res = 100
    t_arr_res = np.linspace(start=0, stop=T_max_res, num=N_t_res)
    print("Done!")
    return (t_arr_res,)


@app.cell
def _(EC, EJ, EL, Fluxonium, resonant_flux, test_cutoff):

    res_fluxonium = Fluxonium(
        EJ=EJ, EC=EC, EL=EL, cutoff=test_cutoff, flux=resonant_flux
    )

    res_energies, res_states = res_fluxonium.eigensys(evals_count=6)
    n_operator_res = res_fluxonium.n_operator(energy_esys=False)
    print("Done!")
    return n_operator_res, res_fluxonium, res_states


@app.cell
def _():


    return


@app.cell
def _(Qobj, np, res_states, test_cutoff):
    f0_res = np.kron(res_states[:, 2], np.array([1.0, 0]))
    g0_res = np.kron(res_states[:, 0], np.array([1.0, 0]))
    g1_res = np.kron(res_states[:, 0], np.array([0.0, 1]))

    f0_res = Qobj(f0_res, dims=[[test_cutoff, 2], [1, 1]])
    g0_res = Qobj(g0_res, dims=[[test_cutoff, 2], [1, 1]])
    g1_res = Qobj(g1_res, dims=[[test_cutoff, 2], [1, 1]])
    print("Done")
    return f0_res, g0_res, g1_res


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Now we outline our process!

    We have our fluxonium set up at resonance and we will fix it. We will set our initial state at f0.

    We will take density as |s><s|.

    $\rho(0) = |f0><f0|$

    Our populations are:
    $P_{state: s} = <s| \rho(t)|s>$

    Use mesolve like this :3

    `mesolve(
    Hamiltonian,
    Initial State,
    time array, c_ops = c_ops, e_ops = [P_f0, P_g0, P_g1]
    )`

    Q: What are c_ops and e_ops?

    A: c_ops -> collapse operators, irreversible processes in the master equation thingy. Here it is

    $\sqrt{\kappa}a_{full}$

    A: e_op -> expectation operators which mesolve will record the expectation of at each time in our t-array.

    Q: what exactly is kappa?
    """)
    return


@app.cell
def _(np, qt, test_cutoff):
    a_cavity = np.array([[0, 1.0], [0, 0]]) # Sends 1 -> 0
    a_full = qt.tensor(
        qt.qeye(test_cutoff),
        qt.destroy(2),
    ) 
    print("Done!")
    return (a_full,)


@app.cell
def _(f0_res, g0_res, g1_res, qt):
    P_f0_res = f0_res * f0_res.dag()
    P_g0_res = g0_res * g0_res.dag()
    P_g1_res = g1_res * g1_res.dag()
    rho0_res = qt.ket2dm(f0_res)
    print("Done!")
    return P_f0_res, P_g0_res, P_g1_res, rho0_res


@app.cell
def _(P_g0_res, P_g1_res, f0_res, rho0_res):
    for name, obj in [
        ("rho0", rho0_res),
        ("f0", f0_res),
        ("P_g0", P_g0_res),
        ("P_g1", P_g1_res),
    ]:
        print(name, obj.shape, obj.dims)
    return


@app.cell
def _(
    H_cavity,
    I_cavity,
    I_flux,
    P_f0_res,
    P_g0_res,
    P_g1_res,
    Qobj,
    f0_res,
    n_operator_res,
    np,
    params,
    qt,
    res_fluxonium,
    rho0_res,
    sigma_y,
    t_arr_res,
    test_cutoff,
):
    # Build hamiltonia
    # H_flux_res = (4 * EC * n_operator_res @ n_operator_res + 0.5 * EL * phi_op @ phi_op - EJ(cos(phi_op - resonant_flux)))
    # or maybe uh

    H_flux_res = res_fluxonium.hamiltonian()  # ??
    H_res = (
        np.kron(H_flux_res, I_cavity)
        + np.kron(I_flux, H_cavity)
        + params["g"] * np.kron(n_operator_res, sigma_y)
    )
    H_res = Qobj(H_res, dims=[[test_cutoff, 2], [test_cutoff, 2]])
    print(H_res.shape)
    print(H_res.dims)
    print(H_res.isherm)
    print(rho0_res.tr())
    result = qt.sesolve(H_res, f0_res, t_arr_res, e_ops=[P_f0_res, P_g0_res, P_g1_res])

    P_f0_res_t = result.expect[0]
    P_g0_res_t = result.expect[1]
    P_g1_res_t = result.expect[2]
    return H_res, P_f0_res_t, P_g0_res_t, P_g1_res_t


@app.cell
def _(P_f0_res_t, P_g0_res_t, P_g1_res_t, plt, t_arr_res):
    plt.plot(t_arr_res, P_f0_res_t)
    plt.plot(t_arr_res, P_g0_res_t)
    plt.plot(t_arr_res, P_g1_res_t)
    plt.show()
    return


@app.cell
def _(
    H_res,
    P_f0_res,
    P_g0_res,
    P_g1_res,
    a_full,
    np,
    qt,
    rho0_res,
    t_arr_res,
):
    # Temporary/test cavity decay rate
    kappa = 0.01  # 1/ns; photon lifetime = 100 ns

    # Full-system cavity annihilation operator


    # Cavity photon-loss collapse operator
    c_ops = [np.sqrt(kappa) * a_full]

    options = {
        "method": "bdf",
        "atol": 1e-8,
        "rtol": 1e-6,
        "nsteps": 5000,
        "progress_bar": "text",
    }

    result_decay = qt.mesolve(
        H_res,
        rho0_res,
        t_arr_res,
        c_ops=c_ops,
        e_ops=[P_f0_res, P_g0_res, P_g1_res],
        options=options,
    )

    P_f0_res_decay = result_decay.expect[0]
    P_g0_res_decay = result_decay.expect[1]
    P_g1_res_decay = result_decay.expect[2]
    return P_f0_res_decay, P_g0_res_decay, P_g1_res_decay, c_ops, options


@app.cell
def _(P_f0_res_decay, P_g0_res_decay, P_g1_res_decay, plt, t_arr_res):
    BG_COLOR = "#1e1e2e"  # Deep soft dark blue-grey
    TEXT_COLOR = "#cdd6f4"  # Soft warm white/lavender
    GRID_COLOR = "#313244"  # Subtle grid line color

    # Pastel line colors
    color_f0 = "#f38ba8"  # Soft pink
    color_g0 = "#89b4fa"  # Soft blue (for the rising g0)
    color_g1 = "#a6e3a1"  # Soft mint green

    # Set up the figure with the custom background
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Plot with thicker, smooth lines and custom pastel colors
    ax.plot(
        t_arr_res, P_f0_res_decay, label="f0 state", color=color_f0, linewidth=2.5
    )
    ax.plot(
        t_arr_res,
        P_g0_res_decay,
        label="g0 state (the slow riser!)",
        color=color_g0,
        linewidth=2.5,
    )
    ax.plot(
        t_arr_res, P_g1_res_decay, label="g1 state", color=color_g1, linewidth=2.5
    )

    # Style labels, title, and ticks to match the dark theme
    ax.set_xlabel("Time ($t$)", color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("Population ($P$)", color=TEXT_COLOR, fontsize=12)
    ax.set_title(
        "Resonant Flux Dynamics with Cavity Decay ($\kappa$)",
        color=TEXT_COLOR,
        fontsize=14,
        pad=12,
    )

    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    # Subtle, non-intrusive grid
    ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.6)

    # Cute matching legend box
    legend = ax.legend(
        frameon=True,
        facecolor="#313244",
        edgecolor="none",
        labelcolor=TEXT_COLOR,
        fontsize=10,
    )

    plt.tight_layout()

    # Save with the background color intact
    plt.savefig("populations_cav_decay.png.png", dpi=300, facecolor=fig.get_facecolor())
    plt.show()
    return BG_COLOR, GRID_COLOR, TEXT_COLOR, ax, color_f0, color_g0, color_g1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Replicate the same away from resonance to see the picture :3
    """)
    return


@app.cell
def _(
    EC,
    EJ,
    EL,
    Fluxonium,
    H_cavity,
    I_cavity,
    I_flux,
    Qobj,
    np,
    params,
    qt,
    sigma_y,
    test_cutoff,
):
    non_resonance = 0.3
    non_res_flux = Fluxonium(
        EJ=EJ, EC=EC, EL=EL, cutoff=test_cutoff, flux=non_resonance
    )

    non_res_energies, non_res_states = non_res_flux.eigensys(evals_count=6)
    non_res_n_operator = non_res_flux.n_operator(energy_esys=False)
    non_res_f0 = np.kron(non_res_states[:, 2], np.array([1.0, 0]))
    non_res_g0 = np.kron(non_res_states[:, 0], np.array([1.0, 0]))
    non_res_g1 = np.kron(non_res_states[:, 0], np.array([0.0, 1]))

    non_res_f0 = Qobj(non_res_f0, dims=[[test_cutoff, 2], [1, 1]])
    non_res_g0= Qobj(non_res_g0, dims=[[test_cutoff, 2], [1, 1]])
    non_res_g1 = Qobj(non_res_g1, dims=[[test_cutoff, 2], [1, 1]])
    non_res_P_f0 = non_res_f0 * non_res_f0.dag()
    non_res_P_g0 = non_res_g0 * non_res_g0.dag()
    non_res_P_g1 = non_res_g1 * non_res_g1.dag()
    non_res_rho0 = qt.ket2dm(non_res_f0)

    non_res_H_flux= non_res_flux.hamiltonian()  # ??
    non_res_H = Qobj((
        np.kron(non_res_H_flux, I_cavity)
        + np.kron(I_flux, H_cavity)
        + params["g"] * np.kron(non_res_n_operator, sigma_y)
    ))
    non_res_H = Qobj(non_res_H, dims=[[test_cutoff, 2], [test_cutoff, 2]])
    print("Done")
    print(non_res_H.shape)
    print(non_res_H.dims)
    print(non_res_H.isherm)
    print(non_res_rho0.tr())
    return (
        non_res_H,
        non_res_P_f0,
        non_res_P_g0,
        non_res_P_g1,
        non_res_rho0,
        non_resonance,
    )


@app.cell
def _(
    c_ops,
    non_res_H,
    non_res_P_f0,
    non_res_P_g0,
    non_res_P_g1,
    non_res_rho0,
    options,
    qt,
    t_arr_res,
):
    non_res_result_decay = qt.mesolve(
        non_res_H,
        non_res_rho0,
        t_arr_res,
        c_ops=c_ops,
        e_ops=[non_res_P_f0, non_res_P_g0, non_res_P_g1],
        options=options,
    )

    non_res_P_f0_decay = non_res_result_decay.expect[0]
    non_res_P_g0_decay = non_res_result_decay.expect[1]
    non_res_P_g1_decay = non_res_result_decay.expect[2]
    return non_res_P_f0_decay, non_res_P_g0_decay, non_res_P_g1_decay


@app.cell
def _(
    BG_COLOR,
    GRID_COLOR,
    TEXT_COLOR,
    ax,
    color_f0,
    color_g0,
    color_g1,
    non_res_P_f0_decay,
    non_res_P_g0_decay,
    non_res_P_g1_decay,
    non_resonance,
    plt,
    t_arr_res,
):


    # Set up the figure with the custom background
    _fig, _ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
    _ax.set_facecolor(BG_COLOR)

    # Plot with thicker, smooth lines and custom pastel colors
    _ax.plot(
        t_arr_res, non_res_P_f0_decay, label="f0 state", color=color_f0, linewidth=2.5
    )
    _ax.plot(
        t_arr_res,
        non_res_P_g0_decay,
        label="g0 state (the slow riser!)",
        color=color_g0,
        linewidth=2.5,
    )
    _ax.plot(
        t_arr_res, non_res_P_g1_decay, label="g1 state", color=color_g1, linewidth=2.5
    )

    # Style labels, title, and ticks to match the dark theme
    _ax.set_xlabel("Time ($t$)", color=TEXT_COLOR, fontsize=12)
    _ax.set_ylabel("Population ($P$)", color=TEXT_COLOR, fontsize=12)
    _ax.set_title(
        f"Non-resonant Flux Dynamics with Cavity Decay ($\kappa$), Flux = {non_resonance}",
        color=TEXT_COLOR,
        fontsize=14,
        pad=12,
    )

    _ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for _spine in ax.spines.values():
        _spine.set_edgecolor(GRID_COLOR)

    # Subtle, non-intrusive grid
    _ax.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.6)

    # Cute matching legend box
    _legend = _ax.legend(
        frameon=True,
        facecolor="#313244",
        edgecolor="none",
        labelcolor=TEXT_COLOR,
        fontsize=10,
    )

    plt.tight_layout()

    # Save with the background color intact
    plt.savefig("non_res_populations_cav_decay.png", dpi=300, facecolor=_fig.get_facecolor())
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
