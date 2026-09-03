import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import math
import argparse
import numpy as np
import scipy.sparse as sp
from typing import Optional


try:
    import cma
except Exception as e:
    raise ImportError("Missing package 'cma'. Install with: pip install cma") from e

# SciPy DST (for initializing coeffs from an initial guess pulse)
try:
    from scipy.fft import dst
except Exception:
    from scipy.fftpack import dst  # fallback

try:
    import scqubits as scq
except Exception as e:
    raise ImportError("scqubits not found. Install with: pip install scqubits") from e

try:
    from qutip import Qobj, mesolve, Options
except Exception as e:
    raise ImportError("qutip not found. Install with: pip install qutip") from e

import multiprocessing as mp

_PHYS = None  # per-process physics cache


def build_sin_basis(t: np.ndarray, T: float, n_coeffs: int) -> np.ndarray:
    k = np.arange(1, n_coeffs + 1, dtype=float)[:, None]
    return np.sin(np.pi * k * t[None, :] / T)


def init_physics(params: dict) -> dict:
    EJ = params["EJ"]
    EC = params["EC"]
    EL = params["EL"]
    cutoff = params["cutoff"]
    flux1 = params["flux1"]
    flux2 = params["flux2"]

    Omega = params["Omega"]
    omega_q = params["omega_q"]
    omega_ef = params["omega_ef"]
    omega_r = params["omega_r"]
    n_ef = params["n_ef"]
    n_gf = params["n_gf"]
    g = params["g"]

    t = params["t"]
    T = params["T"]
    n_coeffs = params["n_coeffs"]

    fluxonium = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=flux1)
    fluxonium2 = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=flux2)

    # ---- force CSR sparse operators to avoid dense Liouvillian ----
    I_c = sp.eye(cutoff, format="csr")
    I2  = sp.eye(2, format="csr")

    def kron(A, B):
        return sp.kron(A, B, format="csr")

    cos_op = sp.csr_matrix(fluxonium.cos_phi_operator())
    sin_op = sp.csr_matrix(fluxonium.sin_phi_operator())
    n_op   = sp.csr_matrix(fluxonium.n_operator())

    cos_phi  = Qobj(kron(cos_op, I2))
    sin_phi  = Qobj(kron(sin_op, I2))
    n_tensor = Qobj(kron(n_op,   I2))

    levels = (np.arange(cutoff, dtype=float) + 0.5)
    Hq = Qobj(kron(sp.diags(omega_q * levels, 0, format="csr"), I2))

    Hr2 = sp.csr_matrix(omega_r * np.array([[0.5, 0.0], [0.0, 1.5]], dtype=float))
    Hr  = Qobj(kron(I_c, Hr2))

    sigma_y = sp.csr_matrix(np.array([[0.0, -1j], [1j, 0.0]], dtype=complex))
    Hqr = Qobj(g * kron(n_op, sigma_y))

    H0 = Hq + Hr + Hqr

    gamma = 2 * np.pi / 40.0
    c_mat = kron(I_c, sp.csr_matrix(np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=float)))
    c_ops = [Qobj(c_mat)]


    # Initial state uses flux=flux2 (0.5) eigenstates
    _, states = fluxonium2.eigensys()
    g0 = Qobj(np.kron(states[:, 0], np.array([1.0, 0.0])).reshape(-1, 1))
    e0 = Qobj(np.kron(states[:, 1], np.array([1.0, 0.0])).reshape(-1, 1))
    rho0 = 0.5 * (g0 * g0.dag()) + 0.5 * (e0 * e0.dag())

    e_ops = (g0 * g0.dag()) # reward projector (flux=0.5 ground state)

    options = Options(
        nsteps=params["qutip_nsteps"],
        atol=params["qutip_atol"],
        rtol=params["qutip_rtol"],
        normalize_output=True,
    )

    # optional method
    if params.get("qutip_method", None) is not None:
        options["method"] = params["qutip_method"]

    sin_basis = build_sin_basis(t, T, n_coeffs)

    return {
        "params": params,
        "cos_phi": cos_phi,
        "sin_phi": sin_phi,
        "n_tensor": n_tensor,
        "H0": H0,
        "c_ops": c_ops,
        "rho0": rho0,
        "e_ops": e_ops,
        "options": options,
        "sin_basis": sin_basis,
    }


def _worker_init(params: dict):
    global _PHYS
    _PHYS = init_physics(params)


def pulse_from_coeffs(coeffs: np.ndarray, sin_basis: np.ndarray, u_final: float, A_max: Optional[float]):
    pulse = u_final + coeffs @ sin_basis
    if A_max is not None:
        pulse = A_max * np.tanh(pulse / A_max)
    return pulse


def evolve_with_pulse(pulse: np.ndarray, params) -> float:
    global _PHYS
    if not _PHYS:
        P = init_physics(params)
    else:
        P = init_physics(_PHYS["params"])  # re-init physics to ensure pulse is up-to-date (could optimize by caching pulse-specific parts)

    t = P["params"]["t"]
    EJ = P["params"]["EJ"]
    Omega = P["params"]["Omega"]
    omega_ef = P["params"]["omega_ef"]
    n_ef = P["params"]["n_ef"]

    cosphi = np.cos(pulse)
    sinphi = np.sin(pulse)
    Asin = (Omega / n_ef) * np.sin(omega_ef * t)

    H = [
        P["H0"],
        [-EJ * P["cos_phi"], cosphi],
        [EJ * P["sin_phi"], sinphi],
        [P["n_tensor"], Asin],
    ]

    result = mesolve(H, P["rho0"], t, P["c_ops"], e_ops=[P["e_ops"]], options=P["options"])
    return float(result.expect[0][-1])  # Pg0 at flux=0.5


def _simulate_one(coeffs_np: np.ndarray) -> float:
    global _PHYS
    P = _PHYS
    pulse = pulse_from_coeffs(coeffs_np, P["sin_basis"], P["params"]["u_final"], P["params"]["A_max"])
    return evolve_with_pulse(pulse)

def main():
    parser = argparse.ArgumentParser()

    # CMA-ES
    parser.add_argument("--iters", type=int, default=50,
                        help="Number of CMA-ES generations (iterations of ask→evaluate→tell).")
    parser.add_argument("--popsize", type=int, default=0,
                        help="Population size (#candidates per generation). 0 uses CMA-ES default heuristic.")
    parser.add_argument("--sigma0", type=float, default=0.3,
                        help="Initial global step-size (sampling std scale). Larger explores more broadly.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for NumPy and CMA-ES (reproducibility).")
    parser.add_argument("--clip_coeff", type=float, default=0.0,
                        help=("If >0, clip sampled coefficients to [-clip_coeff, +clip_coeff] "
                              "before simulation (hard safety clamp)."))
    parser.add_argument("--coeff_bound", type=float, default=0.0,
                        help=("If >0, set CMA-ES box constraints (bounds) to [-coeff_bound, +coeff_bound]. "
                              "This affects sampling/search, not just evaluation-time clipping."))

    # Parallel + output
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes for objective evaluations (pool size).")
    parser.add_argument("--out_dir", type=str, default="out_autotune_cmaes",
                        help="Output directory for checkpoints, best coefficients/pulse, and logs.")

    # Time grid / pulse
    parser.add_argument("--T", type=float, default=150.0,
                        help="Total pulse duration (same units as your Hamiltonian uses, e.g., ns).")
    parser.add_argument("--nt", type=int, default=100,
                        help="Number of time points in the simulation grid (time discretization).")
    parser.add_argument("--A_max", type=float, default=1.0,
                        help=("Amplitude limiter for the pulse via tanh: pulse := A_max*tanh(pulse/A_max). "
                              "Set A_max < 0 to disable the tanh limiter entirely."))

    # QuTiP
    parser.add_argument("--qutip_nsteps", type=int, default=20000,
                        help="Maximum number of internal ODE steps allowed per time interval in QuTiP solver.")
    parser.add_argument("--qutip_atol", type=float, default=1e-8,
                        help="Absolute tolerance for QuTiP ODE solver (smaller = more accurate, slower).")
    parser.add_argument("--qutip_rtol", type=float, default=1e-6,
                        help="Relative tolerance for QuTiP ODE solver (smaller = more accurate, slower).")
    parser.add_argument("--qutip_method", type=str, default="",
                        help=("Optional QuTiP ODE method, e.g. 'adams' (non-stiff) or 'bdf' (stiff). "
                              "Leave empty to use QuTiP's default choice."))

    # Early stopping
    parser.add_argument("--target", type=float, default=0.999,
                        help="Early-stop target reward: stop when best_reward >= target.")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early-stop patience: stop after this many generations without meaningful improvement.")
    parser.add_argument("--min_improve", type=float, default=1e-6,
                        help="Minimum improvement in best_reward to reset patience counter (plateau threshold).")


    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed)

    # ---- physics params ----
    EJ = 4.164
    EC = 1.531
    EL = 0.685
    cutoff = 20
    flux1 = 0.25
    flux2 = 0.5

    phi0 = np.pi / 2
    u_final = 2 * phi0

    Omega = 71e-3 * 3
    omega_q = 2.89652896
    omega_ef = 1.78991881
    omega_gf = 6.5172172187
    omega_r = omega_gf

    n_ef = 0.304437775
    n_gf = 0.374859134
    g = 56e-3 * 2 * np.pi / n_gf / 2

    n_coeffs = 40
    T = float(args.T)
    t = np.linspace(0.0, T, int(args.nt))

    A_max = None if args.A_max < 0 else float(args.A_max)

    # initial guess pulse (piecewise)
    init_pulse = np.zeros_like(t)
    for i, time_t in enumerate(t):
        if time_t < 10:
            init_pulse[i] = 2 * phi0 - phi0 * time_t / 10
        elif time_t < 110:
            init_pulse[i] = phi0
        else:
            init_pulse[i] = phi0 + phi0 * (time_t - 110) / 40

    # DST init for x0
    best_path = os.path.join(args.out_dir, "best_coeffs.npy")
    if os.path.exists(best_path):
        x0 = np.load(best_path).astype(np.float64)
    else:
        sin_fft = dst(init_pulse - u_final) / init_pulse.shape[0]
        x0 = np.real(sin_fft[:n_coeffs]).astype(np.float64)


    phys_params = dict(
        EJ=EJ, EC=EC, EL=EL, cutoff=cutoff,
        flux1=flux1, flux2=flux2,
        phi0=phi0, u_final=u_final,
        Omega=Omega, omega_q=omega_q, omega_ef=omega_ef, omega_gf=omega_gf, omega_r=omega_r,
        n_ef=n_ef, n_gf=n_gf, g=g,
        n_coeffs=n_coeffs, T=T, t=t,
        A_max=A_max,
        qutip_nsteps=int(args.qutip_nsteps),
        qutip_atol=float(args.qutip_atol),
        qutip_rtol=float(args.qutip_rtol),
    )
    if args.qutip_method.strip():
        phys_params["qutip_method"] = args.qutip_method.strip()

    # init physics in main (for saving best pulse)
    global _PHYS
    _PHYS = init_physics(phys_params)

    # multiprocessing pool
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(phys_params,),
        maxtasksperchild=10
    )

    # CMA options
    cma_opts = {
        "seed": int(args.seed),
        "verbose": -9,  # CMA-ES verbosity level: -9 = very quiet (you print your own logs)
    }
    if args.popsize and args.popsize > 0:
        cma_opts["popsize"] = int(args.popsize)
    if args.coeff_bound and args.coeff_bound > 0:
        b = float(args.coeff_bound)
        cma_opts["bounds"] = [-b, b]

    es = cma.CMAEvolutionStrategy(x0, float(args.sigma0), cma_opts)

    best_reward = -1e9
    best_coeffs = None
    history = []

    # ---- early stop controls  ----
    target = float(args.target)             # Reward threshold to stop immediately when reached
    patience = int(args.patience)           # Max generations allowed without meaningful improvement
    min_improve = float(args.min_improve)   # Improvement threshold to be counted as "meaningful"

    best_so_far = best_reward
    no_improve = 0

    # --- convergence stop (population collapsed) ---
    sigma_min = 0.004       # Stop if step-size becomes very small (tune ~0.003–0.006)
    std_min   = 2e-5        # Stop if population reward std is tiny (tune ~1e-5–5e-5)
    mean_gap  = 5e-5        # Stop if mean reward is extremely close to best (best - mean < mean_gap)
    conv_patience = 10      # Require collapse condition for several consecutive gens (avoid random triggers)
    conv_cnt = 0

    start = time.time()
    try:
        for gen in range(int(args.iters)):
            X = es.ask()  # list of candidate vectors

            # prepare candidates (optional clip before sim)
            cand = []
            for x in X:
                x = np.asarray(x, dtype=np.float64)
                if args.clip_coeff and args.clip_coeff > 0:
                    c = float(args.clip_coeff)
                    x = np.clip(x, -c, c)
                cand.append(x)

            # parallel evaluate rewards
            rewards = pool.map(_simulate_one, cand)

            # CMA-ES minimizes; we want maximize reward -> minimize (1 - reward)
            losses = [1.0 - float(r) for r in rewards]
            es.tell(X, losses)

            # bookkeeping
            r_best = float(np.max(rewards))
            r_mean = float(np.mean(rewards))
            r_std = float(np.std(rewards))
            sigma = float(es.sigma)

            idx = int(np.argmax(rewards))
            if r_best > best_reward:
                best_reward = r_best
                best_coeffs = cand[idx].copy()

            history.append([gen, best_reward, r_best, r_mean, r_std, sigma, len(X)])

            print(
                f"Gen {gen:03d} | bestEver={best_reward:.6f} | "
                f"bestGen={r_best:.6f} | mean={r_mean:.6f} | std={r_std:.6f} | "
                f"sigma={sigma:.4f} | pop={len(X)}"
            )


            # ---- early stop: hit target OR plateau ----
            if best_reward >= target:
                print(f"[STOP] reached >= {target}")
                break

            if best_reward > best_so_far + min_improve:
                best_so_far = best_reward
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[STOP] no improvement for {patience} generations (best={best_reward:.6f})")
                    break

            # 3) convergence (population collapsed near best)
            if (sigma < sigma_min) and (r_std < std_min) and ((best_reward - r_mean) < mean_gap):
                conv_cnt += 1
            else:
                conv_cnt = 0

            if conv_cnt >= conv_patience:
                print(f"[STOP] converged: sigma={sigma:.4g}, std={r_std:.4g}, mean_gap={(best_reward-r_mean):.3g}")
                break


        # save outputs
        np.save(os.path.join(args.out_dir, "best_coeffs.npy"), best_coeffs)
        best_pulse = pulse_from_coeffs(best_coeffs, _PHYS["sin_basis"], u_final, A_max)
        np.save(os.path.join(args.out_dir, "best_pulse.npy"), best_pulse)

        # save CMA mean + sigma + (optional) best
        np.save(os.path.join(args.out_dir, "cma_mean.npy"), np.asarray(es.mean, dtype=np.float64))
        with open(os.path.join(args.out_dir, "cma_state.txt"), "w", encoding="utf-8") as f:
            f.write(f"best_reward={best_reward}\n")
            f.write(f"sigma={es.sigma}\n")
            f.write(f"popsize={es.popsize}\n")

        hist_path = os.path.join(args.out_dir, "history.csv")
        np.savetxt(
            hist_path,
            np.array(history, dtype=float),
            delimiter=",",
            header="gen,best_ever,best_gen,mean_gen,std_gen,sigma,popsize",
            comments="",
        )

        print(f"\nDONE. Best reward={best_reward:.6f}")
        print(f"Saved to: {args.out_dir}")
        print(f"Total minutes: {(time.time() - start)/60:.2f}")

    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
