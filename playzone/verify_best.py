# verify_best_fluxonium_pulse.py
# Full verification script:
#  - loads best_coeffs.npy from OUT_DIR
#  - rebuilds physics (sparse CSR, matches CMA-ES version)
#  - reconstructs pulse on chosen NT
#  - runs QuTiP mesolve and prints reward
#  - prints pulse/coeff stats to diagnose bounds/limits
#
# Usage:
#   python verify_best_fluxonium_pulse.py
#   (edit OUT_DIR / NT / A_max / tolerances below)

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import numpy as np
import scipy.sparse as sp
import scqubits as scq
from qutip import Qobj, mesolve, Options

# ===================== user knobs =====================
# OUT_DIR = "out_autotune_cmaes"
OUT_DIR="out_nt200_bound4_restart96"
BEST_PATH = os.path.join(OUT_DIR, "best_coeffs.npy")

T = 150.0
NT = 400                 # try 100, 200, 400
A_max = None             # None => no limiter; e.g. 2.0 to limit modulation
LIMIT_MODE = "delta"     # "delta" (recommended) or "absolute"
# QuTiP solver accuracy for verification
QUTIP_NSTEPS = 200000
ATOL = 1e-10
RTOL = 1e-8
# ======================================================


def build_sin_basis(t: np.ndarray, T: float, n_coeffs: int) -> np.ndarray:
    k = np.arange(1, n_coeffs + 1, dtype=float)[:, None]
    return np.sin(np.pi * k * t[None, :] / T)


def pulse_from_coeffs(coeffs: np.ndarray, sin_basis: np.ndarray, u_final: float, A_max, limit_mode: str):
    """
    limit_mode:
      - "delta": limit only modulation delta(t) = coeffs @ sin_basis  (recommended)
      - "absolute": limit whole pulse u_final + delta(t) (can crush u_final if u_final >> A_max)
    """
    delta = coeffs @ sin_basis
    if A_max is not None:
        if limit_mode == "delta":
            delta = A_max * np.tanh(delta / A_max)
        elif limit_mode == "absolute":
            pulse0 = u_final + delta
            pulse0 = A_max * np.tanh(pulse0 / A_max)
            return pulse0
        else:
            raise ValueError("LIMIT_MODE must be 'delta' or 'absolute'")
    return u_final + delta


def kron(A, B):
    return sp.kron(A, B, format="csr")


def main():
    if not os.path.exists(BEST_PATH):
        raise FileNotFoundError(f"best coeffs not found: {BEST_PATH}")

    coeffs = np.load(BEST_PATH).astype(np.float64)
    n_coeffs = int(coeffs.shape[0])

    # ---- physics params (match your CMA-ES script) ----
    EJ = 4.164
    EC = 1.531
    EL = 0.685
    cutoff = 20
    flux1 = 0.25
    flux2 = 0.5

    phi0 = np.pi / 2
    u_final = 2 * phi0  # = pi

    Omega = 71e-3 * 3
    omega_q = 2.89652896
    omega_ef = 1.78991881
    omega_gf = 6.5172172187
    omega_r = omega_gf

    n_ef = 0.304437775
    n_gf = 0.374859134
    g = 56e-3 * 2 * np.pi / n_gf / 2

    # ---- time grid + pulse ----
    t = np.linspace(0.0, T, int(NT))
    sin_basis = build_sin_basis(t, T, n_coeffs)
    pulse = pulse_from_coeffs(coeffs, sin_basis, u_final, A_max, LIMIT_MODE)

    # ---- print diagnostics ----
    print("=== Diagnostics ===")
    print(f"NT={NT}, T={T}, n_coeffs={n_coeffs}, A_max={A_max}, LIMIT_MODE={LIMIT_MODE}")
    print(f"coeffs: min={coeffs.min():+.6f}, max={coeffs.max():+.6f}, mean={coeffs.mean():+.6f}, std={coeffs.std():.6f}")
    print(f"pulse : min={pulse.min():+.6f}, max={pulse.max():+.6f}, mean={pulse.mean():+.6f}, std={pulse.std():.6f}")
    print("===================")

    # ---- build fluxonium ----
    fluxonium = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=flux1)
    fluxonium2 = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=flux2)

    # ---- force CSR sparse operators to match your CMA script ----
    I_c = sp.eye(cutoff, format="csr")
    I2 = sp.eye(2, format="csr")

    cos_op = sp.csr_matrix(fluxonium.cos_phi_operator())
    sin_op = sp.csr_matrix(fluxonium.sin_phi_operator())
    n_op = sp.csr_matrix(fluxonium.n_operator())

    cos_phi = Qobj(kron(cos_op, I2))
    sin_phi = Qobj(kron(sin_op, I2))
    n_tensor = Qobj(kron(n_op, I2))

    levels = (np.arange(cutoff, dtype=float) + 0.5)
    Hq = Qobj(kron(sp.diags(omega_q * levels, 0, format="csr"), I2))

    Hr2 = sp.csr_matrix(omega_r * np.array([[0.5, 0.0], [0.0, 1.5]], dtype=float))
    Hr = Qobj(kron(I_c, Hr2))

    sigma_y = sp.csr_matrix(np.array([[0.0, -1j], [1j, 0.0]], dtype=complex))
    Hqr = Qobj(g * kron(n_op, sigma_y))

    H0 = Hq + Hr + Hqr

    # ---- dissipation ----
    gamma = 2 * np.pi / 40.0
    c_mat = kron(I_c, sp.csr_matrix(np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=float)))
    c_ops = [Qobj(c_mat)]

    # ---- initial rho0 + reward projector (flux=0.5 eigenstates) ----
    _, states = fluxonium2.eigensys()
    g0 = Qobj(np.kron(states[:, 0], np.array([1.0, 0.0])).reshape(-1, 1))
    e0 = Qobj(np.kron(states[:, 1], np.array([1.0, 0.0])).reshape(-1, 1))
    rho0 = 0.5 * (g0 * g0.dag()) + 0.5 * (e0 * e0.dag())
    e_ops = g0 * g0.dag()

    # ---- controls ----
    cosphi = np.cos(pulse)
    sinphi = np.sin(pulse)
    Asin = (Omega / n_ef) * np.sin(omega_ef * t)

    H = [
        H0,
        [-EJ * cos_phi, cosphi],
        [EJ * sin_phi, sinphi],
        [n_tensor, Asin],
    ]

    options = Options(
        nsteps=int(QUTIP_NSTEPS),
        atol=float(ATOL),
        rtol=float(RTOL),
        normalize_output=True,
    )

    result = mesolve(H, rho0, t, c_ops, e_ops=[e_ops], options=options)
    reward = float(result.expect[0][-1])

    print(f"\n[VERIFY] NT={NT} reward={reward:.9f}")


if __name__ == "__main__":
    main()
