import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import math
import json
import hashlib
import argparse
import traceback
import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple, Dict, Any, List

try:
    import cma
except Exception as e:
    raise ImportError("Missing package 'cma'. Install with: pip install cma") from e

# SciPy DST
try:
    from scipy.fft import dst
except Exception:
    from scipy.fftpack import dst

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


# =========================
# helpers
# =========================
def build_sin_basis(t: np.ndarray, T: float, n_coeffs: int) -> np.ndarray:
    k = np.arange(1, n_coeffs + 1, dtype=float)[:, None]
    return np.sin(np.pi * k * t[None, :] / T)


def make_default_init_pulse(t: np.ndarray, phi0: float) -> np.ndarray:
    """
    Original piecewise init pulse, vectorized.
    """
    pulse = np.empty_like(t, dtype=float)

    mask1 = (t < 10)
    mask2 = (t >= 10) & (t < 110)
    mask3 = (t >= 110)

    pulse[mask1] = 2 * phi0 - phi0 * t[mask1] / 10.0
    pulse[mask2] = phi0
    pulse[mask3] = phi0 + phi0 * (t[mask3] - 110.0) / 40.0
    return pulse


def resolve_workers(user_workers: int) -> int:
    if user_workers and user_workers > 0:
        return int(user_workers)

    ncpu = os.cpu_count() or 1
    if ncpu <= 2:
        return 1
    return max(1, min(8, ncpu - 1))


def resolve_popsize(user_popsize: int, n_coeffs: int, workers: int) -> int:
    if user_popsize and user_popsize > 0:
        return int(user_popsize)

    base = 4 + int(3 * np.log(max(2, n_coeffs)))
    return max(base, 2 * workers)


def resolve_sigma0(user_sigma0: float, x0: np.ndarray, coeff_bound: float) -> float:
    if user_sigma0 and user_sigma0 > 0:
        return float(user_sigma0)

    s = float(np.std(x0))
    if not np.isfinite(s) or s < 1e-8:
        s = 0.3

    sigma0 = 0.5 * s
    sigma0 = float(np.clip(sigma0, 0.05, 0.5))

    if coeff_bound and coeff_bound > 0:
        sigma0 = min(sigma0, 0.25 * float(coeff_bound))

    return float(sigma0)


def resolve_restart_params(
    user_restarts: int,
    user_restart_growth: float,
    user_restart_local_jitter: float,
    user_restart_global_jitter: float,
    user_restart_min_sigma: float,
    user_restart_max_sigma: float,
    base_sigma0: float,
    workers: int,
):
    if user_restarts >= 0:
        restarts = int(user_restarts)
    else:
        if base_sigma0 < 0.08:
            restarts = 3
        elif base_sigma0 < 0.18:
            restarts = 4
        else:
            restarts = 5

    if user_restart_growth > 0:
        restart_growth = float(user_restart_growth)
    else:
        restart_growth = 1.25 if workers >= 4 else 1.18

    if user_restart_local_jitter > 0:
        restart_local_jitter = float(user_restart_local_jitter)
    else:
        restart_local_jitter = 0.12

    if user_restart_global_jitter > 0:
        restart_global_jitter = float(user_restart_global_jitter)
    else:
        restart_global_jitter = 1.00

    if user_restart_min_sigma > 0:
        restart_min_sigma = float(user_restart_min_sigma)
    else:
        restart_min_sigma = max(0.02, 0.20 * base_sigma0)

    if user_restart_max_sigma > 0:
        restart_max_sigma = float(user_restart_max_sigma)
    else:
        restart_max_sigma = min(0.5, 1.80 * base_sigma0)

    return (
        restarts,
        restart_growth,
        restart_local_jitter,
        restart_global_jitter,
        restart_min_sigma,
        restart_max_sigma,
    )


def pulse_from_coeffs(
    coeffs: np.ndarray,
    sin_basis: np.ndarray,
    u_final: float,
    A_max: Optional[float]
) -> np.ndarray:
    pulse = u_final + coeffs @ sin_basis
    if A_max is not None:
        pulse = A_max * np.tanh(pulse / A_max)
    return pulse


def json_sha1(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_resume_fingerprint(phys_params: Dict[str, Any]) -> str:
    fp = {
        "EJ": round(float(phys_params["EJ"]), 12),
        "EC": round(float(phys_params["EC"]), 12),
        "EL": round(float(phys_params["EL"]), 12),
        "cutoff": int(phys_params["cutoff"]),
        "flux1": round(float(phys_params["flux1"]), 12),
        "flux2": round(float(phys_params["flux2"]), 12),
        "phi0": round(float(phys_params["phi0"]), 12),
        "u_final": round(float(phys_params["u_final"]), 12),
        "Omega": round(float(phys_params["Omega"]), 12),
        "omega_q": round(float(phys_params["omega_q"]), 12),
        "omega_ef": round(float(phys_params["omega_ef"]), 12),
        "omega_gf": round(float(phys_params["omega_gf"]), 12),
        "omega_r": round(float(phys_params["omega_r"]), 12),
        "n_ef": round(float(phys_params["n_ef"]), 12),
        "n_gf": round(float(phys_params["n_gf"]), 12),
        "g": round(float(phys_params["g"]), 12),
        "n_coeffs": int(phys_params["n_coeffs"]),
        "T": round(float(phys_params["T"]), 12),
        "nt": int(len(phys_params["t"])),
        "A_max": None if phys_params["A_max"] is None else round(float(phys_params["A_max"]), 12),
        "qutip_nsteps": int(phys_params["qutip_nsteps"]),
        "qutip_atol": float(phys_params["qutip_atol"]),
        "qutip_rtol": float(phys_params["qutip_rtol"]),
        "qutip_method": phys_params.get("qutip_method", ""),
    }
    return json_sha1(fp)


def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_warm_start(
    out_dir: str,
    init_pulse: np.ndarray,
    u_final: float,
    n_coeffs: int,
    no_resume: bool,
    resume_fingerprint: str,
    allow_legacy_resume: bool,
) -> Tuple[np.ndarray, Optional[float], str, bool, str]:
    """
    优先级：
      1) best_coeffs.npy
      2) cma_mean.npy
      3) DST(init_pulse - u_final)

    只有 fingerprint 匹配才允许 resume；否则回退到 DST。
    返回: x0, sigma_hint, source, resume_ok, resume_note
    """
    def dst_fallback(note: str):
        sin_fft = dst(init_pulse - u_final) / init_pulse.shape[0]
        x0 = np.real(sin_fft[:n_coeffs]).astype(np.float64)
        return x0, None, "dst(init_pulse)", False, note

    if no_resume:
        return dst_fallback("resume_disabled_by_flag")

    best_path = os.path.join(out_dir, "best_coeffs.npy")
    mean_path = os.path.join(out_dir, "cma_mean.npy")
    state_path = os.path.join(out_dir, "cma_state.json")
    run_summary_path = os.path.join(out_dir, "run_summary.json")

    state = load_json_if_exists(state_path) or {}
    summary = load_json_if_exists(run_summary_path) or {}

    stored_fingerprint = summary.get("resume_fingerprint") or state.get("resume_fingerprint")
    legacy_state = stored_fingerprint is None
    fingerprint_ok = (stored_fingerprint == resume_fingerprint)

    if legacy_state and not allow_legacy_resume:
        return dst_fallback("legacy_resume_blocked_no_fingerprint")

    if (not legacy_state) and (not fingerprint_ok):
        return dst_fallback("resume_fingerprint_mismatch")

    sigma_hint = None
    if "sigma" in state:
        try:
            sigma_hint = float(state["sigma"])
        except Exception:
            sigma_hint = None

    if os.path.exists(best_path):
        try:
            x0 = np.load(best_path).astype(np.float64)
            return x0, sigma_hint, "best_coeffs.npy", True, "resume_ok"
        except Exception:
            pass

    if os.path.exists(mean_path):
        try:
            x0 = np.load(mean_path).astype(np.float64)
            return x0, sigma_hint, "cma_mean.npy", True, "resume_ok"
        except Exception:
            pass

    return dst_fallback("resume_files_missing")


def make_restart_plan(
    rid: int,
    center_x0: np.ndarray,
    dst_x0: np.ndarray,
    best_global_coeffs: Optional[np.ndarray],
    base_sigma0: float,
    base_popsize: int,
    coeff_bound: float,
    seed: int,
    restart_growth: float,
    restart_local_jitter: float,
    restart_global_jitter: float,
    restart_min_sigma: float,
    restart_max_sigma: float,
):
    """
    restart：
      rid=0  : initial
      rid=1  : local around best/current center
      rid=2  : global from DST center
      rid=3  : box/global random restart
      rid=4+ : repeat cycle
    """
    rng = np.random.default_rng(seed + 10007 * rid)

    if rid == 0:
        return (
            np.asarray(center_x0, dtype=np.float64).copy(),
            float(base_sigma0),
            int(base_popsize),
            "initial",
        )

    popsize_r = int(np.ceil(base_popsize * (restart_growth ** rid)))
    cycle_pos = (rid - 1) % 3

    if cycle_pos == 0:
        center = np.asarray(best_global_coeffs if best_global_coeffs is not None else center_x0, dtype=np.float64).copy()
        sigma_r = base_sigma0 * (0.85 ** rid)
        sigma_r = float(np.clip(sigma_r, restart_min_sigma, restart_max_sigma))
        x0_r = center + rng.normal(0.0, restart_local_jitter * sigma_r, size=center.shape)
        source = "local_restart"

    elif cycle_pos == 1:
        center = np.asarray(dst_x0, dtype=np.float64).copy()
        sigma_r = max(base_sigma0 * (1.10 ** (rid // 2)), 0.06)
        sigma_r = float(np.clip(sigma_r, restart_min_sigma, restart_max_sigma))
        x0_r = center + rng.normal(0.0, restart_global_jitter * sigma_r, size=center.shape)
        if coeff_bound and coeff_bound > 0:
            b = float(coeff_bound)
            mix = rng.uniform(-b, b, size=center.shape)
            x0_r = 0.70 * x0_r + 0.30 * mix
        source = "global_restart_dst"

    else:
        sigma_r = max(base_sigma0 * (1.25 ** (1 + rid // 3)), 0.06)
        sigma_r = float(np.clip(sigma_r, restart_min_sigma, restart_max_sigma))
        if coeff_bound and coeff_bound > 0:
            b = float(coeff_bound)
            x0_r = rng.uniform(-b, b, size=center_x0.shape)
        else:
            x0_r = rng.normal(0.0, restart_global_jitter * sigma_r, size=center_x0.shape)
        source = "global_restart_box"

    if coeff_bound and coeff_bound > 0:
        b = float(coeff_bound)
        x0_r = np.clip(x0_r, -b, b)

    return x0_r.astype(np.float64), float(sigma_r), int(popsize_r), source


def save_progress(
    out_dir: str,
    gen: int,
    best_reward: float,
    best_coeffs: Optional[np.ndarray],
    es,
    history: list,
    phys_main: dict,
    history_header: str,
    note: str = "",
    resume_fingerprint: str = "",
):
    os.makedirs(out_dir, exist_ok=True)

    if best_coeffs is not None:
        np.save(os.path.join(out_dir, "best_coeffs.npy"), np.asarray(best_coeffs, dtype=np.float64))
        best_pulse = pulse_from_coeffs(
            np.asarray(best_coeffs, dtype=np.float64),
            phys_main["sin_basis"],
            phys_main["params"]["u_final"],
            phys_main["params"]["A_max"],
        )
        np.save(os.path.join(out_dir, "best_pulse.npy"), best_pulse)
        np.savetxt(os.path.join(out_dir, "best_pulse.csv"), best_pulse, delimiter=",")

    np.save(os.path.join(out_dir, "cma_mean.npy"), np.asarray(es.mean, dtype=np.float64))

    state = {
        "gen": int(gen),
        "best_reward": float(best_reward),
        "sigma": float(es.sigma),
        "popsize": int(es.popsize),
        "mean_norm": float(np.linalg.norm(np.asarray(es.mean, dtype=np.float64))),
        "note": note,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resume_fingerprint": resume_fingerprint,
    }
    with open(os.path.join(out_dir, "cma_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    if len(history) > 0:
        hist_path = os.path.join(out_dir, "history.csv")
        np.savetxt(
            hist_path,
            np.array(history, dtype=float),
            delimiter=",",
            header=history_header,
            comments="",
        )


# =========================
# physics 
# =========================
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
    g = params["g"]

    t = params["t"]
    T = params["T"]
    n_coeffs = params["n_coeffs"]

    fluxonium = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=flux1)
    fluxonium2 = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=flux2)

    I_c = sp.eye(cutoff, format="csr")
    I2 = sp.eye(2, format="csr")

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
    c_mat = kron(
        I_c,
        sp.csr_matrix(np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=float))
    )
    c_ops = [Qobj(c_mat)]

    # Initial state uses flux=flux2 eigenstates
    _, states = fluxonium2.eigensys()
    g0 = Qobj(np.kron(states[:, 0], np.array([1.0, 0.0])).reshape(-1, 1))
    e0 = Qobj(np.kron(states[:, 1], np.array([1.0, 0.0])).reshape(-1, 1))
    rho0 = 0.5 * (g0 * g0.dag()) + 0.5 * (e0 * e0.dag())

    e_ops = (g0 * g0.dag())  # reward projector

    options = Options(
        nsteps=params["qutip_nsteps"],
        atol=params["qutip_atol"],
        rtol=params["qutip_rtol"],
        normalize_output=True,
    )

    if params.get("qutip_method", None) is not None:
        options.method = params["qutip_method"]

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


def evolve_with_pulse(pulse: np.ndarray) -> float:
    global _PHYS
    P = _PHYS

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

    result = mesolve(
        H, P["rho0"], t,
        P["c_ops"],
        e_ops=[P["e_ops"]],
        options=P["options"]
    )
    return float(result.expect[0][-1])


def _simulate_one(coeffs_np: np.ndarray) -> float:
    global _PHYS
    P = _PHYS
    pulse = pulse_from_coeffs(
        coeffs_np,
        P["sin_basis"],
        P["params"]["u_final"],
        P["params"]["A_max"]
    )
    return evolve_with_pulse(pulse)


def _simulate_one_safe(payload):
    coeffs_np, bad_reward = payload
    try:
        r = _simulate_one(coeffs_np)
        if not np.isfinite(r):
            return float(bad_reward), 0
        return float(r), 1
    except Exception:
        return float(bad_reward), 0


# =========================
# autotune helpers
# =========================
def build_regime_physics(base_phys_params: Dict[str, Any], regime_A_max: Optional[float]) -> Dict[str, Any]:
    params = dict(base_phys_params)
    params["A_max"] = regime_A_max
    return params


def build_auto_regimes(
    args,
    workers: int,
    n_coeffs: int,
    x0_dst: np.ndarray,
    resume_ok: bool,
    sigma_hint: Optional[float],
    current_A_max: Optional[float],
) -> List[Dict[str, Any]]:
    base_pop = resolve_popsize(args.popsize, n_coeffs, workers)
    base_sigma = resolve_sigma0(args.sigma0, x0_dst, args.coeff_bound)
    auto_bound = float(args.coeff_bound) if args.coeff_bound > 0 else 2.0
    auto_clip = float(args.clip_coeff) if args.clip_coeff > 0 else 2.0

    pop_small = max(base_pop, max(2 * workers, 16))
    pop_mid = max(pop_small, max(4 * workers, 32))
    pop_large = max(pop_mid, max(8 * workers, 64))
    if workers >= 4:
        pop_large = max(pop_large, 96)

    sig_ref = float(np.clip(base_sigma, 0.05, 0.12))
    limited_A = 1.0 if current_A_max is None else float(current_A_max)

    regimes: List[Dict[str, Any]] = []

    if resume_ok:
        regimes.append({
            "name": "resume_refine",
            "use_resume": True,
            "A_max": current_A_max,
            "popsize": pop_small,
            "sigma0": float(np.clip(sigma_hint if sigma_hint is not None else sig_ref, 0.03, 0.12)),
            "coeff_bound": auto_bound,
            "clip_coeff": auto_clip,
            "restarts": 2 if args.restarts < 0 else int(args.restarts),
            "restart_growth": 1.15,
            "restart_local_jitter": 0.10,
            "restart_global_jitter": 0.80,
            "restart_min_sigma": max(0.02, 0.20 * sig_ref),
            "restart_max_sigma": min(0.25, 1.40 * sig_ref),
        })

    regimes.append({
        "name": "balanced_unlimited",
        "use_resume": False,
        "A_max": None,
        "popsize": pop_mid,
        "sigma0": max(0.06, min(0.09, sig_ref)),
        "coeff_bound": auto_bound,
        "clip_coeff": auto_clip,
        "restarts": 3 if args.restarts < 0 else int(args.restarts),
        "restart_growth": 1.20,
        "restart_local_jitter": 0.12,
        "restart_global_jitter": 1.00,
        "restart_min_sigma": max(0.02, 0.20 * sig_ref),
        "restart_max_sigma": min(0.35, 1.60 * sig_ref),
    })

    regimes.append({
        "name": "large_unlimited",
        "use_resume": False,
        "A_max": None,
        "popsize": pop_large,
        "sigma0": max(0.06, min(0.10, sig_ref)),
        "coeff_bound": auto_bound,
        "clip_coeff": auto_clip,
        "restarts": 4 if args.restarts < 0 else int(args.restarts),
        "restart_growth": 1.25,
        "restart_local_jitter": 0.12,
        "restart_global_jitter": 1.20,
        "restart_min_sigma": max(0.02, 0.20 * sig_ref),
        "restart_max_sigma": min(0.45, 1.80 * sig_ref),
    })

    regimes.append({
        "name": "limited_boxed",
        "use_resume": False,
        "A_max": limited_A,
        "popsize": pop_mid,
        "sigma0": max(0.05, min(0.08, sig_ref)),
        "coeff_bound": auto_bound,
        "clip_coeff": auto_clip,
        "restarts": 3 if args.restarts < 0 else int(args.restarts),
        "restart_growth": 1.18,
        "restart_local_jitter": 0.10,
        "restart_global_jitter": 0.90,
        "restart_min_sigma": max(0.02, 0.20 * sig_ref),
        "restart_max_sigma": min(0.30, 1.50 * sig_ref),
    })

    dedup = []
    seen = set()
    for reg in regimes:
        key = (
            reg["use_resume"],
            -1.0 if reg["A_max"] is None else float(reg["A_max"]),
            int(reg["popsize"]),
            round(float(reg["sigma0"]), 6),
            round(float(reg["coeff_bound"]), 6),
            round(float(reg["clip_coeff"]), 6),
        )
        if key not in seen:
            seen.add(key)
            dedup.append(reg)
    return dedup


def build_manual_regime(args, workers: int, n_coeffs: int, x0: np.ndarray, current_A_max: Optional[float]) -> Dict[str, Any]:
    popsize = resolve_popsize(args.popsize, n_coeffs, workers)
    sigma0 = resolve_sigma0(args.sigma0, x0, args.coeff_bound)
    (
        resolved_restarts,
        resolved_restart_growth,
        resolved_restart_local_jitter,
        resolved_restart_global_jitter,
        resolved_restart_min_sigma,
        resolved_restart_max_sigma,
    ) = resolve_restart_params(
        user_restarts=args.restarts,
        user_restart_growth=args.restart_growth,
        user_restart_local_jitter=args.restart_local_jitter,
        user_restart_global_jitter=args.restart_global_jitter,
        user_restart_min_sigma=args.restart_min_sigma,
        user_restart_max_sigma=args.restart_max_sigma,
        base_sigma0=sigma0,
        workers=workers,
    )
    return {
        "name": "manual",
        "use_resume": True,
        "A_max": current_A_max,
        "popsize": int(popsize),
        "sigma0": float(sigma0),
        "coeff_bound": float(args.coeff_bound),
        "clip_coeff": float(args.clip_coeff),
        "restarts": int(resolved_restarts),
        "restart_growth": float(resolved_restart_growth),
        "restart_local_jitter": float(resolved_restart_local_jitter),
        "restart_global_jitter": float(resolved_restart_global_jitter),
        "restart_min_sigma": float(resolved_restart_min_sigma),
        "restart_max_sigma": float(resolved_restart_max_sigma),
    }


def score_scout_result(result: Dict[str, Any]) -> float:
    improvement = max(0.0, float(result["best_reward"]) - float(result["initial_reward"]))
    ok_frac = float(result.get("ok_frac_mean", 0.0))
    return float(result["best_reward"]) + 0.35 * improvement + 0.01 * ok_frac


def run_cma_campaign(
    regime: Dict[str, Any],
    base_phys_params: Dict[str, Any],
    x0_resume: np.ndarray,
    x0_dst: np.ndarray,
    workers: int,
    iters: int,
    target: float,
    patience: int,
    min_improve: float,
    max_minutes: float,
    bad_reward: float,
    save_every: int,
    maxtasksperchild: int,
    out_dir: Optional[str],
    save_outputs: bool,
    start_time: float,
    seed: int,
    scout_mode: bool,
    x0_override: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    regime_phys_params = build_regime_physics(base_phys_params, regime["A_max"])
    resume_fingerprint = build_resume_fingerprint(regime_phys_params)

    global _PHYS
    _PHYS = init_physics(regime_phys_params)

    if x0_override is not None:
        base_x0 = np.asarray(x0_override, dtype=np.float64).copy()
    else:
        base_x0 = np.asarray(
            x0_resume if regime.get("use_resume", False) else x0_dst,
            dtype=np.float64
        ).copy()
    dst_x0_local = np.asarray(x0_dst, dtype=np.float64).copy()

    pool = None
    best_global_reward = -1e9
    best_global_coeffs = None
    continue_coeffs = None
    continue_sigma = None
    continue_tol = 3e-4   # 1e-4 ~ 5e-4
    global_history = []
    restart_table = []
    target_reached = False
    stop_reason_global = "max_iters"

    try:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(regime_phys_params,),
            maxtasksperchild=int(maxtasksperchild),
        )

        total_runs = int(regime["restarts"]) + 1
        if save_outputs:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "selected_regime.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "regime": regime,
                    "resume_fingerprint": resume_fingerprint,
                    "scout_mode": bool(scout_mode),
                }, f, indent=2)

        print("=" * 96)
        print(
            f"regime={regime['name']} | A_max={'off' if regime['A_max'] is None else regime['A_max']} | "
            f"pop={regime['popsize']} | sigma0={regime['sigma0']:.4f} | restarts={regime['restarts']} | "
            f"resume={regime.get('use_resume', False)} | mode={'scout' if scout_mode else 'full'}"
        )
        print("=" * 96)

        for rid in range(total_runs):
            if max_minutes > 0:
                elapsed_min = (time.time() - start_time) / 60.0
                if elapsed_min >= float(max_minutes):
                    stop_reason_global = "max_minutes"
                    print(f"[STOP] reached max_minutes={max_minutes} before restart {rid}")
                    break

            if continue_coeffs is not None:
                x0_r = continue_coeffs.copy()
                sigma_r = float(continue_sigma)
                popsize_r = int(regime["popsize"])
                restart_source = "continue_near_best"
                continue_coeffs = None
                continue_sigma = None
            else:
                x0_r, sigma_r, popsize_r, restart_source = make_restart_plan(
                    rid=rid,
                    center_x0=base_x0,
                    dst_x0=dst_x0_local,
                    best_global_coeffs=best_global_coeffs,
                    base_sigma0=float(regime["sigma0"]),
                    base_popsize=int(regime["popsize"]),
                    coeff_bound=float(regime["coeff_bound"]),
                    seed=int(seed),
                    restart_growth=float(regime["restart_growth"]),
                    restart_local_jitter=float(regime["restart_local_jitter"]),
                    restart_global_jitter=float(regime["restart_global_jitter"]),
                    restart_min_sigma=float(regime["restart_min_sigma"]),
                    restart_max_sigma=float(regime["restart_max_sigma"]),
                )

            restart_dir = None
            if save_outputs and out_dir is not None:
                restart_dir = os.path.join(out_dir, f"restart_{rid:02d}")
                os.makedirs(restart_dir, exist_ok=True)

            cma_opts_r = {
                "seed": int(seed + 1000 * rid),
                "verbose": -9,
                "popsize": int(popsize_r),
            }
            if regime["coeff_bound"] and regime["coeff_bound"] > 0:
                b = float(regime["coeff_bound"])
                cma_opts_r["bounds"] = [-b, b]

            es = cma.CMAEvolutionStrategy(
                np.asarray(x0_r, dtype=np.float64),
                float(sigma_r),
                cma_opts_r
            )

            try:
                r0 = float(_simulate_one(np.asarray(x0_r, dtype=np.float64)))
                ok0 = 1.0 if np.isfinite(r0) else 0.0
                if not np.isfinite(r0):
                    r0 = float(bad_reward)
            except Exception:
                r0 = float(bad_reward)
                ok0 = 0.0

            local_best_reward = r0
            local_best_coeffs = np.asarray(x0_r, dtype=np.float64).copy()
            local_history = [[-1, local_best_reward, r0, r0, 0.0, float(es.sigma), int(es.popsize), ok0]]

            if r0 > best_global_reward:
                best_global_reward = r0
                best_global_coeffs = np.asarray(x0_r, dtype=np.float64).copy()

            global_history.append([
                rid, -1, best_global_reward, r0, r0, 0.0, float(es.sigma), int(es.popsize), ok0
            ])

            best_so_far_local = local_best_reward
            no_improve = 0
            conv_cnt = 0
            stop_reason = "max_iters"

            hopeless_warmup = 80        # give it at least 80 generations before judging it as hopeless
            hopeless_window = 20        # estimate the improvement rate from the most recent 20 generations
            hopeless_margin = 5e-4      # even if it runs to the end, it should at least have a chance to get within global best - 0.0005
            hopeless_min_slope = 1e-6   # if the recent improvement rate is almost zero, treat it as essentially not improving

            print(
                f"\n=== Restart {rid:02d}/{total_runs - 1:02d} | source={restart_source} | "
                f"sigma0={sigma_r:.4f} | pop={popsize_r} ==="
            )

            for gen in range(int(iters)):
                if max_minutes > 0:
                    elapsed_min = (time.time() - start_time) / 60.0
                    if elapsed_min >= float(max_minutes):
                        stop_reason = "max_minutes"
                        stop_reason_global = "max_minutes"
                        print(f"[STOP] reached max_minutes={max_minutes}")
                        break

                X = es.ask()

                cand = []
                for x_candidate in X:
                    x_candidate = np.asarray(x_candidate, dtype=np.float64)
                    if regime["clip_coeff"] and regime["clip_coeff"] > 0:
                        c = float(regime["clip_coeff"])
                        x_candidate = np.clip(x_candidate, -c, c)
                    cand.append(x_candidate)

                payloads = [(x_candidate, float(bad_reward)) for x_candidate in cand]
                evals = pool.map(_simulate_one_safe, payloads)

                rewards = [float(r) for r, _ in evals]
                oks = [int(ok) for _, ok in evals]

                losses = [1.0 - float(r) for r in rewards]
                es.tell(X, losses)

                r_best = float(np.max(rewards))
                r_mean = float(np.mean(rewards))
                r_std = float(np.std(rewards))
                ok_frac = float(np.mean(oks))
                sigma_now = float(es.sigma)

                idx = int(np.argmax(rewards))
                if r_best > local_best_reward:
                    local_best_reward = r_best
                    local_best_coeffs = cand[idx].copy()

                if r_best > best_global_reward:
                    best_global_reward = r_best
                    best_global_coeffs = cand[idx].copy()

                local_history.append([
                    gen, local_best_reward, r_best, r_mean, r_std, sigma_now, len(X), ok_frac
                ])
                global_history.append([
                    rid, gen, best_global_reward, r_best, r_mean, r_std, sigma_now, len(X), ok_frac
                ])

                print(
                    f"R{rid:02d} G{gen:03d} | globalBest={best_global_reward:.6f} | "
                    f"localBest={local_best_reward:.6f} | bestGen={r_best:.6f} | "
                    f"mean={r_mean:.6f} | std={r_std:.6f} | sigma={sigma_now:.4f} | "
                    f"pop={len(X)} | ok={ok_frac:.2f}"
                )

                if save_outputs and restart_dir is not None and out_dir is not None:
                    if (gen + 1) % max(1, int(save_every)) == 0:
                        save_progress(
                            restart_dir, gen, local_best_reward, local_best_coeffs, es,
                            local_history, _PHYS,
                            history_header="gen,best_ever,best_gen,mean_gen,std_gen,sigma,popsize,ok_frac",
                            note=f"restart {rid} periodic save",
                            resume_fingerprint=resume_fingerprint,
                        )
                        save_progress(
                            out_dir, gen, best_global_reward, best_global_coeffs, es,
                            global_history, _PHYS,
                            history_header="restart,gen,best_ever,best_gen,mean_gen,std_gen,sigma,popsize,ok_frac",
                            note=f"global periodic save after restart {rid}",
                            resume_fingerprint=resume_fingerprint,
                        )

                if best_global_reward >= target:
                    stop_reason = "target"
                    stop_reason_global = "target"
                    print(f"[STOP] reached target >= {target}")
                    target_reached = True
                    break

                if local_best_reward > best_so_far_local + min_improve:
                    best_so_far_local = local_best_reward
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        stop_reason = "plateau_restart"
                        print(f"[RESTART] no local improvement for {patience} generations")
                        break

                # hopeless pruning: although it is still improving, based on the recent trend it is very unlikely to catch up with the global best
                if (not scout_mode) and (gen >= hopeless_warmup) and (len(local_history) >= hopeless_window + 1):
                    prev_local_best = float(local_history[-(hopeless_window + 1)][1])   # local best from 20 generations ago
                    recent_gain = float(local_best_reward - prev_local_best)
                    recent_slope = max(0.0, recent_gain / hopeless_window)

                    # if the recent slope is very small, treat it as essentially not improving to avoid false hope
                    if recent_slope < hopeless_min_slope:
                        recent_slope = 0.0

                    remaining_gens = int(iters) - 1 - gen
                    projected_final = float(local_best_reward + recent_slope * remaining_gens)

                    # if the current run is still behind the global best, then check whether it is hopeless
                    if local_best_reward < best_global_reward:
                        required_to_be_promising = float(best_global_reward - hopeless_margin)

                        if projected_final < required_to_be_promising:
                            stop_reason = "hopeless_restart"
                            print(
                                f"[PRUNE] hopeless: local={local_best_reward:.6f}, "
                                f"global={best_global_reward:.6f}, slope={recent_slope:.3e}, "
                                f"projected={projected_final:.6f} < need={required_to_be_promising:.6f}"
                            )
                            break

                sigma_min = 0.004
                std_min = 2e-5
                mean_gap = 5e-5
                conv_patience = 6 if scout_mode else 10

                if (sigma_now < sigma_min) and (r_std < std_min) and ((local_best_reward - r_mean) < mean_gap):
                    conv_cnt += 1
                else:
                    conv_cnt = 0

                if conv_cnt >= conv_patience:
                    stop_reason = "converged_restart"
                    print(
                        f"[RESTART] converged: sigma={sigma_now:.4g}, std={r_std:.4g}, "
                        f"mean_gap={(local_best_reward - r_mean):.3g}"
                    )
                    break

            if local_best_coeffs is None:
                local_best_coeffs = np.asarray(x0_r, dtype=np.float64).copy()
                local_best_reward = float(bad_reward)

            if save_outputs and restart_dir is not None and out_dir is not None:
                save_progress(
                    restart_dir,
                    max(0, len(local_history) - 1),
                    local_best_reward,
                    local_best_coeffs,
                    es,
                    local_history,
                    _PHYS,
                    history_header="gen,best_ever,best_gen,mean_gen,std_gen,sigma,popsize,ok_frac",
                    note=f"restart {rid} final save ({stop_reason})",
                    resume_fingerprint=resume_fingerprint,
                )
                save_progress(
                    out_dir,
                    max(0, len(global_history) - 1),
                    best_global_reward,
                    best_global_coeffs,
                    es,
                    global_history,
                    _PHYS,
                    history_header="restart,gen,best_ever,best_gen,mean_gen,std_gen,sigma,popsize,ok_frac",
                    note=f"global final save after restart {rid} ({stop_reason})",
                    resume_fingerprint=resume_fingerprint,
                )

            restart_table.append([
                rid,
                popsize_r,
                sigma_r,
                local_best_reward,
                best_global_reward,
                len(local_history),
                1.0 if stop_reason == "target" else 0.0,
            ])

            gap_to_global = best_global_reward - local_best_reward

            should_continue = (
                stop_reason == "max_iters"
                and local_best_coeffs is not None
                and gap_to_global > 0.0
                and gap_to_global <= continue_tol
            )

            if should_continue and (rid + 1 < total_runs):
                continue_coeffs = local_best_coeffs.copy()
                continue_sigma = max(
                    float(regime["restart_min_sigma"]),
                    min(float(regime["restart_max_sigma"]), 0.7 * float(es.sigma))
                )
                print(
                    f"[CONTINUE] local best {local_best_reward:.6f} is close to global best "
                    f"{best_global_reward:.6f}; continue with sigma={continue_sigma:.4f}"
                )

            if save_outputs and out_dir is not None:
                np.savetxt(
                    os.path.join(out_dir, "restart_summary.csv"),
                    np.array(restart_table, dtype=float),
                    delimiter=",",
                    header="restart,popsize,sigma0,local_best,global_best,gens_done,target_hit",
                    comments="",
                )

            if target_reached:
                break

        if best_global_coeffs is None:
            try:
                best_global_reward = float(_simulate_one(np.asarray(base_x0, dtype=np.float64)))
            except Exception:
                best_global_reward = float(bad_reward)
            best_global_coeffs = np.asarray(base_x0, dtype=np.float64).copy()

        if save_outputs and out_dir is not None:
            summary = {
                "best_reward": float(best_global_reward),
                "workers": int(workers),
                "base_popsize": int(regime["popsize"]),
                "base_sigma0": float(regime["sigma0"]),
                "restarts_done": int(len(restart_table)),
                "seed": int(seed),
                "elapsed_minutes": float((time.time() - start_time) / 60.0),
                "selected_regime": regime,
                "resume_fingerprint": resume_fingerprint,
                "stop_reason": stop_reason_global,
            }
            with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        ok_frac_mean = 0.0
        if len(global_history) > 0:
            ok_frac_mean = float(np.mean([row[-1] for row in global_history]))

        result = {
            "best_reward": float(best_global_reward),
            "best_coeffs": np.asarray(best_global_coeffs, dtype=np.float64),
            "initial_reward": float(global_history[0][3]) if len(global_history) > 0 else float(bad_reward),
            "ok_frac_mean": float(ok_frac_mean),
            "restarts_done": int(len(restart_table)),
            "history_len": int(len(global_history)),
            "resume_fingerprint": resume_fingerprint,
            "stop_reason": stop_reason_global,
            "regime": regime,
        }
        return result

    finally:
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception:
                pass


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # CMA-ES
    parser.add_argument("--iters", type=int, default=800,
                        help="Number of CMA-ES generations for the final selected regime.")
    parser.add_argument("--popsize", type=int, default=0,
                        help="Population size. 0 = auto.")
    parser.add_argument("--sigma0", type=float, default=0.0,
                        help="Initial global step-size. <=0 = auto.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--clip_coeff", type=float, default=0.0,
                        help="If >0, clip sampled coefficients before simulation.")
    parser.add_argument("--coeff_bound", type=float, default=0.0,
                        help="If >0, set CMA-ES box constraints.")

    # Parallel + output
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker processes. 0 = auto.")
    parser.add_argument("--out_dir", type=str, default="out_autotune_cmaes",
                        help="Output directory.")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Auto-save every N generations.")
    parser.add_argument("--no_resume", action="store_true",
                        help="Do not load previous best_coeffs / cma_mean from out_dir.")
    parser.add_argument("--allow_legacy_resume", action="store_true",
                        help="Allow resume from old out_dir without fingerprint metadata.")

    # Time grid / pulse
    parser.add_argument("--T", type=float, default=150.0,
                        help="Total pulse duration.")
    parser.add_argument("--nt", type=int, default=200,
                        help="Number of time points.")
    parser.add_argument("--A_max", type=float, default=1.0,
                        help="Amplitude limiter via tanh; set <0 to disable. In auto mode this is only one candidate.")

    # QuTiP
    parser.add_argument("--qutip_nsteps", type=int, default=20000,
                        help="Max internal ODE steps per interval.")
    parser.add_argument("--qutip_atol", type=float, default=1e-8,
                        help="Absolute tolerance for QuTiP solver.")
    parser.add_argument("--qutip_rtol", type=float, default=1e-6,
                        help="Relative tolerance for QuTiP solver.")
    parser.add_argument("--qutip_method", type=str, default="",
                        help="Optional QuTiP ODE method, e.g. 'adams' or 'bdf'.")

    # Early stopping
    parser.add_argument("--target", type=float, default=0.999,
                        help="Stop when best_reward >= target.")
    parser.add_argument("--patience", type=int, default=80,
                        help="Stop after this many generations without meaningful improvement.")
    parser.add_argument("--min_improve", type=float, default=1e-6,
                        help="Improvement threshold to reset patience counter.")

    # Auto stop / robustness
    parser.add_argument("--max_minutes", type=float, default=-1,
                        help="<=0 disables wall-time limit. Default 180 min.")
    parser.add_argument("--bad_reward", type=float, default=0.0,
                        help="Reward assigned when a candidate simulation fails.")
    parser.add_argument("--maxtasksperchild", type=int, default=10,
                        help="Pool maxtasksperchild.")

    # Multi-restart CMA-ES
    parser.add_argument("--restarts", type=int, default=-1,
                        help="Number of extra restarts. <0 = auto.")
    parser.add_argument("--restart_growth", type=float, default=0.0,
                        help="Population growth factor across restarts. <=0 = auto.")
    parser.add_argument("--restart_local_jitter", type=float, default=0.0,
                        help="Local restart center jitter scale. <=0 = auto.")
    parser.add_argument("--restart_global_jitter", type=float, default=0.0,
                        help="Global restart jitter scale. <=0 = auto.")
    parser.add_argument("--restart_min_sigma", type=float, default=0.0,
                        help="Lower clamp for restart sigma. <=0 = auto.")
    parser.add_argument("--restart_max_sigma", type=float, default=0.0,
                        help="Upper clamp for restart sigma. <=0 = auto.")

    # New automatic regime selection
    parser.add_argument("--no_auto_regime", action="store_true",
                        help="Disable scouting and run a single manual regime built from the CLI arguments.")
    parser.add_argument("--scout_gens", type=int, default=6,
                        help="Short scouting generations per regime before selecting the final regime.")

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

    current_A_max = None if args.A_max < 0 else float(args.A_max)

    init_pulse = make_default_init_pulse(t, phi0)
    x0_dst = np.real(dst(init_pulse - u_final) / init_pulse.shape[0])[:n_coeffs].astype(np.float64)

    workers = resolve_workers(args.workers)

    # build base physics once for fingerprint and regime cloning
    base_phys_params = dict(
        EJ=EJ, EC=EC, EL=EL, cutoff=cutoff,
        flux1=flux1, flux2=flux2,
        phi0=phi0, u_final=u_final,
        Omega=Omega, omega_q=omega_q, omega_ef=omega_ef, omega_gf=omega_gf, omega_r=omega_r,
        n_ef=n_ef, n_gf=n_gf, g=g,
        n_coeffs=n_coeffs, T=T, t=t,
        A_max=current_A_max,
        qutip_nsteps=int(args.qutip_nsteps),
        qutip_atol=float(args.qutip_atol),
        qutip_rtol=float(args.qutip_rtol),
    )
    if args.qutip_method.strip():
        base_phys_params["qutip_method"] = args.qutip_method.strip()

    base_resume_fingerprint = build_resume_fingerprint(base_phys_params)

    x0_resume, sigma_hint, x0_source, resume_ok, resume_note = load_warm_start(
        args.out_dir,
        init_pulse,
        u_final,
        n_coeffs,
        args.no_resume,
        base_resume_fingerprint,
        args.allow_legacy_resume,
    )

    print("=" * 96)
    print(f"workers            : {workers}")
    print(f"x0 source          : {x0_source}")
    print(f"resume_ok          : {resume_ok}")
    print(f"resume_note        : {resume_note}")
    print(f"base A_max         : {'off' if current_A_max is None else current_A_max}")
    print(f"out_dir            : {args.out_dir}")
    print("=" * 96)

    start = time.time()

    if args.no_auto_regime:
        selected_regime = build_manual_regime(args, workers, n_coeffs, x0_resume, current_A_max)
        scout_rows = []
        selected_scout_best_coeffs = None
    else:
        regimes = build_auto_regimes(
            args=args,
            workers=workers,
            n_coeffs=n_coeffs,
            x0_dst=x0_dst,
            resume_ok=resume_ok,
            sigma_hint=sigma_hint,
            current_A_max=current_A_max,
        )

        scout_rows = []
        selected_regime = None
        selected_scout_best_coeffs = None
        selected_scout_best_reward = None
        best_score = -1e18

        scout_patience = max(4, min(int(args.patience), max(4, int(args.scout_gens) // 2 + 1)))

        print("\nStarting automatic regime scouting...")
        for idx, regime in enumerate(regimes):
            scout_result = run_cma_campaign(
                regime=regime,
                base_phys_params=base_phys_params,
                x0_resume=x0_resume,
                x0_dst=x0_dst,
                workers=workers,
                iters=int(args.scout_gens),
                target=float(args.target),
                patience=scout_patience,
                min_improve=float(args.min_improve),
                max_minutes=float(args.max_minutes),
                bad_reward=float(args.bad_reward),
                save_every=max(1, int(args.save_every)),
                maxtasksperchild=int(args.maxtasksperchild),
                out_dir=None,
                save_outputs=False,
                start_time=start,
                seed=int(args.seed + 10000 * idx),
                scout_mode=True,
            )
            score = score_scout_result(scout_result)
            scout_rows.append({
                "name": regime["name"],
                "score": score,
                "initial_reward": float(scout_result["initial_reward"]),
                "best_reward": float(scout_result["best_reward"]),
                "ok_frac_mean": float(scout_result["ok_frac_mean"]),
                "A_max": "off" if regime["A_max"] is None else float(regime["A_max"]),
                "popsize": int(regime["popsize"]),
                "sigma0": float(regime["sigma0"]),
                "use_resume": bool(regime.get("use_resume", False)),
            })
            print(
                f"[SCOUT] {regime['name']}: initial={scout_result['initial_reward']:.6f}, "
                f"best={scout_result['best_reward']:.6f}, ok={scout_result['ok_frac_mean']:.2f}, score={score:.6f}"
            )
            if score > best_score:
                best_score = score
                selected_regime = dict(regime)
                selected_scout_best_coeffs = np.asarray(scout_result["best_coeffs"], dtype=np.float64).copy()
                selected_scout_best_reward = float(scout_result["best_reward"])

        if selected_regime is None:
            selected_regime = build_manual_regime(args, workers, n_coeffs, x0_resume, current_A_max)

        with open(os.path.join(args.out_dir, "scout_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "resume_ok": resume_ok,
                "resume_note": resume_note,
                "scout_rows": scout_rows,
                "selected_regime": selected_regime,
            }, f, indent=2)

        if len(scout_rows) > 0:
            scout_csv = []
            for row in scout_rows:
                scout_csv.append([
                    float(row["score"]),
                    float(row["initial_reward"]),
                    float(row["best_reward"]),
                    float(row["ok_frac_mean"]),
                    float(-1.0 if row["A_max"] == "off" else row["A_max"]),
                    float(row["popsize"]),
                    float(row["sigma0"]),
                    1.0 if row["use_resume"] else 0.0,
                ])
            np.savetxt(
                os.path.join(args.out_dir, "scout_summary.csv"),
                np.array(scout_csv, dtype=float),
                delimiter=",",
                header="score,initial_reward,best_reward,ok_frac_mean,A_max,popsize,sigma0,use_resume",
                comments="",
            )

    print("\nSelected regime:")
    print(json.dumps(selected_regime, indent=2))

    final_result = run_cma_campaign(
        regime=selected_regime,
        base_phys_params=base_phys_params,
        x0_resume=x0_resume,
        x0_dst=x0_dst,
        workers=workers,
        iters=int(args.iters),
        target=float(args.target),
        patience=int(args.patience),
        min_improve=float(args.min_improve),
        max_minutes=float(args.max_minutes),
        bad_reward=float(args.bad_reward),
        save_every=max(1, int(args.save_every)),
        maxtasksperchild=int(args.maxtasksperchild),
        out_dir=args.out_dir,
        save_outputs=True,
        start_time=start,
        seed=int(args.seed),
        scout_mode=False,
        x0_override=selected_scout_best_coeffs,
    )

    print(f"\nDONE. Best reward={final_result['best_reward']:.6f}")
    print(f"Saved to: {args.out_dir}")
    print(f"Total minutes: {(time.time() - start)/60:.2f}")


if __name__ == "__main__":
    main()
