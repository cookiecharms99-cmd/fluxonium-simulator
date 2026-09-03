import json
import os
import argparse
import numpy as np
from pathlib import Path

import datetime




def load_json(filepath):
    """Helper to load a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)

def get_config(model_name, root_dir):
    """
    Merges project-wide params.json with model-specific config.
    Returns a dictionary of parameters.
    """
    params_path = root_dir / "params.json"
    model_path = root_dir / "model_config" / f"{model_name}_config.json"
    
    physics_config = load_json(params_path)
    model_config = {}
    if model_path.exists():
        model_config = load_json(model_path)
    
    # Initialize time array if T and num_t_steps are present
    if "T" in physics_config and "num_t_steps" in physics_config:
        physics_config["t"] = np.linspace(0.0, physics_config["T"], physics_config["num_t_steps"])
        
    return physics_config, model_config

def get_parser():
    """Defines and returns the standard argument parser for optimization runs."""
    parser = argparse.ArgumentParser(description="Fluxonium Optimization Run")

    # CMA-ES / Optimizer Specifics
    parser.add_argument("--iters", type=int, default=None, help="Max iterations/generations.")
    parser.add_argument("--popsize", type=int, default=None, help="Population size.")
    parser.add_argument("--sigma0", type=float, default=None, help="Initial step size.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--clip_coeff", type=float, default=None, help="Hard clamp for coeffs.")
    parser.add_argument("--coeff_bound", type=float, default=None, help="Box constraints for optimizer.")

    # Execution Environment
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker count.")
    parser.add_argument("--out_dir", type=str, default=None, help="Where to save results.")

    # Time / Physics Overrides
    parser.add_argument("--T", type=float, default=None, help="Override pulse duration.")
    parser.add_argument("--num_t_steps", type=int, default=None, help="Override time steps.")
    parser.add_argument("--A_max", type=float, default=None, help="Override amplitude limit.")

    # QuTiP Tuning
    parser.add_argument("--qutip_nsteps", type=int, default=None, help="Max ODE steps.")
    parser.add_argument("--qutip_atol", type=float, default=None, help="Absolute tolerance.")
    parser.add_argument("--qutip_rtol", type=float, default=None, help="Relative tolerance.")
    parser.add_argument("--qutip_method", type=str, default=None, help="ODE solver method.")

    # Stopping Criteria
    parser.add_argument("--target", type=float, default=None, help="Fidelity target.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--min_improve", type=float, default=None, help="Min improvement threshold.")

    return parser

def update_config_with_args(config, args):
    """Updates config dict with any non-None arguments from the parser."""
    for key, value in vars(args).items():
        if value is not None:
            if config.get(key, "not_present") != "not_present":
                config[key] = value
    
    # Re-calculate time array if T or num_t_steps was updated via args if we're dealing with
    # physics config
    if "T" in config and "num_t_steps" in config:
        # Handle the case where 'nt' in args corresponds to 'num_t_steps' in config
        if hasattr(args, 'num_t_steps') and args.num_t_steps is not None:
            config["num_t_steps"] = args.num_t_steps
        config["t"] = np.linspace(0.0, config["T"], config["num_t_steps"])
            
    return config

def get_initial_guess(config, best_path=None):

    # Try to import DST for the initial guess, fallback to a dummy if not available
    try:
        from scipy.fft import dst
    except ImportError:
        try:
            from scipy.fftpack import dst
        except ImportError:
            dst = None




    """
    Returns initial coefficients. 
    Loads from best_path if it exists, otherwise generates from a piecewise guess.
    """
    n_pulse = config.get("n_pulse", config.get("n_coeffs", 40))
    n_drive = config.get("n_drive", 0)
    total_len = n_pulse + n_drive

    if best_path and os.path.exists(best_path):
        x = np.load(best_path).astype(np.float64)
        if len(x) == total_len:
            return x
        elif len(x) < total_len:
            # Pad with zeros (e.g., if loading old 40-len flux-only coeffs)
            print(f"Warning: Loaded coeffs length {len(x)} < {total_len}. Padding with zeros.")
            return np.concatenate([x, np.zeros(total_len - len(x))])
        else:
            # Truncate or handle mismatch
            return x[:total_len]

    if dst is None:
        return np.zeros(total_len)

    # Piecewise guess logic: scaled by T
    t = config["t"]
    T = config["T"]
    phi0 = np.pi / 2
    u_final = config.get("u_final", np.pi)
    
    init_pulse = np.zeros_like(t)
    for i, time_t in enumerate(t):
        if time_t < T * (10/150):
            init_pulse[i] = 2 * phi0 - phi0 * time_t / (T * (10/150))
        elif time_t < T * (110/150):
            init_pulse[i] = phi0
        else:
            init_pulse[i] = phi0 + phi0 * (time_t - T * (110/150)) / (T * (40/150))

    # Convert flux pulse shape to sine basis
    sin_fft = dst(init_pulse - u_final) / len(t)
    flux_coeffs = np.real(sin_fft[:n_pulse]).astype(np.float64)
    
    # Concatenate with zeros for drive
    return np.concatenate([flux_coeffs, np.zeros(n_drive)])



def log_data_to_global(run_name,model_name, params, weights, fidelity, file_path = "global_run_data.csv"):
    """Append summary of best path found!"""
    import pandas as pd
    # Parameter fingerprint

    new_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "runname" : run_name,
        "fidelity": fidelity,
        "T": params["T"],
        "n_pulse": params["n_pulse"],
        "n_drive": params.get("n_drive", 0),
        "weights_path": f"data/weights/{run_name}_weights.npy"
    }


    