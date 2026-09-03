from pathlib import Path
import sys
import os

# --- 0. Threading Control (CRITICAL for Multiprocessing) ---
# Prevent NumPy/OpenBLAS from spawning too many internal threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# --- Path Setup ---
script_dir = Path(__file__).absolute().parent
root_dir = script_dir.parent
sys.path.append(str(root_dir / "source"))

# --- Modular Imports ---
import utils
import time
import simulator
import multiprocessing as mp
# Toggle for heavy imports
SCI_MODE = True 
if SCI_MODE:
    import numpy as np
    import cma


# --- Global worker variables ---
_SIM = None

def init_worker(physics_config):
    """Initializes the simulator once per worker process."""
    global _SIM
    import simulator
    _SIM = simulator.FluxoniumSimulator(physics_config)

def eval_candidate(coeffs):
    """Worker function to evaluate a single candidate."""
    return _SIM.get_fidelity(coeffs)

def save_checkpoint(out_dir, best_reward, best_coeffs, es, history, sim):
    """Helper to save intermediate results."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if best_coeffs is not None:
        np.save(out_path / "best_coeffs.npy", best_coeffs)
        best_pulse = sim.build_pulse(best_coeffs)
        np.save(out_path / "best_pulse.npy", best_pulse)

    np.save(out_path / "cma_mean.npy", np.asarray(es.mean, dtype=np.float64))
    
    with open(out_path / "cma_state.txt", "w", encoding="utf-8") as f:
        f.write(f"best_reward={best_reward}\n")
        f.write(f"sigma={es.sigma}\n")
        f.write(f"popsize={es.popsize}\n")

    hist_path = out_path / "history.csv"
    np.savetxt(
        hist_path,
        np.array(history, dtype=float),
        delimiter=",",
        header="gen,best_ever,best_gen,mean_gen,std_gen,sigma,popsize",
        comments="",
    )

def main():
    # 1. Load configuration
    physics_config, model_config = utils.get_config("cmaes", root_dir)

    # 2. Get CLI overrides
    parser = utils.get_parser()
    args = parser.parse_args()
    
    # 3. Apply overrides
    physics_config = utils.update_config_with_args(physics_config, args)
    model_config = utils.update_config_with_args(model_config, args)
    
    # 4. Set seed
    if model_config.get("seed") is not None:
        np.random.seed(model_config["seed"])

    print(f"--- Configuration Loaded ---")
    print(f"Target Fidelity: {model_config['target']}")
    print(f"Time set: {physics_config['T']} ns! :D")
    print(f"Popsize: {model_config.get('popsize', 'Default')}")
    print(f"Workers: {model_config.get('workers', 1)}")

    # 5. Initialize Simulator (for pulse building/saving)
    sim = simulator.FluxoniumSimulator(physics_config)

    # 6. Ensure output directory exists
    out_dir = Path(model_config.get("out_dir", "data/cmaes"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 7. Setup Initial Guess
    best_path = out_dir / "best_coeffs.npy"
    x0 = utils.get_initial_guess(physics_config, best_path=best_path)
    print(f"Initial coefficients ready. Length: {len(x0)}")

    # 8. CMA-ES Options
    cma_opts = {"seed": model_config["seed"]}
    if model_config.get("popsize") and model_config["popsize"] > 0:
        cma_opts["popsize"] = model_config["popsize"]
    if float(model_config.get("coeff_bound", 0)) > 0:
        b = float(model_config["coeff_bound"])
        cma_opts["bounds"] = [-b, b]

    es = cma.CMAEvolutionStrategy(x0, float(model_config["sigma0"]), cma_opts)

    # 9. Multiprocessing Setup
    n_workers = int(model_config.get("workers", 1))
    pool = None
    if n_workers > 1:
        print(f"Starting pool with {n_workers} workers...")
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=n_workers, initializer=init_worker, initargs=(physics_config,))

    best_reward = -1e9
    best_coeffs = None
    history = []

    # Early stop parameters
    target = float(model_config["target"])
    patience = int(model_config["patience"])
    min_improve = float(model_config["min_improve"])
    best_so_far = best_reward
    no_improve = 0

    # Convergence parameters
    
    sigma_min = model_config["sigma_min"]
    std_min = model_config["std_min"]
    mean_gap = model_config["mean_gap"]
    conv_patience = model_config["conv_patience"]
    conv_cnt = model_config["conv_cnt"]
    

    start = time.time()
    
    try:
        for gen in range(model_config["iters"]):
            X = es.ask()
            candidates = []
            for x in X:
                x = np.asarray(x, dtype=np.float64)
                if model_config.get("clip_coeff", 0) > 0:
                    c = float(model_config["clip_coeff"])
                    x = np.clip(x, -c, c)
                candidates.append(x)

            # --- Evaluation Phase ---
            if pool:
                rewards = pool.map(eval_candidate, candidates)
            else:
                rewards = [sim.get_fidelity(c) for c in candidates]
            
            losses = [1.0 - r for r in rewards]
            es.tell(X, losses)

            # Bookkeeping
            r_best = float(np.max(rewards))
            r_mean = float(np.mean(rewards))
            r_std = float(np.std(rewards))
            sigma = float(es.sigma)

            new_best_found = False
            if r_best > best_reward:
                best_reward = r_best
                idx = int(np.argmax(rewards))
                best_coeffs = candidates[idx].copy()
                new_best_found = True

            history.append([gen, best_reward, r_best, r_mean, r_std, sigma, len(X)])

            print(f"Gen {gen:03d} | bestEver={best_reward:.6f} | bestGen={r_best:.6f} | sigma={sigma:.4f}")

            # --- Intermediate Save ---
            # Save every generation so we don't lose data on timeout!
            save_checkpoint(out_dir, best_reward, best_coeffs, es, history, sim)

            # Early stop logic
            if best_reward >= target:
                print(f"[STOP] reached >= {target}")
                break
            if best_reward > best_so_far + min_improve:
                best_so_far, no_improve = best_reward, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[STOP] no improvement for {patience} generations")
                    break

            # Convergence logic
            if (sigma < sigma_min) and (r_std < std_min) and ((best_reward - r_mean) < mean_gap):
                conv_cnt += 1
            else:
                conv_cnt = 0
            if conv_cnt >= conv_patience:
                print(f"[STOP] converged")
                break

        print(f"\nDONE. Best reward={best_reward:.6f}")
        print(f"Total minutes: {(time.time() - start)/60:.2f}")

    finally:
        if pool:
            pool.close()
            pool.join()
        print("I love pizza :3")

if __name__ == "__main__":
    main()
