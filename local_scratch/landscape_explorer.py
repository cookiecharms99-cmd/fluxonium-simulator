import sys
import os
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- Path Setup ---
script_dir = Path(__file__).absolute().parent
root_dir = script_dir.parent
sys.path.append(str(root_dir / "source"))

import simulator
import utils

def main():
    # 1. Setup
    physics_config, model_config = utils.get_config("cmaes", root_dir)
    sim = simulator.FluxoniumSimulator(physics_config)
    
    # 2. Benchmark Simulation Time
    print("--- ⏱️ Benchmarking ---")
    dummy_x = np.zeros(sim.n_flux + sim.n_drive)
    start = time.time()
    f0 = sim.get_fidelity(dummy_x)
    end = time.time()
    print(f"One simulation takes: {end - start:.2f} seconds")
    print(f"Baseline Fidelity (zero coeffs): {f0:.6f}\n")

    # 3. Setup Landscape Slice
    # Load best coeffs if they exist
    out_dir = Path(model_config.get("out_dir", "data/cmaes"))
    best_path = out_dir / "best_coeffs.npy"
    
    if best_path.exists():
        x_center = np.load(best_path)
        print(f"Found best_coeffs.npy. Scanning around the current minimum.")
    else:
        x_center = dummy_x
        print(f"No best_coeffs found. Scanning around the origin.")

    # Pick two random orthogonal unit vectors (u and v) to define our 2D plane
    u = np.random.randn(len(x_center))
    u /= np.linalg.norm(u)
    
    v = np.random.randn(len(x_center))
    v -= np.dot(v, u) * u # ensure orthogonality
    v /= np.linalg.norm(v)

    # 4. Scanning Parameters
    res = 11 # 11x11 grid = 121 simulations
    limit = 0.2 # How far to explore in coefficient-space
    alphas = np.linspace(-limit, limit, res)
    betas = np.linspace(-limit, limit, res)
    
    Z = np.zeros((res, res))

    print(f"--- 🗺️ Scanning Landscape ({res}x{res} grid) ---")
    start_scan = time.time()
    
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            # Move from center along the random plane
            coeffs = x_center + a * u + b * v
            Z[i, j] = sim.get_fidelity(coeffs)
        
        print(f"Progress: {((i+1)/res)*100:.0f}%...")

    scan_time = (time.time() - start_scan) / 60
    print(f"\nScan complete in {scan_time:.2f} minutes.")

    # 5. Visualization
    plt.figure(figsize=(10, 8))
    # Note: we transpose Z to align axes correctly with alphas/betas
    cp = plt.contourf(alphas, betas, Z.T, levels=25, cmap='magma')
    plt.colorbar(cp, label='Fidelity')
    
    plt.scatter([0], [0], color='cyan', marker='*', s=200, label='Center (Best/Origin)')
    
    plt.title(f"2D Projection of the {len(x_center)}D Fidelity Landscape\n(T={physics_config['T']}ns)")
    plt.xlabel("Random Direction U")
    plt.ylabel("Random Direction V")
    plt.legend()
    
    save_path = out_dir / "landscape_viz.png"
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
