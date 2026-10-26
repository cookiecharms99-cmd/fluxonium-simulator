import json
import time
from pathlib import Path

import numpy as np
from simulator import *

script_dir = Path(__file__).resolve().parent
data_fp = script_dir.parent / "config" / "mini_audit.json"
with open(data_fp, "r") as data_file:
    params = json.load(data_file)

print("Loaded parameters from mini_audit.json.")
params["model_settings"]["cutoff"] = 20
mysim = FluxoniumResetSimulator(params)

# Generate a random control pulse weights array of size n_controls
np.random.seed(42)
weights = np.random.normal(0.0, 0.1, size=mysim.n_controls)

# Define time array for the simulation
# duration_ns is not in mini_audit.json, let's use 150.0 as a default.
duration = 150.0
num_steps = int(params["model_settings"].get("num_t_steps", 500))
times = np.linspace(0.0, duration, num_steps)
start = time.time()
print(f"Running simulation with duration {duration} ns and {num_steps} time steps...")

final_pop = mysim.final_ground_population(weights, times)
print(f"Final ground state population P(g, 0): {final_pop:.6f}")
end = time.time()
print(end - start)
