# Fluxonium Project: Progress & Roadmap

## 🚀 Current Achievements

### 1. Modular Physics Simulator (`source/simulator.py`)
- Refactored the simulation into a `FluxoniumSimulator` class.
- **Trainable Drive:** Added support for optimizing both the flux pulse and the drive envelope simultaneously. (NEED TO REVISIT THIS LOL)
- **Sparse Matrix Optimization:** Uses `scipy.sparse` to keep simulations fast and memory-efficient. (ALSO THIS)

### 2. Pulse Adaptation (`source/pulse_resampler.py`)
- Implemented `update_weights` logic using Least Squares fitting.
- This allows us to "squish" or "stretch" a successful pulse from one time duration ($T_0$) to another ($T_1$), giving the optimizer a warm start.

### 3. "Academic Inspiration" (The Professor's Script)
- Analyzed `cmaes_autotune_1.py`.
- **Key Takeaways:** 
  - **Regime Scouting:** Running short tests to find the best optimizer settings.
  - **Hopeless Pruning:** Stopping bad runs early to save time.
  - **Fingerprinting:** Ensuring we don't accidentally load old data into a new physics model.
**Should do this again**
---

## 📅 Future Plans (The "Best of Both Worlds" Strategy)

### Phase 1: The Master Optimizer
Create a new optimization script (`experiments_legit/cmaes_v2.py`) that merges our modular simulator with the professor's robust search logic.
- **Goal:** Get the highest possible fidelity for a fixed $T$ (e.g., 150ns) using trainable drive.

### Phase 2: The Iterative T-Loop
Implement the "Outer Loop" that:
1. Reaches a fidelity goal (e.g., 99.9%) at time $T$.
2. Reduces $T$ (e.g., $T_{new} = T - 10\text{ns}$).
3. Uses the Resampler to adapt the previous best weights to the new time.
4. Re-optimizes until the fidelity goal is no longer reachable.

### Phase 3: Data Pipeline (`source/data_pipe.py`)
- Standardize how we log these iterative runs so we can visualize how the pulse shape evolves as $T$ gets smaller.

---
*Last Updated: Friday, May 22, 2026*
