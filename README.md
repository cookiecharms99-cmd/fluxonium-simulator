# Fluxonium Reset Optimization

This project studies active reset of a fluxonium qubit coupled to a lossy microwave cavity. The immediate goal is to maximize the final population of the ground state \(|g,0\rangle\), currently targeting at least 0.999, while reducing the total reset time.

The repository has completed its first physics audit and simulator refactor.
CMA-ES is the established optimization baseline; reinforcement learning is a
later exploratory direction, especially if one trained policy can transfer
across device conditions or support feedback. Currently we have implemented

For the current implementation checkpoint and immediate next task, read
[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md). The active checklist is in
[`TODO_TODAY.md`](TODO_TODAY.md).

## Physical protocol

The practical initial condition is modeled as an incoherent mixture,

$\rho(0)=\tfrac12|g,0\rangle\langle g,0|+\tfrac12|e,0\rangle\langle e,0|$

The intended reset pathway is

$|e,0\rangle \xrightarrow{\text{charge drive}} |f,0\rangle
\xleftrightarrow{\text{fluxonium-cavity coupling}} |g,1\rangle
\xrightarrow{\text{cavity decay}} |g,0\rangle$

The fluxonium Hamiltonian is represented in a fixed truncated LC basis,

$H_\mathrm{flux}(t)
=H_\mathrm{LC}
-E_J\cos\hat\phi\cos\varphi_\mathrm{ext}(t)
+E_J\sin\hat\phi\sin\varphi_\mathrm{ext}(t),$

with

$\phi_{ext}=2\pi\frac{\Phi_{ext}}{\Phi_0}.$

Half flux,($\Phi_\mathrm{ext}/\Phi_0=0.5$), corresponds to phase ($pi$) and is the idle endpoint. Quarter flux is the experimentally calibrated interaction neighborhood. A static model places the bare ($g\leftrightarrow f$) cavity resonance near ($0.253980967,\Phi_0$); this model-derived value should not silently replace the experimental calibration point at ($0.25\,\Phi_0$) (though for simulation purposes this does not yet affect anything.)

The full numerical basis is

[$\text{fluxonium LC basis}\otimes\{|0\rangle,|1\rangle\}_\mathrm{cavity}.$]

For fluxonium cutoff 20, the combined Hilbert-space dimension is 40.

(NOTE! I don't concretely know what the charge drive does haha, just that it's an available control xp)

## Controls and objective

Flux controls use sine coefficients, while the exploratory joint charge control
uses an independent full-Fourier envelope (constant, then sine/cosine pairs)
multiplying the configured ``sin(omega_ef * t)`` carrier. The charge envelope
is deliberately not constrained to vanish at the pulse endpoints. The flux
phase has the form

$\phi_\mathrm{ext}(t)=\pi+\sum_k c_k^\Phi
\sin\left(\frac{k\pi t}{T}\right)$

so the pulse automatically begins and ends at the half-flux idle point. The
project simulator can optimize both flux and charge-drive coefficients. The
professor's reference implementation optimizes only the flux pulse and keeps
the calibrated charge drive fixed at
``(Omega / n_ef) * sin(omega_ef * t)``. The full-Fourier joint basis supersedes
the previous sine-only exploratory charge basis, so old joint-control weights
must not be treated as directly reproducible under this new parameterization.

The optimization objective is the final ground-state population

[$P_{g0}(T)=\operatorname{Tr}\!\left[|g,0\rangle\langle g,0|\rho(T)\right].$]

## Repository structure

| Path                                        | Purpose                                                                                           |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `source/simulator.py`                       | Active refactored fluxonium-cavity simulator.                                                     |
| `source/control_initialization.py`          | Physical pulse guesses and pulse-to-CMA-ES-coefficient initialization helpers.                    |
| `PROJECT_HANDOFF.md`                        | Compact current-state handoff for the next collaboration session.                                 |
| `source/optimization/cmaes.py`              | Reusable serial and multiprocessing CMA-ES machinery.                                             |
| `source/old_sim.py`                         | Earlier project simulator retained for comparison.                                                |
| `source/test.py`                            | Current simulator smoke test and timing script.                                                   |
| `main/run_cmaes.py`                         | Command-line entry point for small or full-physics CMA-ES runs.                                   |
| `tests/`                                    | Automated numerical and multiprocessing integration checks.                                       |
| `config/`                                   | Physics, protocol, numerical, and optimizer configuration.                                        |
| `config/legacy_configs/baseline_reset.json` | Historical pre-refactor baseline description; not a runnable simulator config.                   |
| `config/mini_audit.json`                    | Working configuration used by the refactored simulator and audits.                                |
| `config/reference_prof.json`                | Fixed-charge, 40-flux-coefficient reference configuration modeled on the professor's protocol.    |
| `References/Code/`                          | Professor-provided CMA-ES implementations and comparison code. Treat these as reference material. |
| `References/Research-Texts/`                | Papers and background reading, including the reinforcement-learning thesis.                       |
| `References/fluxonium-project-handoff.docx` | Detailed record of confirmed physics and the audit checkpoint.                                    |
| `playzone/`                                 | Marimo notebooks, exploratory calculations, and learning experiments.                             |
| `jobs_to_send/`                             | SLURM scripts for cluster runs.                                                                   |
| `data/`                                     | Simulation and optimizer outputs.                                                                 |
| `main/`                                     | Work-in-progress production optimization scripts.                                                 |

## Setup

The project requires Python 3.12 or newer and uses `uv` for its reproducible environment.

```bash
uv sync
```

Run the current simulator smoke test with:

```bash
uv run python source/test.py
```

Run the automated test suite with:

```bash
uv run pytest
```

## Friendly commands

Start with the compact workflow guide:

```bash
uv run python main/flux.py commands
```

The friendly front door accepts unique fragments of verbose run names and
creates readable timestamped output directories for new campaigns. For example:

```bash
# List prior artifacts, then inspect or audit one by a short unique fragment.
uv run python main/flux.py list-runs --match warm
uv run python main/flux.py plot --run warm_start_4779342
uv run python main/flux.py audit --run warm_start_4779342 --cutoffs 20 40

# Preview a full-physics 140-ns warm-start command without launching it.
uv run python main/flux.py run --protocol reference --init warm_start \
  --warm-start data/output/cmaes/reference_6h_legacy_4776599/best_weights.npy \
  --duration 140 --full-physics --dry-run
```

Use `run`, `audit`, `plot`, or `benchmark` after `main/flux.py`; each forwards
to the underlying trusted script and prints that exact command first.

## Advanced commands

Run the inexpensive CMA-ES smoke campaign directly with:

```bash
uv run python main/run_cmaes.py
```

Pass `--full-physics` only when the configured cutoff and time grid are intended.

Run the fixed-charge, flux-only reference campaign with:

```bash
uv run main/run_cmaes.py \
  --physics-config config/reference_prof.json \
  --initialization piecewise_lstsq \
  --generations 5 --popsize 4 --workers 2
```

Without `--full-physics`, the runner uses fast cutoff-8 smoke settings. The
reference config activates 40 flux coefficients and keeps the calibrated charge
drive fixed; a joint config activates both blocks.
Choose the CMA-ES center explicitly with `--initialization zero`, `random`, or
`piecewise_lstsq`; the professor-inspired projected waveform is defined only for
the 150-ns, flux-active configuration.

For Athene, submit `jobs_to_send/reference_smoke_10min.slurm` first. After it
finishes with complete artifacts, `jobs_to_send/reference_6h_legacy.slurm`
launches the bounded six-hour full-physics reference campaign. The local
equal-budget initialization comparison is
`scripts/run_reference_initialization_compare.sh`.

## Current findings

- At half flux, the audited static model gives ($g\to e\approx0.697075$) GHz and ($e\to f\approx4.323951$) GHz.
- At quarter flux, the calculated charge matrix elements agree closely with the stored calibrated values: ($|\langle e|n|f\rangle|\approx0.3044204$) and ($|\langle g|n|f\rangle|\approx0.3748523$).
- At ($0.25,\Phi_0$), the bare ($g\to f$) transition is about 31.23 MHz detuned from the configured cavity frequency.
- The coupled spectrum shows the expected avoided crossing between the states dominated by ($|f,0\rangle$) and ($|g,1\rangle$).
- Initial timing tests suggest that caching static physics and using sparse operators materially improve objective-evaluation speed.
- The reference CMA-ES execution path supports serial and multiprocessing
  evaluation, checkpointed controls, and fresh-simulator reproduction. It has
  passed a cutoff-8 smoke run; cutoff-20 scientific baseline scores are still
  pending.
- I was able to reach 0.99 fidelity for flux-only optimized circuit for T = 140ns.
- Unverified: noticing common themes for good solutions towards the beginning interestingly enough

See the analysis in `playzone/fluxonium_audit.py`. Run this from root

```bash
uv run marimo edit playzone/fluxonium_audit.py
```

## Open questions

The following must be resolved or validated before numerical optimization results are treated as final:

- Confirm the cycles-per-nanosecond versus radians-per-nanosecond convention for every stored frequency and for QuTiP evolution.
- Confirm the physical origin and units of the cavity decay parameter currently represented as ($\kappa=2\pi/40$), with collapse operator ($\sqrt{\kappa},a$).
- Preserve the distinction between the experimental quarter-flux calibration and the model-derived bare resonance. (though maybe not super meaningful)
- Validate the one-photon cavity truncation dynamically.
- Record the source and hash of every CMA-ES initial coefficient vector.

## Optimization plan

1. Finish the simulator physics audit and resolve unit conventions.
2. Freeze a trusted reference simulator and reproducible CMA-ES baseline.
3. Profile objective evaluations and optimize implementation details without changing the modeled physics.
4. Search for shorter pulses that reach the target ground-state population.
5. Compare CMA-ES, intuitive pulses, and any reinforcement-learning approach under equal simulator-evaluation and wall-clock budgets.

Reinforcement learning is not assumed to be better. It will be most compelling if it offers transfer across a family of device parameters, robustness, or useful feedback behavior rather than merely optimizing one fixed open-loop pulse.
