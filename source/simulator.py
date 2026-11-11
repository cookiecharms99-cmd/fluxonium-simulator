"""Fluxonium--cavity reset simulator in a fixed LC x cavity-Fock basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import control_initialization as control_init
import numpy as np
import qutip as qt
import scipy.sparse as sp
from scqubits import Fluxonium


def phase_from_flux(flux_quanta: float | np.ndarray) -> float | np.ndarray:
    """Convert Phi_ext / Phi_0 to the dimensionless Josephson phase."""
    return 2.0 * np.pi * flux_quanta


# Bases!


def sine_basis(times: np.ndarray, count: int) -> np.ndarray:
    """Return sin(k pi t / T), k=1..count; all functions vanish at endpoints."""
    duration = times[-1]
    if count < 0:
        raise ValueError("The number of control coefficients cannot be negative.")
    if count == 0:
        return np.empty((0, len(times)))
    k = np.arange(1, count + 1, dtype=float)[:, None]
    return np.sin(np.pi * k * times / duration)


def charge_fourier_basis(times: np.ndarray, count: int) -> np.ndarray:
    """Return a full Fourier basis for a charge-drive envelope.

    The rows are ``1, sin(2 pi t/T), cos(2 pi t/T), ...``.  Unlike the flux
    pulse, the charge envelope is not required to vanish at the endpoints:
    the experimental decision about its turn-on/off belongs to the chosen
    control protocol, not to this numerical basis.
    """
    if count < 0:
        raise ValueError("The number of control coefficients cannot be negative.")
    if count == 0:
        return np.empty((0, len(times)))
    duration = times[-1]
    basis = np.zeros(shape=(count, len(times)))
    basis[0] = 1.0
    row = 1
    k = 1
    while row < count:
        basis[row] = np.sin(2 * np.pi * k * times / duration)
        row += 1
        if row < count:
            basis[row] = np.cos(2 * np.pi * k * times / duration)
            row += 1
        k += 1
    return basis


def validate_protocol_times(times: np.ndarray) -> np.ndarray:
    """Validate the simulator's standalone control-clock convention.

    Controls describe one protocol beginning at ``0 ns`` and ending at its
    duration. Accepting an arbitrary offset would make the charge carrier's
    phase depend on an implicit, unexplained laboratory clock.
    """
    time_array = np.asarray(times, dtype=float)
    if time_array.ndim != 1 or len(time_array) < 2:
        raise ValueError(
            "times must be a one-dimensional array with at least two samples."
        )
    if not np.all(np.isfinite(time_array)):
        raise ValueError("times must contain only finite values.")
    if not np.isclose(time_array[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(
            "times must start at 0 ns for this standalone control protocol."
        )
    if not np.all(np.diff(time_array) > 0.0):
        raise ValueError("times must be strictly increasing.")
    duration = float(time_array[-1])
    if duration <= 0.0:
        raise ValueError("times must end at a positive duration.")
    return time_array


@dataclass(frozen=True)
class ControlWaveforms:
    """Sampled external phase and charge-drive signal for one simulation."""

    external_phase: np.ndarray
    charge_drive: np.ndarray


@dataclass(frozen=True)
class PathwayDiagnostics:
    """Analysis-only populations for interpreting one completed control pulse.

    ``instantaneous_populations`` use bare fluxonium eigenstates at the
    waveform's current external phase, tensored with cavity Fock states. They
    are diabatic labels, intentionally preserving the ``f0``/``g1`` pathway
    labels through their avoided crossing.
    """

    instantaneous_populations: dict[str, np.ndarray]
    cavity_photon_number: np.ndarray
    trace: np.ndarray


class FluxoniumResetSimulator:
    """Full fluxonium + one-photon cavity model for the active-reset study.

    Candidate ``weights`` have the layout ``[flux weights, charge weights]``.
    The effective configuration declares the number of active coefficients in
    each block. A missing flux block uses its declared fixed waveform; a
    missing charge block leaves the calibrated charge tone unchanged.

    The model uses a fixed LC basis for fluxonium and a two-state cavity Fock
    basis. All frequency-like parameters and times must use one consistent
    unit convention; resolving cycles/ns versus radians/ns remains a separate
    calibration task.
    """

    def __init__(self, params: dict):
        self.params = params
        self.physics = params["physics"]
        self.settings = params["model_settings"]
        self.cutoff = int(self.settings["cutoff"])
        self.control_settings = self._read_control_settings()
        self.n_flux_controls = self.control_settings["flux"]["coefficient_count"]
        self.n_charge_controls = self.control_settings["charge"]["coefficient_count"]
        self._flux_basis = None
        self._charge_basis = None
        self._times = None
        self._idle_eigenstates: np.ndarray | None = None

        flux_values = self.physics["flux_vals"]
        self.idle_flux = float(flux_values["idle_flux"])
        self.interaction_flux = float(flux_values["interaction_flux"])
        self.idle_phase = float(phase_from_flux(self.idle_flux))

        # Pre-cache solver options
        options = self.settings.get("qutip_opts", {})
        self.solver_options = {
            "nsteps": int(float(options.get("qutip_nsteps", 10_000))),
            "atol": float(options.get("qutip_atol", 1e-8)),
            "rtol": float(options.get("qutip_rtol", 1e-6)),
            "normalize_output": True,
        }
        if options.get("qutip_method"):
            self.solver_options["method"] = options["qutip_method"]

        self.build_static_system()

    @property
    def n_controls(self) -> int:
        """Number of active optimization coefficients in this configuration."""
        return self.n_flux_controls + self.n_charge_controls

    def _read_control_settings(self) -> dict[str, dict[str, object]]:
        """Validate the one supported control-configuration schema.

        Every runnable configuration declares ``model_settings.controls`` with
        ``flux`` and ``charge`` blocks. Historical artifacts retain their full
        saved configuration, but are not silently accepted as new inputs.
        """
        configured = self.settings.get("controls")
        if not isinstance(configured, dict):
            raise ValueError(
                "model_settings.controls must declare flux and charge control blocks."
            )
        normalized: dict[str, dict[str, object]] = {}
        for name in ("flux", "charge"):
            section = configured.get(name)
            if not isinstance(section, dict) or "coefficient_count" not in section:
                raise ValueError(
                    "model_settings.controls."
                    f"{name} must be an object with coefficient_count."
                )
            count = int(section["coefficient_count"])
            if count < 0:
                raise ValueError(f"{name} coefficient_count cannot be negative.")
            normalized[name] = {
                "coefficient_count": count,
                "fixed_default": section.get("fixed_default", "idle"),
                "saved_controls_path": section.get("saved_controls_path"),
                "fixed_harmonic_count": section.get("fixed_harmonic_count"),
                "random_seed": section.get("random_seed"),
                "random_scale": section.get("random_scale"),
                "mode": section.get("mode", "calibrated_plus_fourier_correction"),
            }
        return normalized

    def build_static_system(self) -> None:
        """Build operators once in the fixed LC x {|0>, |1>} basis."""
        # This is the simulator's one persistent scqubits object. Its LC-basis
        # operators and idle eigenstates are both used throughout a run.
        self.idle_fluxonium = Fluxonium(
            EJ=self.physics["EJ"],
            EC=self.physics["EC"],
            EL=self.physics["EL"],
            cutoff=self.cutoff,
            flux=self.idle_flux,
        )
        cutoff = self.cutoff

        # Sparse identity matrices
        I_flux = sp.eye(cutoff, format="csr")
        I_cavity = sp.eye(2, format="csr")

        # Base sparse operators from scqubits (LC basis)
        n_f = sp.csr_matrix(self.idle_fluxonium.n_operator(energy_esys=False))
        cos_f = sp.csr_matrix(self.idle_fluxonium.cos_phi_operator())
        sin_f = sp.csr_matrix(self.idle_fluxonium.sin_phi_operator())

        # a_c = [[0, 1], [0, 0]], sigma_y = [[0, -1j], [1j, 0]]
        sigmay = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)

        # Precompute tensor products as Qobjs for mesolve
        dims = [[cutoff, 2], [cutoff, 2]]
        self.n_tensor = qt.Qobj(sp.kron(n_f, I_cavity, format="csr"), dims=dims)
        cos_tensor_sparse = sp.kron(cos_f, I_cavity, format="csr")
        sin_tensor_sparse = sp.kron(sin_f, I_cavity, format="csr")

        # SHO Hamiltonian bit
        levels = np.arange(cutoff, dtype=float) + 0.5
        h_LC_sparse = sp.diags(self.physics["omega_lc"] * levels, 0, format="csr")
        self._h_lc_flux = h_LC_sparse
        self._cos_phi_flux = cos_f
        self._sin_phi_flux = sin_f

        # Cavity term: frequency * (I_flux @ n_c) where n_c = a_c.dag @ a_c = [[0, 0], [0, 1]]
        # Cookie note: Why is line above like that? I really thought it was a_c
        n_c = sp.csr_matrix(([1.0], ([1], [1])), shape=(2, 2))
        self.cavity_number_operator = qt.Qobj(
            sp.kron(I_flux, n_c, format="csr"), dims=dims
        )
        h_cavity_sparse = self.physics["cavity"]["frequency"] * sp.kron(
            I_flux, n_c, format="csr"
        )

        # Coupling term: coupling * (n_f @ sigmay)
        h_coupling_sparse = self.physics["cavity"]["coupling"] * sp.kron(
            n_f, sigmay, format="csr"
        )

        # Total static Hamiltonian
        h_static_sparse = (
            sp.kron(h_LC_sparse, I_cavity, format="csr")
            + h_cavity_sparse
            + h_coupling_sparse
        )
        self.h_static = qt.Qobj(h_static_sparse, dims=dims)

        EJ = self.physics["EJ"]
        self.h_cos = qt.Qobj(-EJ * cos_tensor_sparse, dims=dims)
        self.h_sin = qt.Qobj(EJ * sin_tensor_sparse, dims=dims)
        self.h_drive = self.n_tensor

        # All current initial-state and reward projectors are defined at the
        # idle point. Diagonalize that one Hamiltonian once, rather than
        # recreating identical Fluxonium objects for g, e, and f separately.
        _, self._idle_eigenstates = self.idle_fluxonium.eigensys(evals_count=3)

        # Keep sparse identity for collapse operators
        self.I_flux = I_flux
        self.c_ops = self.collapse_operators()

        self.initial_state = self.idle_mixture()
        self.ground_projector = self.idle_projectors()["g0"]

    def idle_state(self, level: int, photons: int) -> qt.Qobj:
        """Return an idle-flux eigenstate tensored with a cavity Fock state."""
        if self._idle_eigenstates is None or level >= self._idle_eigenstates.shape[1]:
            _, self._idle_eigenstates = self.idle_fluxonium.eigensys(
                evals_count=level + 1
            )
        flux_state = self._idle_eigenstates[:, level]
        cavity_state = np.zeros(2)
        cavity_state[photons] = 1.0
        full_state = np.kron(flux_state, cavity_state).reshape(-1, 1)
        return qt.Qobj(full_state, dims=[[self.cutoff, 2], [1, 1]])

    def idle_mixture(self) -> qt.Qobj:
        g0 = self.idle_state(level=0, photons=0)
        e0 = self.idle_state(level=1, photons=0)
        return 0.5 * qt.ket2dm(g0) + 0.5 * qt.ket2dm(e0)

    def idle_projectors(self) -> dict[str, qt.Qobj]:
        """Idle-basis projectors used to audit the reset pathway."""
        states = {
            "g0": self.idle_state(level=0, photons=0),
            "e0": self.idle_state(level=1, photons=0),
            "f0": self.idle_state(level=2, photons=0),
            "g1": self.idle_state(level=0, photons=1),
        }
        return {name: qt.ket2dm(state) for name, state in states.items()}

    def fixed_flux_waveform(self, times: np.ndarray) -> np.ndarray:
        """Return the configured flux waveform when flux is not optimized."""
        default = self.control_settings["flux"]["fixed_default"]

        if default in (None, "idle"):
            return np.full_like(times, self.idle_phase)

        if default == "piecewise":
            return control_init.piecewise_pulse_init(times, idle_flux=self.idle_phase)

        if default == "random_sine":
            harmonic_count = self.control_settings["flux"]["fixed_harmonic_count"]
            seed = self.control_settings["flux"]["random_seed"]
            scale = self.control_settings["flux"]["random_scale"]
            if harmonic_count is None or seed is None or scale is None:
                raise ValueError(
                    "A random-sine flux default requires fixed_harmonic_count, "
                    "random_seed, and random_scale."
                )
            fixed_weights = np.random.default_rng(int(seed)).normal(
                0.0, float(scale), size=int(harmonic_count)
            )
            return self.idle_phase + fixed_weights @ sine_basis(
                times, int(harmonic_count)
            )
        if default in ("saved_controls", "warm_copy", "warm_save"):
            saved_path = self.control_settings["flux"]["saved_controls_path"]
            if saved_path is None:
                raise ValueError(
                    "A saved-controls flux default requires saved_controls_path."
                )
            return control_init.load_saved_flux_waveform(saved_path, times)
        raise ValueError(f"Unknown fixed flux default: {default!r}.")

    def controls(
        self,
        weights: Sequence[float],
        times: np.ndarray,
    ) -> ControlWaveforms:
        """Build controls from this configuration's active coefficient blocks."""
        weights = np.asarray(weights, dtype=float)
        expected_count = self.n_controls
        if weights.shape != (expected_count,):
            raise ValueError(f"Expected {expected_count} weights, got {weights.shape}.")

        times = validate_protocol_times(times)

        # Flux and charge bases share a time grid but have independent harmonic
        # counts. Rebuild both whenever that exact grid changes.
        if self._times is None or not np.array_equal(times, self._times):
            self._flux_basis = sine_basis(times, self.n_flux_controls)
            self._charge_basis = charge_fourier_basis(times, self.n_charge_controls)
            self._times = times.copy()

        if self.n_flux_controls == 0:
            flux_control = self.fixed_flux_waveform(times)
        else:
            flux_control = (
                self.idle_phase + weights[: self.n_flux_controls] @ self._flux_basis
            )

        carrier = np.sin(float(self.physics["omega_ef"]) * times)
        envelope_default = float(self.physics["Omega"]) / float(self.physics["n_ef"])
        charge_mode = self.control_settings["charge"]["mode"]

        if charge_mode != "calibrated_plus_fourier_correction":
            raise ValueError(f"Unsupported charge-control mode: {charge_mode!r}.")

        if self.n_charge_controls == 0:
            envelope = envelope_default
        else:
            delta_envelope = weights[self.n_flux_controls :] @ self._charge_basis
            envelope = envelope_default + delta_envelope

        charge_control = envelope * carrier

        return ControlWaveforms(
            external_phase=flux_control, charge_drive=charge_control
        )

    def hamiltonian(self, controls: ControlWaveforms) -> list:
        """Return QuTiP's list-form time-dependent Hamiltonian."""
        return [
            self.h_static,
            [self.h_cos, np.cos(controls.external_phase)],
            [self.h_sin, np.sin(controls.external_phase)],
            [self.h_drive, controls.charge_drive],
        ]

    def collapse_operators(
        self, cavity_decay_rate: float | None = None
    ) -> list[qt.Qobj]:
        """Return sqrt(kappa) I x a; zero/None selects coherent evolution."""
        if cavity_decay_rate is None:
            cavity_decay_rate = self.physics["cavity"].get("decay_rate", 0.0)
        if cavity_decay_rate < 0:
            raise ValueError("cavity_decay_rate must be non-negative.")
        if cavity_decay_rate == 0:
            return []

        # a_c = [[0, 1], [0, 0]]
        a_c = sp.csr_matrix(([1.0], ([0], [1])), shape=(2, 2))
        c_op_sparse = np.sqrt(cavity_decay_rate) * sp.kron(
            self.I_flux, a_c, format="csr"
        )
        return [qt.Qobj(c_op_sparse, dims=[[self.cutoff, 2], [self.cutoff, 2]])]

    def evolve(
        self,
        weights: Sequence[float],
        times: np.ndarray,
        *,
        initial_state: qt.Qobj | None = None,
        e_ops: Sequence[qt.Qobj] | None = None,
        cavity_decay_rate: float | None = None,
    ):
        """Evolve a density matrix and optionally record expectation values."""
        waveform = self.controls(weights, times)
        return self._evolve_waveform(
            waveform,
            times,
            initial_state=initial_state,
            e_ops=e_ops,
            cavity_decay_rate=cavity_decay_rate,
        )

    def _evolve_waveform(
        self,
        waveform: ControlWaveforms,
        times: np.ndarray,
        *,
        initial_state: qt.Qobj | None = None,
        e_ops: Sequence[qt.Qobj] | None = None,
        cavity_decay_rate: float | None = None,
        store_states: bool = False,
    ):
        """Evolve preconstructed controls using the shared device model."""
        c_ops = self.c_ops
        if cavity_decay_rate is not None:
            c_ops = self.collapse_operators(cavity_decay_rate)

        options = dict(self.solver_options)
        if store_states:
            options["store_states"] = True

        return qt.mesolve(
            self.hamiltonian(waveform),
            self.initial_state if initial_state is None else initial_state,
            times,
            c_ops=c_ops,
            e_ops=e_ops,
            options=options,
        )

    def final_ground_population(
        self,
        weights: Sequence[float],
        times: np.ndarray,
    ) -> float:
        """Return the final P(g, 0) at the idle bias; the reset objective."""
        result = self.evolve(weights, times, e_ops=[self.ground_projector])
        return float(np.real(result.expect[0][-1]))

    # I will comment this out for now because I do not think it is needed anymore.
    # def ground_population(
    #     self,
    #     weights: Sequence[float],
    #     times: np.ndarray,
    # ) -> np.ndarray:
    #     """Return the idle-basis ground-state population P(g, 0)[t]."""
    #     result = self.evolve(weights, times, e_ops=[self.ground_projector])
    #     return np.real(result.expect[0])

    def pathway_populations(
        self,
        weights: Sequence[float],
        times: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return selected idle-basis populations along the reset pathway.

        The returned curves are ``g0``, ``e0``, ``f0``, and ``g1`` for now.
        """
        projectors = self.idle_projectors()
        names = tuple(projectors)
        result = self.evolve(
            weights,
            times,
            e_ops=[projectors[name] for name in names],
        )
        return {name: np.real(result.expect[index]) for index, name in enumerate(names)}

    def _instantaneous_flux_eigenvectors(
        self, external_phase: float, levels: int = 3
    ) -> np.ndarray:
        """Return bare-fluxonium LC-basis eigenvectors at one external phase."""
        if levels > self.cutoff:
            raise ValueError("Cannot request more fluxonium levels than the cutoff.")
        h_flux = (
            self._h_lc_flux
            - self.physics["EJ"] * np.cos(external_phase) * self._cos_phi_flux
            + self.physics["EJ"] * np.sin(external_phase) * self._sin_phi_flux
        )
        _, eigenvectors = np.linalg.eigh(h_flux.toarray())
        return eigenvectors[:, :levels]

    def pathway_diagnostics(
        self,
        weights: Sequence[float],
        times: np.ndarray,
    ) -> PathwayDiagnostics:
        """Return static-reference and instantaneous pathway diagnostics.

        This method is intentionally for post-run analysis, not CMA-ES. It
        stores the trajectory and diagonalizes a small bare-fluxonium matrix at
        each requested time so the plotted ``g/e/f`` labels follow the flux.
        """
        times = validate_protocol_times(times)
        waveform = self.controls(weights, times)
        result = self._evolve_waveform(waveform, times, store_states=True)
        if len(result.states) != len(times):
            raise RuntimeError("Pathway diagnostics require one stored state per time.")

        labels = {"g0": (0, 0), "e0": (1, 0), "f0": (2, 0), "g1": (0, 1)}

        instantaneous_populations = {
            name: np.empty(len(times), dtype=float) for name in labels
        }

        cavity_photon_number = np.empty(len(times), dtype=float)
        trace = np.empty(len(times), dtype=float)

        for index, (state, phase) in enumerate(
            zip(result.states, waveform.external_phase)
        ):
            eigenvectors = self._instantaneous_flux_eigenvectors(float(phase))
            density_matrix = state.full()
            for name, (level, photons) in labels.items():
                cavity_state = np.zeros(2)
                cavity_state[photons] = 1.0
                ket = np.kron(eigenvectors[:, level], cavity_state)
                instantaneous_populations[name][index] = float(
                    np.real(np.vdot(ket, density_matrix @ ket))
                )
            cavity_photon_number[index] = float(
                np.real(qt.expect(self.cavity_number_operator, state))
            )
            trace[index] = float(np.real(state.tr()))

        return PathwayDiagnostics(
            instantaneous_populations=instantaneous_populations,
            cavity_photon_number=cavity_photon_number,
            trace=trace,
        )


if __name__ == "__main__":
    print("hi!! :D")
