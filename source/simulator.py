"""Fluxonium--cavity reset simulator in a fixed LC x cavity-Fock basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import control_initialization as control_inits
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
        k += 1
    return basis


ControlProtocol = Literal["joint", "reference"]


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


class FluxoniumResetSimulator:
    """Full fluxonium + one-photon cavity model for the active-reset study.

    Joint-protocol ``weights`` have the layout ``[flux weights, charge-drive
    weights]``.  Reference-protocol weights contain only flux coefficients;
    that protocol keeps its calibrated charge tone fixed. Flux coefficients
    perturb the idle phase pi, so sine control guarantees a pulse begins and
    ends at the half-flux sweet spot.

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
        self.n_flux_controls = int(self.settings.get("n_flux", 0))
        self.n_charge_controls = int(self.settings.get("n_charge", 0))
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
        """Number of coefficients in the exploratory joint protocol."""
        return self.n_flux_controls + self.n_charge_controls

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

        # Cavity term: frequency * (I_flux @ n_c) where n_c = a_c.dag @ a_c = [[0, 0], [0, 1]]
        # Cookie note: Why is line above like that? I really thought it was a_c
        n_c = sp.csr_matrix(([1.0], ([1], [1])), shape=(2, 2))
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

    def get_flux_default(self, times: np.ndarray, protocol=None, **kwargs):
        if protocol is None:
            return np.full_like(times, self.idle_phase)
        elif protocol == "piecewise":
            return control_inits.piecewise_pulse_init(times)
        elif protocol == "warm_copy":
            return control_inits.copy_pulse(kwargs["pulse_copy"])
        else:
            raise NotImplementedError("default protocol not yet implemented sorry! :(")

    def controls(
        self,
        weights: Sequence[float],
        times: np.ndarray,
        *,
        flux_default_protocol=None,
        **kwargs,
    ) -> ControlWaveforms:
        """Reconstruct the controls selected by one explicit protocol.
            When n_flux_controls is 0, fallback to a specified (or not) default protocol
            When n_charge_controls is 0, fallback to a specified (or not) default protocol!


        .
            TODO: perhaps implement something similar for charge when it gets optimized for.

            Charge for now is A(t)sin(wt) where A(t) is our defaulted object of interest.
            For now we have A(t) = A_0 + X_charge * fourier-Basis
        """
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
            flux_control = self.get_flux_default(flux_default_protocol, **kwargs)
        else:
            flux_control = (
                self.idle_phase + weights[: self.n_flux_controls] @ self._flux_basis
            )

        # carrier mode for now!
        carrier = np.sin(float(self.physics["omega_ef"]) * times)
        envelope_default = float(self.physics["Omega"]) / float(self.physics["n_ef"])
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
        control_protocol: ControlProtocol = "joint",
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
    ):
        """Evolve preconstructed controls using the shared device model."""
        c_ops = self.c_ops
        if cavity_decay_rate is not None:
            c_ops = self.collapse_operators(cavity_decay_rate)

        return qt.mesolve(
            self.hamiltonian(waveform),
            self.initial_state if initial_state is None else initial_state,
            times,
            c_ops=c_ops,
            e_ops=e_ops,
            options=self.solver_options,
        )

    def final_ground_population(
        self,
        weights: Sequence[float],
        times: np.ndarray,
        *,
        control_protocol: ControlProtocol = "joint",
    ) -> float:
        """Return the final P(g, 0) at the idle bias; the reset objective."""
        result = self.evolve(
            weights,
            times,
            control_protocol=control_protocol,
            e_ops=[self.ground_projector],
        )
        return float(np.real(result.expect[0][-1]))

    def ground_population(
        self,
        weights: Sequence[float],
        times: np.ndarray,
        *,
        control_protocol: ControlProtocol = "joint",
    ) -> np.ndarray:
        """Return the idle-basis ground-state population P(g, 0)[t]."""
        result = self.evolve(
            weights,
            times,
            control_protocol=control_protocol,
            e_ops=[self.ground_projector],
        )
        return np.real(result.expect[0])

    def pathway_populations(
        self,
        weights: Sequence[float],
        times: np.ndarray,
        *,
        control_protocol: ControlProtocol = "joint",
    ) -> dict[str, np.ndarray]:
        """Return selected idle-basis populations along the reset pathway.

        The returned curves are ``g0``, ``e0``, ``f0``, and ``g1`` for now.
        """
        projectors = self.idle_projectors()
        names = tuple(projectors)
        result = self.evolve(
            weights,
            times,
            control_protocol=control_protocol,
            e_ops=[projectors[name] for name in names],
        )
        return {name: np.real(result.expect[index]) for index, name in enumerate(names)}


if __name__ == "__main__":
    print("hi!! :D")
