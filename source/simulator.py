"""Fluxonium--cavity reset simulator in a fixed LC x cavity-Fock basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import qutip as qt
import scipy.sparse as sp
from scqubits import Fluxonium


def phase_from_flux(flux_quanta: float | np.ndarray) -> float | np.ndarray:
    """Convert Phi_ext / Phi_0 to the dimensionless Josephson phase."""
    return 2.0 * np.pi * flux_quanta


def sine_basis(times: np.ndarray, duration: float, count: int) -> np.ndarray:
    """Return sin(k pi t / T), k=1..count; all functions vanish at endpoints."""
    if count < 0:
        raise ValueError("The number of control coefficients cannot be negative.")
    if count == 0:
        return np.empty((0, len(times)))
    k = np.arange(1, count + 1, dtype=float)[:, None]
    return np.sin(np.pi * k * times[None, :] / duration)


@dataclass(frozen=True)
class ControlWaveforms:
    """Sampled external phase and charge-drive signal for one simulation."""

    external_phase: np.ndarray
    charge_drive: np.ndarray


class FluxoniumResetSimulator:
    """Full fluxonium + one-photon cavity model for the active-reset study.

    ``weights`` has the layout ``[flux weights, charge-drive weights]``. Flux
    coefficients perturb the idle phase pi, so sine control guarantees a pulse
    begins and ends at the half-flux sweet spot.

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
        self._basis = None
        self._basis_shape = None

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
        return self.n_flux_controls + self.n_charge_controls

    def new_fluxonium(self, flux_quanta: float) -> Fluxonium:
        return Fluxonium(
            EJ=self.physics["EJ"],
            EC=self.physics["EC"],
            EL=self.physics["EL"],
            cutoff=self.cutoff,
            flux=flux_quanta,
        )

    def build_static_system(self) -> None:
        """Build operators once in the fixed LC x {|0>, |1>} basis."""
        reference = self.new_fluxonium(0.0)
        cutoff = self.cutoff

        # Sparse identity matrices
        I_f = sp.eye(cutoff, format="csr")
        I_c = sp.eye(2, format="csr")

        # Base sparse operators from scqubits (LC basis)
        n_f = sp.csr_matrix(reference.n_operator(energy_esys=False))
        cos_f = sp.csr_matrix(reference.cos_phi_operator())
        sin_f = sp.csr_matrix(reference.sin_phi_operator())

        # a_c = [[0, 1], [0, 0]], sigma_y = [[0, -1j], [1j, 0]]
        a_c = sp.csr_matrix(([1.0], ([0], [1])), shape=(2, 2))
        sigmay = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)

        # Precompute tensor products as Qobjs for mesolve
        dims = [[cutoff, 2], [cutoff, 2]]
        self.n_tensor = qt.Qobj(sp.kron(n_f, I_c, format="csr"), dims=dims)
        cos_tensor_sparse = sp.kron(cos_f, I_c, format="csr")
        sin_tensor_sparse = sp.kron(sin_f, I_c, format="csr")

        # SHO Hamiltonian bit
        levels = np.arange(cutoff, dtype=float) + 0.5
        h_LC_sparse = sp.diags(self.physics["omega_lc"] * levels, 0, format="csr")

        # Cavity term: frequency * (I_f @ n_c) where n_c = a_c.dag @ a_c = [[0, 0], [0, 1]]
        n_c = sp.csr_matrix(([1.0], ([1], [1])), shape=(2, 2))
        h_cavity_sparse = self.physics["cavity"]["frequency"] * sp.kron(
            I_f, n_c, format="csr"
        )

        # Coupling term: coupling * (n_f @ sigmay)
        h_coupling_sparse = self.physics["cavity"]["coupling"] * sp.kron(
            n_f, sigmay, format="csr"
        )

        # Total static Hamiltonian
        h_static_sparse = (
            sp.kron(h_LC_sparse, I_c, format="csr")
            + h_cavity_sparse
            + h_coupling_sparse
        )
        self.h_static = qt.Qobj(h_static_sparse, dims=dims)

        EJ = self.physics["EJ"]
        self.h_cos = qt.Qobj(-EJ * cos_tensor_sparse, dims=dims)
        self.h_sin = qt.Qobj(EJ * sin_tensor_sparse, dims=dims)
        self.h_drive = self.n_tensor

        # Keep sparse identity for collapse operators
        self.I_flux = I_f
        self.c_ops = self.collapse_operators()

        self.initial_state = self.idle_mixture()
        self.ground_projector = self.projectors_at_flux(self.idle_flux)["g0"]

    def state_at_flux(self, flux_quanta: float, level: int, photons: int) -> qt.Qobj:
        _, states = self.new_fluxonium(flux_quanta).eigensys(evals_count=level + 1)
        flux_state = states[:, level]
        cavity_state = np.zeros(2)
        cavity_state[photons] = 1.0
        full_state = np.kron(flux_state, cavity_state).reshape(-1, 1)
        return qt.Qobj(full_state, dims=[[self.cutoff, 2], [1, 1]])

    def idle_mixture(self) -> qt.Qobj:
        g0 = self.state_at_flux(self.idle_flux, level=0, photons=0)
        e0 = self.state_at_flux(self.idle_flux, level=1, photons=0)
        return 0.5 * qt.ket2dm(g0) + 0.5 * qt.ket2dm(e0)

    def projectors_at_flux(self, flux_quanta: float) -> dict[str, qt.Qobj]:
        """Projectors used to audit the f0 -> g1 -> g0 reset pathway."""
        states = {
            "f0": self.state_at_flux(flux_quanta, level=2, photons=0),
            "g0": self.state_at_flux(flux_quanta, level=0, photons=0),
            "g1": self.state_at_flux(flux_quanta, level=0, photons=1),
        }
        return {name: qt.ket2dm(state) for name, state in states.items()}

    def controls(self, weights: Sequence[float], times: np.ndarray) -> ControlWaveforms:
        """Reconstruct controls; the flux waveform is an external phase."""
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.n_controls,):
            raise ValueError(
                f"Expected {self.n_controls} weights, got {weights.shape}."
            )

        duration = times[-1] - times[0]
        shape = (len(times), self.n_controls)
        if self._basis is None or self._basis_shape != shape:
            self._basis = sine_basis(times - times[0], duration, self.n_controls)
            self._basis_shape = shape

        phase = (
            self.idle_phase
            + weights[: self.n_flux_controls] @ self._basis[: self.n_flux_controls]
        )
        envelope = weights[self.n_flux_controls :] @ self._basis[self.n_flux_controls :]
        drive = envelope * np.sin(float(self.physics["omega_ef"]) * times)
        return ControlWaveforms(external_phase=phase, charge_drive=drive)

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
        self, weights: Sequence[float], times: np.ndarray
    ) -> float:
        """Return the final P(g, 0) at the idle bias; the reset objective."""
        result = self.evolve(weights, times, e_ops=[self.ground_projector])
        return float(np.real(result.expect[0][-1]))
