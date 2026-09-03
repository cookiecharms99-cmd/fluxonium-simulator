import scipy.sparse as sp
import scqubits as scq
import numpy as np
import math

from qutip import Qobj, mesolve

def kron(A, B):
    return sp.kron(A, B, format="csr")

def build_sin_basis(t: np.ndarray, T: float, n_coeffs: int) -> np.ndarray:
    if n_coeffs <= 0:
        return None
    k = np.arange(1, n_coeffs + 1, dtype=float)[:, None]
    return np.sin(np.pi * k * t[None, :] / T)

class FluxoniumSimulator:
    def __init__(self, params):
        self.params = params
        self.physics = self.setup_static_physics(params)
        
        # Precompute operators (Qobj)
        self.cos_op = self.params["EJ"] * self.physics["cos_phi"]
        self.sin_op = self.params["EJ"] * self.physics["sin_phi"]
        self.n_tensor = self.physics["n_tensor"]

        # Number of coefficients for each control
        self.n_flux = self.params.get("n_pulse", self.params.get("n_coeffs", 40))
        self.n_drive = self.params.get("n_drive", 0)

        # Initialize time-dependent variables
        self.T = None
        self.t = None
        self.flux_basis = None
        self.drive_basis = None
        self.baseline = np.pi
        
        # Set default time from params
        if "T" in params:
            self.update_time_grid(params["T"])

    def setup_static_physics(self, params):
        EJ, EC, EL = params["EJ"], params["EC"], params["EL"]
        cutoff = params["cutoff"]
        flux1, flux2 = params["flux1"], params["flux2"]
        omega_q, omega_r, g = params["omega_q"], params["omega_r"], params["g"]

        fluxonium = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=flux1)
        fluxonium2 = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=flux2)

        I_c = sp.eye(cutoff, format="csr")
        I2  = sp.eye(2, format="csr")

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
        c_mat = kron(I_c, sp.csr_matrix(np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=float)))
        c_ops = [Qobj(c_mat)]

        _, states = fluxonium2.eigensys()
        g0 = Qobj(np.kron(states[:, 0], np.array([1.0, 0.0])).reshape(-1, 1))
        e0 = Qobj(np.kron(states[:, 1], np.array([1.0, 0.0])).reshape(-1, 1))
        rho0 = 0.5 * (g0 * g0.dag()) + 0.5 * (e0 * e0.dag())
        
        e_ops = (g0 * g0.dag())

        qutip_options = dict(
            nsteps=params["qutip_nsteps"],
            atol=params["qutip_atol"],
            rtol=params["qutip_rtol"],
            normalize_output=True,
        )
        if params.get("qutip_method"):
            qutip_options["method"] = params["qutip_method"]

        return {
            "cos_phi": cos_phi, "sin_phi": sin_phi, "n_tensor": n_tensor,
            "H0": H0, "c_ops": c_ops, "rho0": rho0, "e_ops": e_ops,
            "qutip_options": qutip_options,
        }

    def update_time_grid(self, T, num_t_steps=None):
        if num_t_steps is None:
            num_t_steps = self.params["num_t_steps"]
        self.T = T
        self.t = np.linspace(0, T, num_t_steps)
        self.flux_basis = build_sin_basis(self.t, T, self.n_flux)
        self.drive_basis = build_sin_basis(self.t, T, self.n_drive)

    def build_drive_envelope(self, weights, t=None):
        """Returns only the envelope (amplitude) of the drive."""
        if weights is None or self.drive_basis is None:
            Omega, n_ef = self.params["Omega"], self.params["n_ef"]
            return np.full_like(t if t is not None else self.t, Omega / n_ef)
        return np.dot(weights, self.drive_basis)

    def build_drive(self, t, weights=None):
        omega_ef = self.params["omega_ef"]
        carrier = np.sin(omega_ef * t)
        envelope = self.build_drive_envelope(weights, t)
        return envelope * carrier
    
    def build_flux(self, weights):
        pulse = np.dot(weights, self.flux_basis) + self.baseline
        A_max = self.params.get("A_max")
        if A_max is not None:
            pulse = A_max * np.tanh(pulse / A_max)
        return pulse

    def update_weights(self, weights, new_T, basis_type="sine"):
        """Updates the weights so that scaling in time retains the pulse shape."""
        weights_flux = weights[:self.n_flux]
        weights_drive = weights[self.n_flux:] if self.n_drive > 0 else None

        if basis_type == "sine":
            # 1. Capture old state
            old_flux_envelope = np.dot(weights_flux, self.flux_basis)
            old_drive_envelope = self.build_drive_envelope(weights_drive)
            t_old = self.t.copy()
            
            # 2. Transition to new time grid
            self.update_time_grid(T=new_T)
            t_new = self.t
            
            # 3. Map old shape into new duration (Time compression/expansion)
            t_scaled = t_new * (t_old[-1] / t_new[-1]) 
            target_flux = np.interp(t_scaled, t_old, old_flux_envelope)
            target_drive = np.interp(t_scaled, t_old, old_drive_envelope)
            
            # 4. Fit new weights using Least Squares
            new_flux, _, _, _ = np.linalg.lstsq(self.flux_basis.T, target_flux, rcond=None)
            
            new_drive = np.array([])
            if self.n_drive > 0:
                new_drive, _, _, _ = np.linalg.lstsq(self.drive_basis.T, target_drive, rcond=None)

            return np.concatenate([new_flux, new_drive])
    
    def get_fidelity(self, weights, T=None):
        if T is not None and T != self.T:
            self.update_time_grid(T)
        
        if self.t is None:
            raise ValueError("Time grid not initialized.")

        flux_weights = weights[:self.n_flux]
        drive_weights = weights[self.n_flux:] if self.n_drive > 0 else None

        flux = self.build_flux(flux_weights)
        drive_signal = self.build_drive(self.t, drive_weights)
        
        H = [
            self.physics["H0"],
            [-self.cos_op, np.cos(flux)],
            [self.sin_op, np.sin(flux)],
            [self.n_tensor, drive_signal],
        ]
         
        result = mesolve( 
            H, self.physics["rho0"], self.t, 
            c_ops=self.physics["c_ops"], 
            e_ops=[self.physics["e_ops"]], 
            options=self.physics["qutip_options"]
        )
        
        return float(result.expect[0][-1].real)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    with open(script_dir.parent / "params.json", "r") as f:
        params = json.load(f)
    
    sim = FluxoniumSimulator(params)
    
    x0 = np.zeros(sim.n_flux + sim.n_drive)
    x_pulse = x0[:sim.n_flux]
    x_drive= x0[sim.n_flux:]
    print(len(x_pulse), len(x_drive))
    f = sim.get_fidelity(x0)
    print(f)


