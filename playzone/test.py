import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
    import qutip as qt
    import matplotlib.pyplot as plt

    return np, plt, qt


@app.cell
def _():
    import sys
    sys.executable
    return


@app.cell
def _():
    N: int = 30
    return (N,)


@app.cell
def _(N: int, qt):
    I = qt.qeye(N)
    return (I,)


@app.cell
def _():
    E_J = 4.164
    E_C = 1.531
    E_L = 0.685
    return E_C, E_J, E_L


@app.cell
def _(np):
    phi_ext = np.pi
    return (phi_ext,)


@app.cell
def _(N: int, qt):
    a = qt.destroy(N)
    a_dag = a.dag()
    return a, a_dag


@app.cell
def _(E_C, E_L, a, a_dag):
    phi_zpf = (2 * E_C / E_L)**0.25
    n_zpf = 1 / (2 * phi_zpf)

    phi = phi_zpf * (a + a_dag)
    n = 1j * n_zpf * (a_dag - a)
    return n, phi


@app.cell
def _(E_C, E_J, E_L, I, n, phi, phi_ext):
    def cos_operator(op):
        return (1j * op).expm() / 2 + (-1j * op).expm() / 2

    cos_term = cos_operator(phi - phi_ext * I)

    H = 4 * E_C * n**2 + 0.5 * E_L * (phi - phi_ext * I)**2 - E_J * cos_term
    return (H,)


@app.cell
def _(H):
    energies, states = H.eigenstates()

    for k, energy in enumerate(energies[:6]):
        print(f"E[{k}] = {energy:.6f}")
    return (states,)


@app.cell
def _(n, np, plt, qt, states):
    n_energy_basis = qt.Qobj(
        np.array([
            [(states[i].dag() * n * states[j]) for j in range(8)]
            for i in range(8)
        ])
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(np.abs(n_energy_basis.full()), origin="lower", cmap="magma")
    plt.colorbar(label=r"$|\langle i|\hat n|j\rangle|$")
    plt.xlabel("initial state  j")
    plt.ylabel("final state  i")
    plt.title("Charge-operator transition strengths")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
