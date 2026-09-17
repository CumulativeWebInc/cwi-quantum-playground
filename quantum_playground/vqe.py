"""VQE on H2: variational ground-state energy of the hydrogen molecule.

Uses PennyLane's qchem molecular Hamiltonian (STO-3G basis) and a single
double-excitation ansatz, optimized with gradient descent. Deterministic:
fixed initial parameter and fixed optimizer trajectory on default.qubit.
"""

import pennylane as qml
from pennylane import numpy as np

from .util import result_envelope, to_py

_BACKEND = "default.qubit"
_BOHR_PER_ANGSTROM = 1.8897261246


def vqe_h2(bond_length_angstrom=0.74, steps=10, stepsize=0.4):
    """Estimate the H2 ground-state energy (Hartree) via VQE.

    bond_length_angstrom: H-H distance; 0.74 A is near equilibrium.
    steps: gradient-descent iterations (deterministic, fixed start theta=0).
    Returns initial/final energies, final parameter, and per-step trace.
    """
    bond = float(bond_length_angstrom)
    if bond <= 0:
        raise ValueError(f"bond length must be positive, got {bond}")
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be >= 0")

    half = bond * _BOHR_PER_ANGSTROM / 2.0
    coords = np.array([0.0, 0.0, -half, 0.0, 0.0, half])
    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(["H", "H"], coords)
    dev = qml.device(_BACKEND, wires=n_qubits)

    @qml.qnode(dev)
    def _circuit(theta):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=range(n_qubits))
        qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
        return qml.expval(hamiltonian)

    theta = np.array(0.0, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=float(stepsize))
    e_init = float(_circuit(theta))
    trace = [e_init]
    for _ in range(steps):
        theta = opt.step(_circuit, theta)
        trace.append(float(_circuit(theta)))

    out = result_envelope(circuit="vqe_h2", wires=n_qubits, backend=_BACKEND)
    out.update(
        {
            "bond_length_angstrom": bond,
            "basis": "sto-3g",
            "ansatz": "single double-excitation from Hartree-Fock |1100>",
            "optimizer": f"gradient-descent stepsize={stepsize}",
            "steps": steps,
            "energy_initial_hartree": e_init,
            "energy_final_hartree": trace[-1],
            "theta_final": float(theta),
            "energy_trace_hartree": trace,
            "unit": "hartree",
        }
    )
    return out
