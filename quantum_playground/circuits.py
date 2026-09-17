"""Textbook circuits: Bell state and GHZ state, run for real on default.qubit."""

import pennylane as qml
from pennylane import numpy as np

from .util import result_envelope, to_py

_BACKEND = "default.qubit"


def bell_state():
    """Prepare (|00> + |11>)/sqrt(2) and return measurement statistics.

    Returns a dict with the full 2-qubit probability distribution and the
    <Z> expectation on each wire. Analytic (no shots) — deterministic.
    """
    dev = qml.device(_BACKEND, wires=2)

    @qml.qnode(dev)
    def _circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1]), qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

    probs, z0, z1 = _circuit()
    labels = ["00", "01", "10", "11"]
    out = result_envelope(
        circuit="bell_state",
        wires=2,
        backend=_BACKEND,
    )
    out.update(
        {
            "probs": {lab: to_py(p) for lab, p in zip(labels, probs)},
            "expval_z0": to_py(z0),
            "expval_z1": to_py(z1),
            "entangled": bool(abs(to_py(probs[0]) - 0.5) < 1e-9 and abs(to_py(probs[3]) - 0.5) < 1e-9),
        }
    )
    return out


def ghz_state(n):
    """Prepare the n-qubit GHZ state (|0...0> + |1...1>)/sqrt(2).

    n must be an int in [2, 8]. Returns the full 2^n probability
    distribution (analytic). Raises ValueError on bad n.
    """
    if not isinstance(n, int) or isinstance(n, bool) or not (2 <= n <= 8):
        raise ValueError(f"ghz_state requires int n in [2, 8], got {n!r}")
    dev = qml.device(_BACKEND, wires=n)

    @qml.qnode(dev)
    def _circuit():
        qml.Hadamard(wires=0)
        for i in range(1, n):
            qml.CNOT(wires=[0, i])
        return qml.probs(wires=list(range(n)))

    probs = _circuit()
    labels = [format(i, f"0{n}b") for i in range(2 ** n)]
    out = result_envelope(circuit=f"ghz_state(n={n})", wires=n, backend=_BACKEND)
    out.update(
        {
            "n": n,
            "probs": {lab: to_py(p) for lab, p in zip(labels, probs)},
            "p_all_zero": to_py(probs[0]),
            "p_all_one": to_py(probs[-1]),
        }
    )
    return out
