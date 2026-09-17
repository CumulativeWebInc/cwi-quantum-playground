"""QAOA for MaxCut: a combinatorial optimization demo on a small graph.

Cost Hamiltonian C = sum_{(i,j) in edges} (I - Z_i Z_j)/2, whose expectation
is the expected cut size. One QAOA layer (p=1): cost unitary then mixer
unitary, optimized with gradient descent from fixed initial angles.
Deterministic on default.qubit.
"""

import pennylane as qml
from pennylane import numpy as np

from .util import result_envelope, to_py

_BACKEND = "default.qubit"
_DEFAULT_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 4-node ring, max cut = 4


def qaoa_maxcut(edges=None, n_wires=4, p=1, steps=40, stepsize=0.1):
    """Run p-layer QAOA on MaxCut and return expected cut size before/after.

    edges: list of (i, j) pairs; default is the 4-node ring.
    Returns initial/final expected cut values, final angles, and the most
    likely measured bitstrings after optimization.
    """
    edges = list(_DEFAULT_EDGES) if edges is None else [tuple(e) for e in edges]
    n = int(n_wires)
    p = int(p)
    steps = int(steps)
    if n < 2 or n > 8:
        raise ValueError(f"n_wires must be in [2, 8], got {n}")
    if p < 1:
        raise ValueError("p must be >= 1")
    for i, j in edges:
        if not (0 <= i < n and 0 <= j < n and i != j):
            raise ValueError(f"bad edge {(i, j)} for {n} wires")

    cost_h = sum((qml.I(i) - qml.Z(i) @ qml.Z(j)) / 2 for i, j in edges)
    mixer_h = sum(qml.X(i) for i in range(n))
    dev = qml.device(_BACKEND, wires=n)

    def _qaoa_layer(gamma, beta):
        qml.ApproxTimeEvolution(cost_h, gamma, 1)
        qml.ApproxTimeEvolution(mixer_h, beta, 1)

    @qml.qnode(dev)
    def _cost(params):
        for w in range(n):
            qml.Hadamard(wires=w)
        for layer in range(p):
            _qaoa_layer(params[0, layer], params[1, layer])
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def _probs(params):
        for w in range(n):
            qml.Hadamard(wires=w)
        for layer in range(p):
            _qaoa_layer(params[0, layer], params[1, layer])
        return qml.probs(wires=list(range(n)))

    params = np.array([[0.5] * p, [0.5] * p], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=float(stepsize))

    def _neg_cost(p_):
        return -_cost(p_)

    c_init = float(_cost(params))
    trace = [c_init]
    for _ in range(steps):
        params = opt.step(_neg_cost, params)
        trace.append(float(_cost(params)))

    probs = _probs(params)
    labels = [format(i, f"0{n}b") for i in range(2 ** n)]
    ranked = sorted(zip(labels, [float(x) for x in probs]), key=lambda kv: -kv[1])
    top = [{"bitstring": b, "probability": pr} for b, pr in ranked[:4]]

    out = result_envelope(circuit=f"qaoa_maxcut(p={p})", wires=n, backend=_BACKEND)
    out.update(
        {
            "edges": [list(e) for e in edges],
            "p": p,
            "steps": steps,
            "expected_cut_initial": c_init,
            "expected_cut_final": trace[-1],
            "expected_cut_trace": trace,
            "angles_final": {
                "gamma": [float(g) for g in params[0]],
                "beta": [float(b) for b in params[1]],
            },
            "most_likely_bitstrings": top,
        }
    )
    return out
