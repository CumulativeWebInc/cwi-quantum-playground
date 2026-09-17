"""Shared helpers: JSON-safe envelopes and scalar conversion."""

import math

import pennylane as qml


def to_py(x):
    """Convert numpy scalars/arrays to plain Python floats/lists."""
    if hasattr(x, "tolist"):
        v = x.tolist()
        return v
    if isinstance(x, float):
        return x
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def result_envelope(circuit, wires, backend):
    """Standard envelope every API function returns."""
    return {
        "circuit": circuit,
        "wires": wires,
        "backend": backend,
        "simulator": True,
        "shots": None,
        "pennylane_version": qml.version(),
        "hardware": False,
        "note": "Result of a real state-vector simulation on PennyLane "
        "default.qubit (CPU). Not quantum hardware.",
    }
