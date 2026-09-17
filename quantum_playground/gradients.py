"""Parameter-shift gradients: the primitive behind all quantum ML.

rotation_demo(theta) runs RX(theta)|0> and returns <Z> together with its
analytic gradient d<Z>/dtheta computed by PennyLane's parameter-shift rule.
"""

import pennylane as qml
from pennylane import numpy as np

from .util import result_envelope, to_py

_BACKEND = "default.qubit"


def rotation_demo(theta):
    """RX(theta) on |0>; return <Z>(theta) and d<Z>/dtheta.

    theta: float rotation angle in radians.
    The gradient is computed by qml.grad (parameter-shift), not finite
    differences. Analytic values: <Z> = cos(theta), grad = -sin(theta).
    """
    theta = float(theta)
    dev = qml.device(_BACKEND, wires=1)

    @qml.qnode(dev, diff_method="parameter-shift")
    def _circuit(t):
        qml.RX(t, wires=0)
        return qml.expval(qml.Z(0))

    t = np.array(theta, requires_grad=True)
    expval = float(_circuit(t))
    grad = float(qml.grad(_circuit)(t))

    out = result_envelope(circuit="rotation_demo", wires=1, backend=_BACKEND)
    out.update(
        {
            "theta": theta,
            "expval_z": expval,
            "gradient": grad,
            "diff_method": "parameter-shift",
        }
    )
    return out
