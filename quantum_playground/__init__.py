"""CWI Quantum Playground — PennyLane-powered quantum experiments for AI agents.

Every function runs a REAL PennyLane circuit on the ``default.qubit``
state-vector simulator (CPU, analytic, no shots) and returns plain
JSON-serializable dicts. Results are simulations — always labeled as such.
No hardware, no credentials, no invented numbers.
"""

from .circuits import bell_state, ghz_state
from .gradients import rotation_demo
from .vqe import vqe_h2
from .qaoa import qaoa_maxcut
from .demo import demo

__version__ = "1.0.0"
__all__ = ["bell_state", "ghz_state", "rotation_demo", "vqe_h2", "qaoa_maxcut", "demo"]
