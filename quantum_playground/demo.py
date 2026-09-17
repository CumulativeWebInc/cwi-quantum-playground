"""demo(): run every experiment once and return the combined results dict."""

from .circuits import bell_state, ghz_state
from .gradients import rotation_demo
from .qaoa import qaoa_maxcut
from .vqe import vqe_h2


def demo():
    """Run the full playground suite with default parameters."""
    return {
        "playground": "cwi-quantum-playground",
        "experiments": {
            "bell_state": bell_state(),
            "ghz_state_n3": ghz_state(3),
            "rotation_demo": rotation_demo(0.5),
            "vqe_h2": vqe_h2(),
            "qaoa_maxcut": qaoa_maxcut(),
        },
    }
