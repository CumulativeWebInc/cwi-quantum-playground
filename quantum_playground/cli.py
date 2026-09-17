"""`qp` command-line interface: every API function as a subcommand."""

import argparse
import json
import sys

from .circuits import bell_state, ghz_state
from .demo import demo
from .gradients import rotation_demo
from .qaoa import qaoa_maxcut
from .vqe import vqe_h2


def _dump(result):
    print(json.dumps(result, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qp", description="CWI Quantum Playground — real PennyLane circuits on default.qubit (simulator)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("bell", help="Bell state: probs + <Z> per wire")
    p_ghz = sub.add_parser("ghz", help="GHZ state for n qubits")
    p_ghz.add_argument("n", type=int, nargs="?", default=3, help="qubit count 2..8 (default 3)")
    p_rot = sub.add_parser("rotation", help="RX(theta): <Z> + parameter-shift gradient")
    p_rot.add_argument("theta", type=float, nargs="?", default=0.5, help="angle in radians (default 0.5)")
    p_vqe = sub.add_parser("vqe", help="VQE ground-state energy of H2")
    p_vqe.add_argument("--bond", type=float, default=0.74, help="H-H distance in angstrom (default 0.74)")
    p_vqe.add_argument("--steps", type=int, default=10, help="optimizer steps (default 10)")
    p_qaoa = sub.add_parser("qaoa", help="QAOA MaxCut on the 4-node ring")
    p_qaoa.add_argument("--steps", type=int, default=40, help="optimizer steps (default 40)")
    sub.add_parser("demo", help="run every experiment and print the summary")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "bell":
            _dump(bell_state())
        elif args.cmd == "ghz":
            _dump(ghz_state(args.n))
        elif args.cmd == "rotation":
            _dump(rotation_demo(args.theta))
        elif args.cmd == "vqe":
            _dump(vqe_h2(bond_length_angstrom=args.bond, steps=args.steps))
        elif args.cmd == "qaoa":
            _dump(qaoa_maxcut(steps=args.steps))
        elif args.cmd == "demo":
            results = demo()
            _dump(results)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
