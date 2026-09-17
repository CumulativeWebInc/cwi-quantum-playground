# CWI Quantum Playground (#42)

**Purpose.** Give AI agents a real quantum playground: run textbook circuits, see
parameter-shift gradients, and learn QML primitives (VQE, QAOA) against actual
PennyLane executions — not slides, not mocked numbers.

**Audience.** AI agents and developers who want hands-on quantum-computing
primitives with zero setup friction and honest labels.

**Mechanism.** A small Python package (`quantum_playground`) over PennyLane
`default.qubit` (CPU state-vector simulator). Five experiments:
`bell_state()`, `ghz_state(n)`, `rotation_demo(theta)` (<Z> + analytic
gradient), `vqe_h2()` (H2 ground-state energy, STO-3G), `qaoa_maxcut()`
(p=1 QAOA on the 4-node ring). Every function returns plain JSON-serializable
dicts stamped `simulator: true, hardware: false`. A `qp` CLI mirrors the API;
`qp demo` runs everything.

**Verification.** 47/47 pytest tests green — every number shown in the docs was
produced by a real run and is reproduced by a test (Bell probs, cos/−sin
gradient values, VQE −1.1373 Ha, QAOA 2.9986 expected cut). Live site serves
`results.json` precomputed with PennyLane 0.45.1 on 2026-09-17, labeled as
simulator output.

**Kill rule.** No external agent use by 2026-10-01 → retire the app, keep learnings.

## Honest limits

- Everything here is **simulation** on a classical CPU (`default.qubit`).
  It is not quantum hardware, not a speedup, and not an advantage claim.
- VQE/QAOA results are small pedagogical instances; they prove the workflow,
  not quantum utility.
- No accounts, keys, or spending involved — the $0 path is the only path.

## Quickstart

```bash
pip install pennylane>=0.40
pip install cwi-quantum-playground   # (or clone this repo)
qp demo
```

```python
from quantum_playground import bell_state, rotation_demo, vqe_h2
print(bell_state()["probs"])            # {'00': 0.5, '01': 0.0, ...}
print(rotation_demo(0.5)["gradient"])   # -0.4794 (parameter-shift)
print(vqe_h2()["energy_final_hartree"]) # -1.1373 Ha
```

## Repo layout

- `quantum_playground/` — the package (`circuits`, `gradients`, `vqe`, `qaoa`, `cli`, `demo`)
- `tests/` — 47 pytest tests; every documented number reproduced
- `docs/` — static GitHub Pages site (`index.html`, `results.json`, `llms.txt`, `.well-known/agent-card.json`)

## Results (real runs, PennyLane 0.45.1, default.qubit)

| Experiment | Result |
|---|---|
| Bell state | P(00)=P(11)=0.5, <Z>=0 |
| RX(0.5) | <Z>=0.8776, d<Z>/dθ=−0.4794 (parameter-shift) |
| VQE H2 @0.74Å | −1.1168 → **−1.1373 Ha** (10 GD steps) |
| QAOA MaxCut 4-ring | expected cut 2.765 → **2.999** (Adam, 40 steps); top outcomes 0101/1010 = the two optimal cuts |

Live demo: https://cumulativewebinc.github.io/cwi-quantum-playground/
