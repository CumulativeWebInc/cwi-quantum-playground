"""Tests for vqe_h2(): H2 ground-state energy via VQE (real PennyLane runs)."""

import pytest

from quantum_playground import vqe_h2


def test_vqe_initial_energy_hartree_fock():
    r = vqe_h2(steps=0)
    # Hartree-Fock reference at 0.74 A, STO-3G — reproduced by the package
    assert r["energy_initial_hartree"] == pytest.approx(-1.116759307555259, abs=1e-6)


def test_vqe_lowes_the_energy():
    r = vqe_h2()
    assert r["energy_final_hartree"] < r["energy_initial_hartree"]


def test_vqe_final_energy_near_fci():
    r = vqe_h2()
    # 10 GD steps from theta=0, stepsize 0.4 — reproduced by the package
    assert r["energy_final_hartree"] == pytest.approx(-1.1372755464686324, abs=1e-4)


def test_vqe_trace_length():
    r = vqe_h2(steps=6)
    assert len(r["energy_trace_hartree"]) == 7
    assert r["steps"] == 6


def test_vqe_trace_monotone_decreasing():
    r = vqe_h2(steps=10)
    trace = r["energy_trace_hartree"]
    assert all(b <= a + 1e-12 for a, b in zip(trace, trace[1:]))


def test_vqe_labels():
    r = vqe_h2(steps=1)
    assert r["unit"] == "hartree"
    assert r["basis"] == "sto-3g"
    assert r["simulator"] is True
    assert r["hardware"] is False


def test_vqe_rejects_bad_bond():
    with pytest.raises(ValueError):
        vqe_h2(bond_length_angstrom=-0.5)
