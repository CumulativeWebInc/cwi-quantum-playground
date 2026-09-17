"""Tests for qaoa_maxcut(): p=1 QAOA on the 4-node ring (real PennyLane runs)."""

import pytest

from quantum_playground import qaoa_maxcut


def test_qaoa_improves_expected_cut():
    r = qaoa_maxcut()
    assert r["expected_cut_final"] > r["expected_cut_initial"]


def test_qaoa_initial_expected_cut():
    r = qaoa_maxcut(steps=0)
    assert r["expected_cut_initial"] == pytest.approx(2.7651474012342847, abs=1e-6)


def test_qaoa_final_expected_cut():
    r = qaoa_maxcut()
    # Adam, stepsize 0.1, 40 steps from gamma=beta=0.5 — reproduced by package
    assert r["expected_cut_final"] == pytest.approx(2.99858026068573, abs=1e-4)


def test_qaoa_most_likely_are_optimal_cuts():
    r = qaoa_maxcut()
    top2 = {b["bitstring"] for b in r["most_likely_bitstrings"][:2]}
    # the 4-ring's two maximum cuts (cut size 4)
    assert top2 == {"0101", "1010"}


def test_qaoa_deterministic():
    a = qaoa_maxcut(steps=5)
    b = qaoa_maxcut(steps=5)
    assert a["expected_cut_final"] == b["expected_cut_final"]


def test_qaoa_trace_length():
    r = qaoa_maxcut(steps=5)
    assert len(r["expected_cut_trace"]) == 6


def test_qaoa_rejects_bad_edge():
    with pytest.raises(ValueError):
        qaoa_maxcut(edges=[(0, 9)], n_wires=4)


def test_qaoa_rejects_bad_wires():
    with pytest.raises(ValueError):
        qaoa_maxcut(n_wires=1)
