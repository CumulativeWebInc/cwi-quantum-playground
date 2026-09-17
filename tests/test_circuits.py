"""Tests for bell_state() and ghz_state() — real PennyLane runs, analytic values."""

import math

import pytest

from quantum_playground import bell_state, ghz_state


def test_bell_probs_sum_to_one():
    r = bell_state()
    assert math.isclose(sum(r["probs"].values()), 1.0, abs_tol=1e-12)


def test_bell_correlated_outcomes():
    r = bell_state()["probs"]
    assert r["00"] == pytest.approx(0.5, abs=1e-9)
    assert r["11"] == pytest.approx(0.5, abs=1e-9)
    assert r["01"] == pytest.approx(0.0, abs=1e-12)
    assert r["10"] == pytest.approx(0.0, abs=1e-12)


def test_bell_expvals_zero():
    r = bell_state()
    assert r["expval_z0"] == pytest.approx(0.0, abs=1e-12)
    assert r["expval_z1"] == pytest.approx(0.0, abs=1e-12)


def test_bell_flagged_entangled():
    assert bell_state()["entangled"] is True


def test_bell_honest_envelope():
    r = bell_state()
    assert r["backend"] == "default.qubit"
    assert r["simulator"] is True
    assert r["hardware"] is False
    assert r["shots"] is None
    assert r["pennylane_version"]


def test_ghz2():
    r = ghz_state(2)["probs"]
    assert r["00"] == pytest.approx(0.5, abs=1e-9)
    assert r["11"] == pytest.approx(0.5, abs=1e-9)
    assert r["01"] == pytest.approx(0.0, abs=1e-12)


def test_ghz3():
    r = ghz_state(3)
    assert r["n"] == 3
    assert len(r["probs"]) == 8
    assert r["p_all_zero"] == pytest.approx(0.5, abs=1e-9)
    assert r["p_all_one"] == pytest.approx(0.5, abs=1e-9)
    assert r["probs"]["010"] == pytest.approx(0.0, abs=1e-12)


def test_ghz4_full_distribution():
    r = ghz_state(4)["probs"]
    assert len(r) == 16
    assert math.isclose(sum(r.values()), 1.0, abs_tol=1e-12)
    assert r["0000"] == pytest.approx(0.5, abs=1e-9)
    assert r["1111"] == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("bad", [0, 1, 9, 100, "3", 3.0, None])
def test_ghz_rejects_bad_n(bad):
    with pytest.raises(ValueError):
        ghz_state(bad)
