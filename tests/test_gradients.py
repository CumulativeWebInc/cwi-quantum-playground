"""Tests for rotation_demo(): RX(theta) expectation + parameter-shift gradient."""

import math

import pytest

from quantum_playground import rotation_demo


def test_rotation_expval_matches_cos():
    r = rotation_demo(0.5)
    assert r["expval_z"] == pytest.approx(0.8775825618903726, abs=1e-9)
    assert r["expval_z"] == pytest.approx(math.cos(0.5), abs=1e-9)


def test_rotation_gradient_matches_minus_sin():
    r = rotation_demo(0.5)
    assert r["gradient"] == pytest.approx(-0.479425538604203, abs=1e-9)
    assert r["gradient"] == pytest.approx(-math.sin(0.5), abs=1e-9)


def test_rotation_zero_angle():
    r = rotation_demo(0.0)
    assert r["expval_z"] == pytest.approx(1.0, abs=1e-12)
    assert r["gradient"] == pytest.approx(0.0, abs=1e-9)


def test_rotation_pi_flips():
    r = rotation_demo(math.pi)
    assert r["expval_z"] == pytest.approx(-1.0, abs=1e-9)


def test_rotation_gradient_agrees_with_analytic():
    for theta in (0.2, 1.2, 2.1):
        r = rotation_demo(theta)
        assert r["gradient"] == pytest.approx(-math.sin(theta), abs=1e-9)


def test_rotation_diff_method_labeled():
    r = rotation_demo(0.5)
    assert r["diff_method"] == "parameter-shift"
    assert r["simulator"] is True
