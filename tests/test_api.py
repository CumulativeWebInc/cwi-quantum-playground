"""API contract tests: JSON-serializable dicts, honest envelopes, demo bundle."""

import json

import pytest

import quantum_playground as qp


def _all_results():
    return {
        "bell": qp.bell_state(),
        "ghz": qp.ghz_state(3),
        "rotation": qp.rotation_demo(0.5),
        "vqe": qp.vqe_h2(steps=2),
        "qaoa": qp.qaoa_maxcut(steps=2),
    }


def test_every_result_json_serializable():
    for name, r in _all_results().items():
        assert isinstance(r, dict), name
        json.dumps(r)  # must not raise


def test_every_result_carries_honest_envelope():
    for name, r in _all_results().items():
        assert r["simulator"] is True, name
        assert r["hardware"] is False, name
        assert r["backend"] == "default.qubit", name
        assert r["pennylane_version"], name
        assert "Not quantum hardware" in r["note"], name


def test_public_api_exports():
    for fn in ("bell_state", "ghz_state", "rotation_demo", "vqe_h2", "qaoa_maxcut", "demo"):
        assert callable(getattr(qp, fn)), fn


def test_demo_bundles_five_experiments():
    d = qp.demo()
    assert set(d["experiments"]) == {
        "bell_state",
        "ghz_state_n3",
        "rotation_demo",
        "vqe_h2",
        "qaoa_maxcut",
    }
    json.dumps(d)
