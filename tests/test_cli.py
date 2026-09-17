"""CLI tests: `qp` subcommands exit 0 and print valid JSON."""

import io
import json
from contextlib import redirect_stdout

import pytest

from quantum_playground.cli import main


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    assert rc == 0
    return json.loads(buf.getvalue())


def test_cli_bell():
    r = _run(["bell"])
    assert r["probs"]["00"] == pytest.approx(0.5, abs=1e-9)


def test_cli_ghz():
    r = _run(["ghz", "2"])
    assert r["n"] == 2


def test_cli_rotation():
    r = _run(["rotation", "0.5"])
    assert r["gradient"] == pytest.approx(-0.479425538604203, abs=1e-9)


def test_cli_vqe_fast():
    r = _run(["vqe", "--steps", "2"])
    assert r["energy_final_hartree"] < r["energy_initial_hartree"]


def test_cli_qaoa_fast():
    r = _run(["qaoa", "--steps", "2"])
    assert len(r["expected_cut_trace"]) == 3


def test_cli_demo():
    r = _run(["demo"])
    assert len(r["experiments"]) == 5


def test_cli_ghz_bad_n_exits_2(capsys):
    rc = main(["ghz", "99"])
    assert rc == 2
