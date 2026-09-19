"""The case data is what the literature says it is.

MATPOWER's distribution feeders convert their published units (ohm, kW) in
MATLAB code after the matrices, which no tool's importer runs; the registry
reads the submodule's `matpower-plain/` copies with that code evaluated.
These tests pin that: two feeders reproduce their papers' power-flow results,
and no registered file still carries conversion code.
"""
import re

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from cases.matpower import parse_m
from cases.registry import CASES
from oracle.residual import specified_injections
from oracle.ybus import make_ybus


def _newton(mpc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Plain polar NR for single-slack, PQ-only feeders. Returns (ids, V)."""
    ids, y = make_ybus(mpc)
    y = y.tocsc()
    s, _ = specified_injections(mpc, ids)
    types = mpc["bus"][:, 1].astype(int)
    v = np.ones(len(ids), complex)
    v[types == 3] = mpc["gen"][0, 5]
    pq = np.flatnonzero(types != 3)
    n = len(pq)
    for _ in range(20):
        mis = v * np.conj(y @ v) - s
        f = np.r_[mis.real[pq], mis.imag[pq]]
        if np.abs(f).max() < 1e-10:
            return ids, v
        dv, di, dvn = sp.diags(v), sp.diags(y @ v), sp.diags(v / abs(v))
        ds_da = 1j * dv @ np.conj(di - y @ dv)
        ds_dm = dv @ np.conj(y @ dvn) + np.conj(di) @ dvn
        jac = sp.bmat([[ds_da.real[pq][:, pq], ds_dm.real[pq][:, pq]],
                       [ds_da.imag[pq][:, pq], ds_dm.imag[pq][:, pq]]]).tocsc()
        dx = spl.spsolve(jac, -f)
        va, vm = np.angle(v), np.abs(v)
        va[pq] += dx[:n]
        vm[pq] += dx[n:]
        v = vm * np.exp(1j * va)
    raise AssertionError("did not converge")


# case, losses kW, min |V| p.u., at bus: Baran & Wu (1989) as reproduced by
# MATPOWER's own runpf on these files.
LITERATURE = [("case33bw", 202.68, 0.91309, 18), ("case69", 224.95, 0.90919, 65)]


@pytest.mark.parametrize("case,losses_kw,vmin,bus", LITERATURE)
def test_feeder_matches_literature(case, losses_kw, vmin, bus):
    mpc = parse_m(CASES[case]["file"])
    ids, v = _newton(mpc)
    y = make_ybus(mpc)[1]
    losses = (v * np.conj(y @ v)).real.sum() * mpc["baseMVA"] * 1e3
    assert losses == pytest.approx(losses_kw, abs=0.05)
    assert abs(v).min() == pytest.approx(vmin, abs=1e-5)
    assert ids[np.argmin(abs(v))] == bus


def test_no_registered_case_needs_matlab():
    unevaluated = [k for k, c in CASES.items() if c.get("format") == "matpower"
                   and re.search(r"^\s*mpc\.(bus|branch)\(", c["file"].read_text(), re.M)]
    assert unevaluated == []
