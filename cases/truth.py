"""The true state behind a state-estimation case: the case's own power flow,
solved here with no tool.

Measurements are generated from this state (`cases.measurements`), so it has
to be the solution of the `.m` and nothing else. It is built on the oracle's
Ybus and the oracle's reading of bus types (`oracle.residual`), and
`cases.prep` checks it with the tier-1 residual before writing anything: a
wrong truth would make every tool look wrong, and could not go unnoticed.

A plain polar Newton-Raphson from flat start, as every tool is asked to run.
Isolated buses keep NaN: they are not part of any SE problem.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from cases.matpower import ISOLATED, PQ, REF
from oracle.residual import effective_bus_types, specified_injections
from oracle.ybus import make_ybus

# Far below the oracle's 1e-3 MVA, so the truth never limits a tool's grade;
# not lower: PEGASE and RTE cases stall at 1e-11, rounding error in Ybus @ V.
TOLERANCE_PU = 1e-10
MAX_ITERATIONS = 30


def solve_pf(mpc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns `(bus_ids, V)`, complex p.u., NaN on isolated buses. Raises
    RuntimeError if Newton-Raphson does not converge from flat start."""
    ids, y = make_ybus(mpc)
    s, vset = specified_injections(mpc, ids)
    types = effective_bus_types(mpc, vset)
    on = np.flatnonzero(types != ISOLATED)
    y, s, vset, types = y[on][:, on].tocsc(), s[on], vset[on], types[on]
    pvpq, pq = np.flatnonzero(types != REF), np.flatnonzero(types == PQ)
    v = np.where(np.isnan(vset), 1.0, vset).astype(complex)
    for _ in range(MAX_ITERATIONS):
        mis = v * np.conj(y @ v) - s
        f = np.r_[mis.real[pvpq], mis.imag[pq]]
        if np.abs(f).max() < TOLERANCE_PU:
            out = np.full(len(ids), np.nan, complex)
            out[on] = v
            return ids, out
        dv, di, dvn = sp.diags(v), sp.diags(y @ v), sp.diags(v / abs(v))
        ds_da = 1j * dv @ np.conj(di - y @ dv)
        ds_dm = dv @ np.conj(y @ dvn) + np.conj(di) @ dvn
        jac = sp.bmat([[ds_da.real[pvpq][:, pvpq], ds_dm.real[pvpq][:, pq]],
                       [ds_da.imag[pq][:, pvpq], ds_dm.imag[pq][:, pq]]]).tocsc()
        dx = spl.spsolve(jac, -f)
        va, vm = np.angle(v), np.abs(v)
        va[pvpq] += dx[:len(pvpq)]
        vm[pq] += dx[len(pvpq):]
        v = vm * np.exp(1j * va)
    raise RuntimeError(f"no convergence from flat start in {MAX_ITERATIONS} iterations")
