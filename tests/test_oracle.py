"""The oracle is the measuring stick, so it is tested before any tool.

1. Ybus against a network small enough to write out by hand, covering every
   branch-model term (series impedance, line charging, off-nominal tap,
   phase shift, bus shunt, out-of-service branch).
2. A solution that satisfies the power-flow equations gives zero residual,
   and one that does not gives a large one on the right bus.
3. A planted data bug (the sign of one reactance flipped, as `np.hypot(r, x)`
   did in gridoxide's converter) is caught.

Independent of these, five tools (pandapower, lightsim2grid, PyPSA,
pypowsybl, VeraGrid) each reach ~1e-10 MW on the oracle for case9241pegase,
building their own admittance matrices; that could not happen if the
oracle's Ybus were wrong.
"""
import numpy as np
import pytest

from cases.matpower import ANGLE, BR_STATUS, BR_X, parse_m
from cases.registry import CASES
from oracle.residual import residual, specified_injections
from oracle.ybus import make_ybus


def _case(branches, buses=None, gens=None):
    buses = buses or [[1, 3, 0, 0, 0, 0, 1, 1.0, 0, 1, 1, 1.1, 0.9],
                      [2, 1, 50, 20, 0, 10, 1, 1.0, 0, 1, 1, 1.1, 0.9],
                      [3, 2, 30, 10, 5, 0, 1, 1.0, 0, 1, 1, 1.1, 0.9]]
    gens = gens or [[1, 0, 0, 100, -100, 1.02, 100, 1, 200, 0],
                    [3, 40, 0, 100, -100, 1.01, 100, 1, 200, 0]]
    rows = [b + [0] * (13 - len(b)) for b in branches]
    return {"baseMVA": 100.0, "bus": np.array(buses, float), "gen": np.array(gens, float),
            "branch": np.array(rows, float)}


#          f  t   r     x    b   rA rB rC ratio angle status
LINE = [1, 2, 0.01, 0.1, 0.04, 0, 0, 0, 0, 0, 1]
XFMR = [2, 3, 0.0, 0.05, 0.0, 0, 0, 0, 0.95, 5.0, 1]
OFF = [1, 3, 0.02, 0.2, 0.0, 0, 0, 0, 0, 0, 0]


def test_ybus_by_hand():
    ids, y = make_ybus(_case([LINE, XFMR, OFF]))
    y = y.toarray()
    ys1 = 1 / (0.01 + 0.1j)
    ys2 = 1 / 0.05j
    t = 0.95 * np.exp(1j * np.deg2rad(5.0))
    expected = np.zeros((3, 3), complex)
    expected[0, 0] = ys1 + 0.02j
    expected[0, 1] = expected[1, 0] = -ys1
    expected[1, 1] = ys1 + 0.02j + ys2 / abs(t) ** 2 + 0.10j   # bus 2 shunt Bs=10 MVAr
    expected[1, 2] = -ys2 / np.conj(t)
    expected[2, 1] = -ys2 / t
    expected[2, 2] = ys2 + 0.05                                # bus 3 shunt Gs=5 MW
    assert list(ids) == [1, 2, 3]
    np.testing.assert_allclose(y, expected, atol=1e-12)


def _newton(mpc, tol=1e-12):
    """Reference NR in polar form, only for these tests. Consistency with the
    oracle's own Ybus is what is under test here; independence comes from the
    by-hand Ybus test above and from the tools."""
    ids, y = make_ybus(mpc)
    y = y.toarray()
    s, vset = specified_injections(mpc, ids)
    types = mpc["bus"][:, 1].astype(int)
    vm = np.where(np.isnan(vset), 1.0, vset)
    va = np.zeros(len(ids))
    pv_pq, pq = np.flatnonzero(types != 3), np.flatnonzero(types == 1)
    for _ in range(20):
        v = vm * np.exp(1j * va)
        mis = v * np.conj(y @ v) - s
        f = np.r_[mis.real[pv_pq], mis.imag[pq]]
        if np.abs(f).max() < tol:
            break
        jac = np.zeros((len(f), len(f)))
        x0 = np.r_[va[pv_pq], vm[pq]]
        for k in range(len(x0)):                         # numerical Jacobian: fine at 3 buses
            x = x0.copy()
            x[k] += 1e-7
            va2, vm2 = va.copy(), vm.copy()
            va2[pv_pq], vm2[pq] = x[:len(pv_pq)], x[len(pv_pq):]
            v2 = vm2 * np.exp(1j * va2)
            m2 = v2 * np.conj(y @ v2) - s
            jac[:, k] = (np.r_[m2.real[pv_pq], m2.imag[pq]] - f) / 1e-7
        dx = np.linalg.solve(jac, -f)
        va[pv_pq] += dx[:len(pv_pq)]
        vm[pq] += dx[len(pv_pq):]
    keys = [str(i) for i in ids]
    return dict(zip(keys, vm)), dict(zip(keys, np.rad2deg(va)))


def test_consistent_solution_has_zero_residual():
    mpc = _case([LINE, XFMR, OFF])
    vm, va = _newton(mpc)
    r = residual(mpc, vm, va)
    assert max(r.max_dp_mw, r.max_dq_mvar) < 1e-6
    assert r.max_dvm_pu < 1e-12
    assert r.n_checked == r.n_buses == 3


def test_residual_ignores_global_angle_shift():
    mpc = _case([LINE, XFMR, OFF])
    vm, va = _newton(mpc)
    r = residual(mpc, vm, {k: a + 17.0 for k, a in va.items()})
    assert max(r.max_dp_mw, r.max_dq_mvar) < 1e-6


def test_wrong_voltage_is_caught_on_the_right_bus():
    mpc = _case([LINE, XFMR, OFF])
    vm, va = _newton(mpc)
    vm["2"] += 0.01
    r = residual(mpc, vm, va)
    assert r.max_dq_mvar > 1.0
    assert r.worst_bus == "2"


def test_missing_bus_limits_what_is_checked():
    mpc = _case([LINE, XFMR, OFF])
    vm, va = _newton(mpc)
    del vm["3"], va["3"]
    r = residual(mpc, vm, va)
    assert r.n_checked == 1     # bus 1 only: bus 2 neighbours the missing bus 3


def test_planted_reactance_sign_flip_is_caught():
    mpc = parse_m(CASES["case14"]["file"])
    vm, va = _newton(mpc)
    assert residual(mpc, vm, va).max_dp_mw < 1e-6
    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in mpc.items()}
    live = np.flatnonzero(bad["branch"][:, BR_STATUS] != 0)
    bad["branch"][live[0], BR_X] *= -1
    vm_bad, va_bad = _newton(bad)
    r = residual(mpc, vm_bad, va_bad)
    assert max(r.max_dp_mw, r.max_dq_mvar) > 10.0


@pytest.mark.parametrize("shift", [0.0, 10.0])
def test_zero_phase_shift_diagnostic(shift):
    branch = list(XFMR)
    branch[ANGLE] = shift
    mpc = _case([LINE, branch])
    no_shift = _case([LINE, branch[:ANGLE] + [0.0] + branch[ANGLE + 1:]])
    vm, va = _newton(no_shift)       # a tool that drops phase shifts
    real = residual(mpc, vm, va)
    zero = residual(mpc, vm, va, zero_phase_shifts=True)
    assert zero.max_dp_mw < 1e-6
    assert (real.max_dp_mw > 1.0) == (shift != 0.0)
