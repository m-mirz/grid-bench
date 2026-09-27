"""The AC-OPF oracle (`oracle.opf`) and the reference it grades against
(`cases.pglib`), tested before any tool.

1. PGLib's reference objectives are read from the release's `BASELINE.md`,
   one per registered case.
2. An AC-OPF solved here (SLSQP with analytic Jacobians, dense; test-only,
   the oracle never solves) is accepted on the three case14 variants, and
   lands within the reference's five-digit rounding: the oracle's cost,
   balance and limits agree with an independent solve and with PGLib's
   published numbers.
3. Solutions of a different problem are rejected on the check that
   separates them: the case solved without its branch limits (congested
   variant), without its angle limits (small angle-difference variant), with
   another cost; and a dispatch that the voltages do not carry, a voltage
   out of its limits, a missing generator.
"""
import numpy as np
import pytest
from scipy.optimize import minimize

from cases.matpower import (ANGMAX, ANGMIN, BUS_TYPE, COST, GEN_BUS, GEN_STATUS, PD, PMAX, PMIN, QD, QMAX, QMIN,
                            RATE_A, REF, VMAX, VMIN, parse_m)
from cases.pglib import reference_objective, reference_objectives
from cases.registry import CASES, select
from oracle import opf
from oracle.opf import GAP_OK
from oracle.ybus import make_ybus, make_yft


def _solve_opf(mpc: dict):
    """Test-only AC-OPF, flat start. Returns (result, vm, va_deg, pg_mw, qg_mvar)."""
    bus, gen, br = mpc["bus"], mpc["gen"], mpc["branch"]
    base = float(mpc["baseMVA"])
    ids, y = make_ybus(mpc)
    y = y.toarray()
    rows, f, t, yf, yt = make_yft(mpc)
    yf, yt = yf.toarray(), yt.toarray()
    n, on = len(ids), np.flatnonzero(gen[:, GEN_STATUS] > 0)
    ng = len(on)
    pos = {int(b): i for i, b in enumerate(ids)}
    cg = np.zeros((n, ng))
    cg[[pos[int(b)] for b in gen[on, GEN_BUS]], np.arange(ng)] = 1
    nr = np.array([i for i in range(n) if bus[i, BUS_TYPE] != REF])
    sd = (bus[:, PD] + 1j * bus[:, QD]) / base
    cf, ct = np.eye(n)[f], np.eye(n)[t]
    rate = br[rows, RATE_A] / base
    lim = rate > 0
    lo, hi = np.deg2rad(br[rows, ANGMIN]), np.deg2rad(br[rows, ANGMAX])
    nx = len(nr) + n + 2 * ng
    gc = mpc["gencost"][on]

    def unpack(x):
        va = np.zeros(n)
        va[nr] = x[:len(nr)]
        k = len(nr)
        return va, x[k:k + n], x[k + n:k + n + ng], x[k + n + ng:]

    def volt(x):
        va, vm, _, _ = unpack(x)
        return vm * np.exp(1j * va), vm

    def dsbus(v, vm):   # MATPOWER dSbus_dV, polar, columns: all buses
        ib = y @ v
        dva = 1j * np.diag(v) @ np.conj(np.diag(ib) - y @ np.diag(v))
        dvm = np.diag(v) @ np.conj(y @ np.diag(v / vm)) + np.diag(np.conj(ib)) @ np.diag(v / vm)
        return dva, dvm

    def dsbr(yb, c, v, vm):   # MATPOWER dSbr_dV for one end
        i = yb @ v
        vb = c @ v
        dva = 1j * (np.diag(np.conj(i)) @ c @ np.diag(v) - np.diag(vb) @ np.conj(yb @ np.diag(v)))
        dvm = np.diag(vb) @ np.conj(yb @ np.diag(v / vm)) + np.diag(np.conj(i)) @ c @ np.diag(v / vm)
        return vb * np.conj(i), dva, dvm

    def cost(x):
        return opf.cost(gc, unpack(x)[2] * base)

    def cost_grad(x):
        g = np.zeros(nx)
        p = unpack(x)[2] * base
        g[len(nr) + n:len(nr) + n + ng] = [np.polyval(np.polyder(r[4:4 + int(r[3])]), pi) * base for r, pi in zip(gc, p)]
        return g

    def balance(x):
        v, _ = volt(x)
        _, _, pg, qg = unpack(x)
        m = v * np.conj(y @ v) - cg @ (pg + 1j * qg) + sd
        return np.r_[m.real, m.imag]

    def balance_jac(x):
        v, vm = volt(x)
        dva, dvm = dsbus(v, vm)
        z = np.zeros((n, ng))
        return np.block([[dva.real[:, nr], dvm.real, -cg, z], [dva.imag[:, nr], dvm.imag, z, -cg]])

    def flows(x):
        v, vm = volt(x)
        out = []
        for yb, c in ((yf, cf), (yt, ct)):
            s = (c @ v) * np.conj(yb @ v)
            out.append((rate ** 2 - np.abs(s) ** 2)[lim])
        return np.concatenate(out)

    def flows_jac(x):
        v, vm = volt(x)
        out = []
        for yb, c in ((yf, cf), (yt, ct)):
            s, dva, dvm = dsbr(yb, c, v, vm)
            ds = lambda d: -2 * (s.real[:, None] * d.real + s.imag[:, None] * d.imag)
            out.append(np.hstack([ds(dva)[:, nr], ds(dvm), np.zeros((len(rows), 2 * ng))])[lim])
        return np.vstack(out)

    bounded = (lo > -2 * np.pi) | (hi < 2 * np.pi)
    a = np.zeros((len(rows), n))
    a[np.arange(len(rows)), f], a[np.arange(len(rows)), t] = 1, -1
    a = np.hstack([a[:, nr], np.zeros((len(rows), n + 2 * ng))])[bounded]
    angles = lambda x: np.r_[a @ x - lo[bounded], hi[bounded] - a @ x]
    angles_jac = lambda x: np.vstack([a, -a])

    bounds = ([(None, None)] * len(nr) + list(zip(bus[:, VMIN], bus[:, VMAX]))
              + list(zip(gen[on, PMIN] / base, gen[on, PMAX] / base)) + list(zip(gen[on, QMIN] / base, gen[on, QMAX] / base)))
    x0 = np.r_[np.zeros(len(nr)), np.ones(n), (gen[on, PMIN] + gen[on, PMAX]) / 2 / base,
               (gen[on, QMIN] + gen[on, QMAX]) / 2 / base]
    scale = 1 / max(1.0, abs(cost(x0)))   # an objective of order 1, so the stopping test means something
    r = minimize(lambda x: cost(x) * scale, x0, jac=lambda x: cost_grad(x) * scale, method="SLSQP", bounds=bounds,
                 constraints=[{"type": "eq", "fun": balance, "jac": balance_jac},
                              {"type": "ineq", "fun": flows, "jac": flows_jac},
                              {"type": "ineq", "fun": angles, "jac": angles_jac}],
                 options={"ftol": 1e-12, "maxiter": 2000})
    va, vm, pg, qg = unpack(r.x)
    keys = [str(i) for i in ids]
    return (r, dict(zip(keys, vm)), dict(zip(keys, np.rad2deg(va))),
            {str(g): p * base for g, p in zip(on, pg)}, {str(g): q * base for g, q in zip(on, qg)})


def _mpc(key):
    return parse_m(CASES[key]["file"])


def _grade(key, sol, mpc=None):
    _, vm, va, pg, qg = sol
    return opf.check(mpc if mpc is not None else _mpc(key), reference_objective(key), vm, va, pg, qg)


@pytest.fixture(scope="module")
def solved():
    return {k: _solve_opf(_mpc(k)) for k in ("pglib_opf_case14_ieee", "pglib_opf_case14_ieee__api",
                                             "pglib_opf_case14_ieee__sad")}


def test_every_case_has_a_reference():
    refs = reference_objectives()
    assert len(refs) == 198   # 66 cases, typical / congested / small angle difference
    assert reference_objective("pglib_opf_case14_ieee") == 2178.1
    assert all(k in refs for k in select("opf-pglib"))


@pytest.mark.parametrize("key", ["pglib_opf_case14_ieee", "pglib_opf_case14_ieee__api", "pglib_opf_case14_ieee__sad"])
def test_independent_solve_is_accepted(solved, key):
    out = _grade(key, solved[key])
    assert solved[key][0].success
    assert out["oracle_ok"], out
    assert abs(out["opf_gap"]) < 5e-5   # the reference's five significant digits
    assert max(out["opf_max_dp_mw"], out["opf_max_dq_mvar"]) < 1e-8


def test_angle_reference_does_not_matter(solved):
    r, vm, va, pg, qg = solved["pglib_opf_case14_ieee"]
    assert _grade("pglib_opf_case14_ieee", (r, vm, {b: a + 17.0 for b, a in va.items()}, pg, qg))["oracle_ok"]


def _without(mpc, column, value):
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in mpc.items()}
    out["branch"][:, column] = value
    return out


def test_ignored_branch_limits_are_caught():
    key = "pglib_opf_case14_ieee__api"
    out = _grade(key, _solve_opf(_without(_mpc(key), RATE_A, 0)))
    assert not out["oracle_ok"] and out["opf_max_flow_violation_mva"] > 1


def test_ignored_angle_limits_are_caught():
    key = "pglib_opf_case14_ieee__sad"
    no_angles = _without(_without(_mpc(key), ANGMIN, -360), ANGMAX, 360)
    out = _grade(key, _solve_opf(no_angles))
    assert not out["oracle_ok"] and out["opf_max_angle_violation_deg"] > 0.1


def test_another_cost_is_caught():
    key = "pglib_opf_case14_ieee"
    cheap = _mpc(key)
    cheap["gencost"][1, COST + 1] /= 4   # generator 2's linear cost, the one that sets the dispatch
    out = _grade(key, _solve_opf(cheap))
    assert not out["oracle_ok"] and out["opf_gap"] > GAP_OK


def test_dispatch_not_carried_by_voltages_is_caught(solved):
    r, vm, va, pg, qg = solved["pglib_opf_case14_ieee"]
    out = _grade("pglib_opf_case14_ieee", (r, vm, va, {**pg, "1": pg["1"] + 1.0}, qg))
    assert not out["oracle_ok"] and out["opf_max_dp_mw"] > 0.9


def test_voltage_limit_is_caught(solved):
    r, vm, va, pg, qg = solved["pglib_opf_case14_ieee"]
    mpc = _mpc("pglib_opf_case14_ieee")
    tight = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in mpc.items()}
    tight["bus"][:, VMAX] = np.array([vm[str(int(b))] for b in tight["bus"][:, 0]]) - 0.01
    out = _grade("pglib_opf_case14_ieee", solved["pglib_opf_case14_ieee"], tight)
    assert not out["oracle_ok"] and out["opf_max_vm_violation_pu"] > 0.009


def test_missing_generator_fails(solved):
    r, vm, va, pg, qg = solved["pglib_opf_case14_ieee"]
    pg = {k: v for k, v in pg.items() if k != "0"}
    out = _grade("pglib_opf_case14_ieee", (r, vm, va, pg, qg))
    assert not out["oracle_ok"] and out["opf_n_reported_gens"] == out["opf_n_gens"] - 1
