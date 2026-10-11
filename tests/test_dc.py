"""The DC power-flow oracle (`oracle.dc`), tested before any tool.

1. Its own solution of a case passes, whatever reference angle it is given
   (adding a constant to every angle changes no balance and no flow).
2. Planted model errors a DC tool could make are caught: a resistance
   included in the branch admittance, a phase shift or tap ratio dropped, a
   shunt conductance ignored, a flow reported to the wrong end. Each on the
   case where it can show (case14 has taps; case300 Gs; case2848rte phase
   shifters).
3. Missing results are failures: an unreported bus or branch. A
   flows-only tool (`bus_ids=None`) is graded on its flows alone, and
   wrong flows still fail.
4. N-1 (`check_batch` over `cases.contingency.outage`): every outage's own
   solution passes; the base case's flows reported for an outage, or the
   wrong branch taken out, fail on exactly those outages.
"""
import numpy as np
import pytest

from cases import contingency
from cases.matpower import ANGLE, BR_R, BR_X, GS, RATIO, VA, parse_m
from cases.registry import CASES
from oracle import dc

TOL = 1e-3


def _solved(mpc, va_ref=None):
    mdl = dc.model(mpc)
    va, pf = dc.solve(mdl, float(mpc["bus"][mdl.ref, VA]) if va_ref is None else va_ref)
    return mdl.ids, va, list(range(len(pf))), pf


@pytest.fixture(scope="module", params=["case14", "case300"])
def case(request):
    return parse_m(CASES[request.param]["file"])


def test_own_solution_passes_at_any_reference_angle(case):
    for va_ref in (None, 0.0, 37.0):
        r = dc.check(case, *_solved(case, va_ref), TOL)
        assert r["oracle_ok"], r
        assert r["dc_max_balance_mw"] < 1e-9 and r["dc_max_dflow_mw"] < 1e-9


def _planted(mpc, change):
    wrong = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in mpc.items()}
    change(wrong)
    return dc.check(mpc, *_solved(wrong), TOL)


def test_resistance_included_fails(case):
    def with_r(m):
        z = m["branch"][:, BR_R] + 1j * m["branch"][:, BR_X]
        m["branch"][:, BR_X] = 1 / np.imag(1 / z) * -1   # x_eff such that 1/x_eff = -Im(1/z), as a lossy solver's B
    assert not _planted(case, with_r)["oracle_ok"]


def test_tap_ratio_dropped_fails(case):
    assert (case["branch"][:, RATIO] != 0).any()
    assert not _planted(case, lambda m: m["branch"].__setitem__((slice(None), RATIO), 0))["oracle_ok"]


def test_shunt_conductance_ignored_fails():
    mpc = parse_m(CASES["case300"]["file"])
    assert (mpc["bus"][:, GS] != 0).any()
    assert not _planted(mpc, lambda m: m["bus"].__setitem__((slice(None), GS), 0))["oracle_ok"]


def test_phase_shift_dropped_fails():
    mpc = parse_m(CASES["case2848rte"]["file"])
    assert (mpc["branch"][:, ANGLE] != 0).any()
    r = _planted(mpc, lambda m: m["branch"].__setitem__((slice(None), ANGLE), 0))
    assert not r["oracle_ok"] and r["dc_max_dflow_mw"] > 1


def test_flow_at_the_wrong_end_fails(case):
    ids, va, rows, pf = _solved(case)
    assert not dc.check(case, ids, va, rows, -pf, TOL)["oracle_ok"]


def test_missing_results_fail(case):
    ids, va, rows, pf = _solved(case)
    assert not dc.check(case, ids[1:], va[1:], rows, pf, TOL)["oracle_ok"]
    assert not dc.check(case, ids, va, rows[1:], pf[1:], TOL)["oracle_ok"]



def test_flows_only(case):
    _, _, rows, pf = _solved(case)
    r = dc.check(case, None, None, rows, pf, TOL)
    assert r["oracle_ok"] and not r["dc_angles"] and r["dc_max_balance_mw"] is None, r
    assert not dc.check(case, None, None, rows, -pf, TOL)["oracle_ok"]
    assert not dc.check(case, None, None, rows[1:], pf[1:], TOL)["oracle_ok"]


@pytest.fixture(scope="module")
def outages():
    key = "case14#n1"
    mpc = parse_m(CASES[key]["file"])
    rows = list(contingency.candidates(mpc, key))[:8]
    data = {"branch_row": np.array(rows)}
    sols = [_solved(contingency.outage(mpc, data, k)) for k in range(len(rows))]
    ids, rows_all = sols[0][0], sols[0][2]
    return mpc, data, ids, rows_all, np.array([s[1] for s in sols]), np.array([s[3] for s in sols])


def _check_n1(mpc, data, ids, rows, va, pf):
    return dc.check_batch(lambda k: contingency.outage(mpc, data, k), len(data["branch_row"]), ids, va, rows, pf, TOL)


def test_every_outage_passes(outages):
    r = _check_n1(*outages)
    assert r["oracle_ok"] and r["scenarios_failed"] == 0, r


def test_base_flows_reported_for_an_outage_fail_there(outages):
    mpc, data, ids, rows, va, pf = outages
    _, va0, _, pf0 = _solved(mpc)
    va, pf = va.copy(), pf.copy()
    va[3], pf[3] = va0, pf0
    r = _check_n1(mpc, data, ids, rows, va, pf)
    assert r["scenarios_failed"] == 1 and r["worst_scenario"] == 3


def test_flows_only_outages(outages):
    mpc, data, ids, rows, va, pf = outages
    assert _check_n1(mpc, data, None, rows, None, pf)["oracle_ok"]
    pf = pf.copy()
    pf[4] = pf[6]
    r = _check_n1(mpc, data, None, rows, None, pf)
    assert r["scenarios_failed"] == 1 and r["worst_scenario"] == 4 and r["dc_max_balance_mw"] is None


def test_wrong_branch_out_fails_there(outages):
    mpc, data, ids, rows, va, pf = outages
    va, pf = va.copy(), pf.copy()
    va[2], pf[2] = va[5], pf[5]
    r = _check_n1(mpc, data, ids, rows, va, pf)
    assert r["scenarios_failed"] == 1 and r["worst_scenario"] == 2
