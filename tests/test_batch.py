"""The batch oracle (`oracle.batch`) and the sweeps it grades
(`cases.sweep`), tested before any tool.

1. The sweep is reproducible, keeps zero injections at zero and stays
   between its bounds.
2. Every scenario's own power flow (`cases.truth`) passes, scenario by
   scenario.
3. Planted errors a batch API could make are caught, on exactly the
   scenarios they affect: one scenario's result reported for another, the
   base case's demand left in place of a scenario's, a missing bus.
"""
import numpy as np
import pytest

from cases import sweep
from cases.matpower import BUS_I, PD, parse_m
from cases.registry import CASES
from cases.truth import solve_pf
from oracle import batch

TOL = {"tol_mva": 1e-3, "tol_pu": 1e-6}


@pytest.fixture(scope="module", params=["case14#sweep", "case33bw#sweep"])
def setup(request):
    """The case, its sweep, and the truth of every scenario (scenarios x buses)."""
    key = request.param
    mpc = parse_m(CASES[key]["file"])
    data = sweep.generate(mpc, key)
    v = []
    for k in range(len(data["scale"])):
        ids, vk = solve_pf(sweep.scenario(mpc, data, k))
        v.append(vk)
    v = np.array(v)
    return key, mpc, data, [str(int(b)) for b in ids], np.abs(v), np.rad2deg(np.angle(v))


def test_sweep_reproducible_and_bounded(setup):
    key, mpc, data, *_ = setup
    again = sweep.generate(mpc, key)
    assert data.keys() == again.keys() and all(np.array_equal(data[k], again[k]) for k in data)
    assert data["scale"].min() == pytest.approx(sweep.SCALE_MIN) and data["scale"].max() <= sweep.SCALE_MAX
    out = sweep.scenario(mpc, data, 7)
    unloaded = ~np.isin(mpc["bus"][:, BUS_I], data["load_bus"])
    assert (out["bus"][unloaded, PD] == 0).all()
    assert not np.array_equal(out["bus"][~unloaded, PD], mpc["bus"][~unloaded, PD])


def test_truth_passes_every_scenario(setup):
    _, mpc, data, ids, vm, va = setup
    out = batch.check(lambda k: sweep.scenario(mpc, data, k), len(data["scale"]), ids, vm, va, **TOL)
    assert out["oracle_ok"] and out["scenarios_failed"] == 0 and out["scenarios"] == sweep.SWEEP_SIZE
    assert max(out["residual_max_dp_mw"], out["residual_max_dq_mvar"]) < 1e-6


def test_swapped_scenarios_are_caught(setup):
    _, mpc, data, ids, vm, va = setup
    vm, va = vm.copy(), va.copy()
    vm[[10, 60]], va[[10, 60]] = vm[[60, 10]], va[[60, 10]]
    out = batch.check(lambda k: sweep.scenario(mpc, data, k), len(data["scale"]), ids, vm, va, **TOL)
    assert not out["oracle_ok"] and out["scenarios_failed"] == 2 and out["worst_scenario"] in (10, 60)


def test_base_demand_is_caught(setup):
    """A tool that never applied scenario 30 solves the base case there."""
    _, mpc, data, ids, vm, va = setup
    base_ids, v = solve_pf(mpc)
    assert [str(int(b)) for b in base_ids] == ids
    vm, va = vm.copy(), va.copy()
    vm[30], va[30] = np.abs(v), np.rad2deg(np.angle(v))
    out = batch.check(lambda k: sweep.scenario(mpc, data, k), len(data["scale"]), ids, vm, va, **TOL)
    assert out["scenarios_failed"] == 1 and out["worst_scenario"] == 30


def test_missing_bus_is_caught(setup):
    _, mpc, data, ids, vm, va = setup
    out = batch.check(lambda k: sweep.scenario(mpc, data, k), len(data["scale"]), ids[1:], vm[:, 1:], va[:, 1:], **TOL)
    assert not out["oracle_ok"] and out["residual_n_checked"] < out["residual_n_buses"]
