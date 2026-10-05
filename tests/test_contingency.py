"""The N-1 outage sets (`cases.contingency`) and their grading
(`oracle.batch` over `cases.contingency.outage`), tested before any tool.

1. Outages are in-service branches, never island the grid, and come in the
   same order every time.
2. Every outage's own power flow (`cases.truth`, from the base case's
   solution) passes.
3. Planted errors a contingency API could make are caught on exactly the
   outages they affect: the base case's voltages reported for an outage,
   the wrong branch taken out.
"""
import itertools

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from cases import contingency
from cases.matpower import BR_STATUS, BUS_I, F_BUS, T_BUS, parse_m
from cases.registry import CASES
from cases.truth import solve_pf
from oracle import batch
from oracle.evaluate import VM_FLOOR_PU

TOL = {"tol_mva": 1e-3, "tol_pu": 1e-6, "vm_floor": VM_FLOOR_PU}
N = 12   # outages per test case: case14 has 20 branches


@pytest.fixture(scope="module")
def setup():
    key = "case14#n1"
    mpc = parse_m(CASES[key]["file"])
    rows = list(itertools.islice(contingency.candidates(mpc, key), N))
    data = {"branch_row": np.array(rows)}
    ids, v_base = solve_pf(mpc)
    v = np.array([solve_pf(contingency.outage(mpc, data, k), v_base)[1] for k in range(N)])
    return key, mpc, data, [str(int(b)) for b in ids], v_base, np.abs(v), np.rad2deg(np.angle(v))


def _check(mpc, data, ids, vm, va):
    return batch.check(lambda k: contingency.outage(mpc, data, k), len(data["branch_row"]), ids, vm, va, **TOL)


def test_outages_keep_the_grid_connected(setup):
    key, mpc, data, *_ = setup
    assert list(itertools.islice(contingency.candidates(mpc, key), N)) == list(data["branch_row"])
    pos = {int(b): i for i, b in enumerate(mpc["bus"][:, BUS_I])}
    for k in range(N):
        branch = contingency.outage(mpc, data, k)["branch"]
        assert branch[data["branch_row"][k], BR_STATUS] == 0 and (mpc["branch"][:, BR_STATUS] > 0).sum() - 1 == \
            (branch[:, BR_STATUS] > 0).sum()
        on = branch[branch[:, BR_STATUS] > 0]
        f, t = [pos[int(b)] for b in on[:, F_BUS]], [pos[int(b)] for b in on[:, T_BUS]]
        n, _ = connected_components(sp.coo_matrix((np.ones(len(f)), (f, t)), shape=(len(pos), len(pos))))
        assert n == 1


def test_truth_passes_every_outage(setup):
    _, mpc, data, ids, _, vm, va = setup
    out = _check(mpc, data, ids, vm, va)
    assert out["oracle_ok"] and out["scenarios"] == N


def test_base_case_voltages_are_caught(setup):
    _, mpc, data, ids, v_base, vm, va = setup
    vm, va = vm.copy(), va.copy()
    vm[3], va[3] = np.abs(v_base), np.rad2deg(np.angle(v_base))
    out = _check(mpc, data, ids, vm, va)
    assert out["scenarios_failed"] == 1 and out["worst_scenario"] == 3


def test_wrong_branch_is_caught(setup):
    _, mpc, data, ids, _, vm, va = setup
    wrong = {"branch_row": np.roll(data["branch_row"], 1)}
    out = _check(mpc, wrong, ids, vm, va)
    assert out["scenarios_failed"] == N
