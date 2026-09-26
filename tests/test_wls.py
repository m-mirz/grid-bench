"""The state-estimation oracle (`oracle.wls`) and the data it grades against
(`cases.truth`, `cases.measurements`), tested before any tool.

1. The true state solves the power flow; the Jacobian matches finite
   differences of h(x).
2. On `exact` measurements the truth is the optimum (J = 0, no step).
3. A plain Gauss-Newton, written here and only for these tests, converges
   on `noisy` measurements and passes; stopped early, it fails.
4. Planted errors a tool could make are caught: a wrong angle, a flipped
   injection sign, a flipped flow direction, a missing bus.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from cases import measurements
from cases.matpower import parse_m
from cases.registry import CASES
from oracle import wls
from oracle.residual import residual


def _setup(key):
    mpc = parse_m(CASES[key]["file"])
    data = measurements.generate(mpc, key, CASES[key]["scenario"])
    return mpc, data, wls.Model(mpc, data)


def _grade(mpc, data, vm, va, tmp_path):
    case_file = tmp_path / "case.m"
    case_file.write_text(CASES[data["case"]]["file"].read_text())
    meas_file = tmp_path / "meas.json"
    measurements.write(data, meas_file)
    return wls.check(case_file, meas_file, vm, va)


def _truth(data):
    return ({b: t[0] for b, t in data["true_state"].items()}, {b: t[1] for b, t in data["true_state"].items()})


def _gauss_newton(model, data, tol=1e-10, max_it=30):
    """Flat-start Gauss-Newton WLS, slack angle fixed. Test-only: the oracle
    never solves."""
    v = np.zeros(len(model.ids), complex)
    v[model.energized] = 1.0
    for _ in range(max_it):
        dx = model.step(v)
        va, vm = np.angle(v), np.abs(v)
        va[model.va_cols] += dx[:len(model.va_cols)]
        vm[model.vm_cols] += dx[len(model.va_cols):]
        v = np.where(np.isin(np.arange(len(v)), model.energized), vm * np.exp(1j * va), 0)
        if np.abs(dx).max() < tol:
            break
    on = [model.ids[i] for i in model.energized]
    idx = {b: i for i, b in enumerate(model.ids)}
    return {b: float(abs(v[idx[b]])) for b in on}, {b: float(np.rad2deg(np.angle(v[idx[b]]))) for b in on}


@pytest.mark.parametrize("key", ["case14~exact", "case118~noisy", "case33bw~noisy", "case4_dist~noisy"])
def test_truth_solves_the_case(key):
    mpc, data, _ = _setup(key)
    r = residual(mpc, *_truth(data))
    assert max(r.max_dp_mw, r.max_dq_mvar) < 1e-8 and r.n_checked == r.n_buses


@pytest.mark.parametrize("key", ["case14~noisy", "case4_dist~noisy"])
def test_jacobian_matches_finite_differences(key):
    mpc, data, model = _setup(key)
    vm, va = _truth(data)
    v = model.voltages(vm, va)
    h0, big_h = model.h_and_H(v)
    for col in range(big_h.shape[1]):
        va_, vm_ = np.angle(v), np.abs(v)
        eps = 1e-7
        if col < len(model.va_cols):
            va_[model.va_cols[col]] += eps
        else:
            vm_[model.vm_cols[col - len(model.va_cols)]] += eps
        h1, _ = model.h_and_H(np.where(v != 0, vm_ * np.exp(1j * va_), 0))
        np.testing.assert_allclose((h1 - h0) / eps, big_h[:, col].toarray().ravel(), rtol=1e-4, atol=1e-3)


def test_exact_truth_is_the_optimum(tmp_path):
    mpc, data, _ = _setup("case14~exact")
    out = _grade(mpc, data, *_truth(data), tmp_path)
    assert out["oracle_ok"]
    assert out["se_J"] < 1e-15 and out["se_max_step"] < 1e-10


def test_observable(tmp_path):
    for key in ("case14~noisy", "case118~noisy", "case33bw~noisy"):
        _, data, model = _setup(key)
        v = model.voltages(*_truth(data))
        _, big_h = model.h_and_H(v)
        gain = (big_h.T @ sp.diags(1 / model.sigma ** 2) @ big_h).tocsc()
        assert abs(spl.splu(gain).U.diagonal()).min() > 0


@pytest.mark.parametrize("key", ["case14~noisy", "case118~noisy", "case33bw~noisy"])
def test_converged_gauss_newton_passes(key, tmp_path):
    mpc, data, model = _setup(key)
    vm, va = _gauss_newton(model, data)
    out = _grade(mpc, data, vm, {b: a + 23.0 for b, a in va.items()}, tmp_path)   # any angle reference
    assert out["oracle_ok"], out
    assert out["se_J"] < out["se_J_true"]
    assert out["se_max_dvm_true_pu"] < 0.02


def test_stopped_gauss_newton_fails(tmp_path):
    mpc, data, model = _setup("case118~noisy")
    out = _grade(mpc, data, *_gauss_newton(model, data, max_it=1), tmp_path)
    assert not out["oracle_ok"] and out["se_max_step"] > wls.STEP_OK


def test_wrong_angle_is_caught(tmp_path):
    mpc, data, model = _setup("case14~noisy")
    vm, va = _gauss_newton(model, data)
    va["5"] += 0.01
    assert not _grade(mpc, data, vm, va, tmp_path)["oracle_ok"]


def test_flipped_injection_sign_is_caught(tmp_path):
    """A tool that reads injections as loads estimates a different problem."""
    mpc, data, _ = _setup("case14~noisy")
    flipped = {**data, "measurements": [{**m, "value": -m["value"]} if m["kind"] in ("p_inj", "q_inj") else m
                                        for m in data["measurements"]]}
    vm, va = _gauss_newton(wls.Model(mpc, flipped), flipped)
    assert not _grade(mpc, data, vm, va, tmp_path)["oracle_ok"]


def test_flow_direction_is_caught(tmp_path):
    """A tool that reads a from-end flow as flowing into the from bus (the
    to-end convention, near enough on short lines) estimates a different
    problem."""
    mpc, data, _ = _setup("case14~noisy")
    flipped = {**data, "measurements": [{**m, "value": -m["value"]} if m["kind"] in ("p_from", "q_from") else m
                                        for m in data["measurements"]]}
    vm, va = _gauss_newton(wls.Model(mpc, flipped), flipped)
    assert not _grade(mpc, data, vm, va, tmp_path)["oracle_ok"]


def test_missing_bus_fails(tmp_path):
    mpc, data, model = _setup("case14~noisy")
    vm, va = _gauss_newton(model, data)
    del vm["3"], va["3"]
    out = _grade(mpc, data, vm, va, tmp_path)
    assert not out["oracle_ok"] and out["se_n_reported"] == out["se_n_buses"] - 1


def test_measurements_are_deterministic():
    mpc = parse_m(CASES["case14"]["file"])
    a = measurements.generate(mpc, "case14~noisy", "noisy")
    b = measurements.generate(mpc, "case14~noisy", "noisy")
    assert a == b
    assert a != measurements.generate(mpc, "case14~other", "noisy")


def test_branch_measurements_name_their_branch():
    mpc = parse_m(CASES["case118"]["file"])
    data = measurements.generate(mpc, "case118~noisy", "noisy")
    for m in data["measurements"]:
        if "branch_row" in m:
            assert (m["from_bus"], m["to_bus"]) == tuple(int(x) for x in mpc["branch"][m["branch_row"], :2])
