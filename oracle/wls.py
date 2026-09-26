"""Grades a state estimate against the weighted least-squares problem it
claims to solve, with no tool and no solver of its own.

A state-estimation case is a `.m` (the network) plus a measurement set
(`cases.measurements`). WLS state estimation minimizes

    J(x) = sum_i ((z_i - h_i(x)) / sigma_i)^2

over the state x (bus voltage angles except the slack's, and magnitudes).
The oracle rebuilds h(x) and its Jacobian H from the oracle's own Ybus and
Yf (`oracle.ybus`), evaluates them at the tool's estimate, and checks:

- stationarity: one Gauss-Newton step from the estimate,
  dx = (H'WH)^-1 H'W (z - h(x)), is how far the estimate is from the WLS
  optimum to first order. `max_step` is its largest entry, in p.u. and
  radians: the same units, and the same meaning, as the state-update
  tolerance every tool is given. A tool that solved a different problem
  (a sign, a side, a unit, a dropped measurement) is far from this optimum.
- no worse than the truth: the true state is a candidate, so the optimum
  cannot have a larger J; this catches a stationary point that is not the
  minimum.

The error against the true state is recorded as well, for information
only: with noise, the correct WLS answer is not the true state.

J and the step are invariant to a global rotation of all angles, so tools
that pick a different angle reference need no alignment; the informational
angle error is aligned at the slack.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from cases import measurements
from cases.matpower import BUS_TYPE, F_BUS, ISOLATED, T_BUS, parse_m
from oracle.ybus import make_ybus, make_yf

# Calibrated with the Gauss-Newton in tests/test_wls.py: converged, the step is
# 1e-9 to 1e-14; one iteration short of that, 3e-6 (case118~noisy), 8e-6
# (case33bw~noisy), 5e-7 (case14~noisy, where J is already 41.4 of 41.4).
STEP_OK = 1e-6        # p.u. / rad; tools are asked to converge to 1e-8 on the state update
J_SLACK = 1e-9        # absolute, for `exact` cases: J is 1e-24 at the truth, 2e-7 one iteration short


class Model:
    """h(x) and H for one measurement set, on one network."""

    def __init__(self, mpc: dict, meas: dict):
        ids, self.ybus = make_ybus(mpc)
        self.ybus = self.ybus.tocsr()
        rows, from_pos, yf = make_yf(mpc)
        self.base = float(mpc["baseMVA"])
        self.ids = [str(i) for i in ids]
        pos = {b: i for i, b in enumerate(self.ids)}
        self.energized = np.flatnonzero(mpc["bus"][:, BUS_TYPE] != ISOLATED)
        self.slack = pos[str(meas["slack"])]
        row_of = {int(r): k for k, r in enumerate(rows)}
        ms = meas["measurements"]
        self.z = np.array([m["value"] for m in ms])
        self.sigma = np.array([m["sigma"] for m in ms])
        kinds = np.array([m["kind"] for m in ms])
        self.vm_i = np.flatnonzero(kinds == "vm")
        self.pi_i, self.qi_i = np.flatnonzero(kinds == "p_inj"), np.flatnonzero(kinds == "q_inj")
        self.pf_i, self.qf_i = np.flatnonzero(kinds == "p_from"), np.flatnonzero(kinds == "q_from")
        bus_of = lambda idx: np.array([pos[str(ms[i]["bus"])] for i in idx], dtype=int)
        branch_of = lambda idx: np.array([row_of[ms[i]["branch_row"]] for i in idx], dtype=int)
        self.vm_bus, self.pi_bus, self.qi_bus = bus_of(self.vm_i), bus_of(self.pi_i), bus_of(self.qi_i)
        self.pf_br, self.qf_br = branch_of(self.pf_i), branch_of(self.qf_i)
        assert all(ms[i]["from_bus"] == int(mpc["branch"][rows[k], F_BUS])
                   and ms[i]["to_bus"] == int(mpc["branch"][rows[k], T_BUS])
                   for i, k in zip(np.r_[self.pf_i, self.qf_i], np.r_[self.pf_br, self.qf_br]))
        self.yf = yf.tocsr()
        n = len(self.ids)
        self.cf = sp.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), from_pos)), shape=(len(rows), n))
        # State columns: angles of energized non-slack buses, then magnitudes of energized buses.
        self.va_cols = np.array([i for i in self.energized if i != self.slack])
        self.vm_cols = self.energized

    def h_and_H(self, v: np.ndarray) -> tuple[np.ndarray, sp.csr_matrix]:
        """Measurement functions (in the measurements' units) and their
        Jacobian with respect to the state, at complex bus voltages `v`."""
        vn = np.where(v != 0, v / np.where(v != 0, abs(v), 1), 0)
        dv, dvn = sp.diags(v), sp.diags(vn)
        ibus = self.ybus @ v
        sbus = v * np.conj(ibus) * self.base
        ds_da = 1j * dv @ np.conj(sp.diags(ibus) - self.ybus @ dv) * self.base
        ds_dm = (dv @ np.conj(self.ybus @ dvn) + sp.diags(np.conj(ibus)) @ dvn) * self.base
        i_f = self.yf @ v
        v_f = self.cf @ v
        sf = v_f * np.conj(i_f) * self.base
        dsf_da = 1j * (sp.diags(np.conj(i_f)) @ self.cf @ dv - sp.diags(v_f) @ np.conj(self.yf @ dv)) * self.base
        dsf_dm = (sp.diags(v_f) @ np.conj(self.yf @ dvn) + sp.diags(np.conj(i_f)) @ self.cf @ dvn) * self.base

        h = np.empty(len(self.z))
        h[self.vm_i] = abs(v[self.vm_bus])
        h[self.pi_i], h[self.qi_i] = sbus.real[self.pi_bus], sbus.imag[self.qi_bus]
        h[self.pf_i], h[self.qf_i] = sf.real[self.pf_br], sf.imag[self.qf_br]

        n = len(v)
        eye = sp.identity(n, format="csr")
        blocks = [(self.vm_i, sp.csr_matrix((len(self.vm_i), n)), eye[self.vm_bus]),
                  (self.pi_i, ds_da.real.tocsr()[self.pi_bus], ds_dm.real.tocsr()[self.pi_bus]),
                  (self.qi_i, ds_da.imag.tocsr()[self.qi_bus], ds_dm.imag.tocsr()[self.qi_bus]),
                  (self.pf_i, dsf_da.real.tocsr()[self.pf_br], dsf_dm.real.tocsr()[self.pf_br]),
                  (self.qf_i, dsf_da.imag.tocsr()[self.qf_br], dsf_dm.imag.tocsr()[self.qf_br])]
        order = np.concatenate([b[0] for b in blocks])
        stacked = sp.vstack([sp.hstack([a[:, self.va_cols], m[:, self.vm_cols]]) for _, a, m in blocks]).tocsr()
        big_h = stacked[np.argsort(order)]
        return h, big_h

    def objective(self, v: np.ndarray) -> float:
        h, _ = self.h_and_H(v)
        return float((((self.z - h) / self.sigma) ** 2).sum())

    def step(self, v: np.ndarray) -> np.ndarray:
        """One Gauss-Newton step from `v`, as [d_va (non-slack), d_vm]."""
        h, big_h = self.h_and_H(v)
        w = sp.diags(1 / self.sigma ** 2)
        gain = (big_h.T @ w @ big_h).tocsc()
        return spl.spsolve(gain, big_h.T @ w @ (self.z - h))

    def voltages(self, vm: dict[str, float], va_deg: dict[str, float]) -> np.ndarray:
        """Complex voltages from a solution keyed by bus number; 0 where
        not reported (and on isolated buses)."""
        return np.array([vm[b] * np.exp(1j * np.deg2rad(va_deg[b])) if b in vm and b in va_deg else 0
                         for b in self.ids])


@lru_cache(maxsize=None)
def _model(case_file: str, meas_file: str) -> tuple[Model, dict]:
    meas = measurements.read(meas_file)
    return Model(parse_m(case_file), meas), meas


def check(case_file, meas_file, vm: dict[str, float], va_deg: dict[str, float]) -> dict:
    """Flat fields for a result record's `extra_info` (see the module docstring)."""
    model, meas = _model(str(case_file), str(meas_file))
    truth = meas["true_state"]
    reported = [b for b in truth if b in vm and b in va_deg and np.isfinite(vm[b]) and np.isfinite(va_deg[b])]
    out = {"se_n_buses": len(truth), "se_n_reported": len(reported), "se_n_measurements": len(model.z),
           "se_measurements_sha256": hashlib.sha256(Path(meas_file).read_bytes()).hexdigest()}
    if len(reported) < len(truth):
        return out | {"oracle_ok": False}

    v = model.voltages(vm, va_deg)
    v_true = model.voltages({b: t[0] for b, t in truth.items()}, {b: t[1] for b, t in truth.items()})
    j, j_true = model.objective(v), model.objective(v_true)
    step = np.abs(model.step(v)).max()
    slack = model.ids[model.slack]
    dva = [(va_deg[b] - va_deg[slack]) - (t[1] - truth[slack][1]) for b, t in truth.items()]
    out |= {
        "se_J": j, "se_J_true": j_true, "se_max_step": float(step),
        "se_max_dvm_true_pu": max(abs(vm[b] - t[0]) for b, t in truth.items()),
        "se_max_dva_true_deg": float(np.abs((np.array(dva) + 180) % 360 - 180).max()),
    }
    out["oracle_ok"] = bool(step <= STEP_OK and j <= j_true * (1 + 1e-9) + J_SLACK)
    return out
