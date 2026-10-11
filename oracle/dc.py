"""DC power flow, graded against MATPOWER's DC model built from the raw `.m`
(`makeBdc`, `dcpf`, `rundcpf`), tool-free (numpy, scipy).

The model, as MATPOWER defines it:
- every in-service branch whose ends are both energized carries
  b = 1 / (x * tau) per unit (x its reactance, tau its tap ratio, 1 where the
  `.m` says 0); resistance and line charging are not part of it;
- a phase shifter's angle enters as injections: Pfinj = -b * shift at the
  branch, Pbusinj = Cft' Pfinj at its ends;
- every energized bus's specified injection is P = (Pg of its online
  generators - Pd - Gs) / baseMVA - Pbusinj (shunt conductance counts as
  load, as in MATPOWER);
- the reference bus's angle is the case's VA; every other energized bus's
  angle solves B theta = P on the remaining rows. Branch flows are
  Pf = (Bf theta + Pfinj) * baseMVA, from end, MW; 0 for a branch out of
  service.

A tool's result is graded on both:
- angles: the nodal balance |B theta - P| (MW) at every energized bus but the
  reference, with the tool's own angles. Adding a constant to every angle
  changes neither B theta nor any flow, so a tool's choice of reference
  angle does not matter;
- flows: |Pf - Pf_ref| (MW) on every in-service branch row against this
  module's own solution of the same problem. An out-of-service branch (an
  outage's own included) need not be reported; if it is, its flow must be 0.

A tool that computes flows without angles (PTDF and LODF only) reports
`bus_ids=None`: its flows are graded as above, the balance is not graded
(`dc_angles` False, the balance fields None), and its verdict says so. The
flows of every in-service branch determine the angle differences across
them, so on a connected grid this is the same solution, only not reported
in angles. An empty list is not None: it fails as missing buses.

DC N-1 (`check_outages`): every outage is graded the same way against the
case with that branch out of service. Its reference solution comes from the
base case's factorization and an exact rank-one update per outage
(Sherman-Morrison: taking branch r out subtracts b_r c c' from B, c the
branch's incidence vector, and moves its phase-shift injection back into
P), one sparse solve per outage instead of a new factorization, so that
thousands of outages on case9241pegase grade in minutes;
`tests/test_dc.py` checks it against the rebuilt model of every outage.

Both against `tol_mw` (the AC tolerance, `RESIDUAL_OK_MVA`): DC is linear,
so a correct tool is exact to round-off, and a model difference (a
resistance included, a phase shift or tap dropped, Gs ignored) shows as a
large residual at exactly the buses and branches it touches.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

from cases.matpower import (ANGLE, BR_STATUS, BR_X, BUS_I, BUS_TYPE, F_BUS, GEN_BUS, GEN_STATUS, GS, ISOLATED, PD,
                            PG, RATIO, REF, T_BUS, VA)


@dataclass
class Model:
    ids: list[str]            # MATPOWER bus numbers, bus row order
    energized: np.ndarray     # bool per bus row
    ref: int                  # row of the reference bus
    B: sp.csr_matrix          # bus x bus, per unit
    Bf: sp.csr_matrix         # branch x bus, per unit (rows of out-of-service branches empty)
    Pfinj: np.ndarray         # per branch, per unit
    P: np.ndarray             # specified injection per bus, per unit (Pbusinj and Gs included)
    base: float


def model(mpc: dict) -> Model:
    bus, branch, gen = mpc["bus"], mpc["branch"], mpc["gen"]
    base = float(mpc["baseMVA"])
    n, m = len(bus), len(branch)
    ids = [str(int(b)) for b in bus[:, BUS_I]]
    row = {int(b): i for i, b in enumerate(bus[:, BUS_I])}
    energized = bus[:, BUS_TYPE] != ISOLATED
    f = np.array([row[int(b)] for b in branch[:, F_BUS]])
    t = np.array([row[int(b)] for b in branch[:, T_BUS]])
    on = (branch[:, BR_STATUS] > 0) & energized[f] & energized[t]
    tap = np.where(branch[:, RATIO] == 0, 1.0, branch[:, RATIO])
    b = np.where(on, 1.0 / (branch[:, BR_X] * tap), 0.0)
    Pfinj = -b * np.deg2rad(branch[:, ANGLE])
    k = np.arange(m)
    Bf = sp.csr_matrix((np.r_[b, -b], (np.r_[k, k], np.r_[f, t])), shape=(m, n))
    Cft = sp.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[k, k], np.r_[f, t])), shape=(m, n))
    B = (Cft.T @ Bf).tocsr()
    online = gen[:, GEN_STATUS] > 0
    pg = np.zeros(n)
    np.add.at(pg, [row[int(g)] for g in gen[online, GEN_BUS]], gen[online, PG])
    P = (pg - bus[:, PD] - bus[:, GS]) / base - Cft.T @ Pfinj
    refs = np.flatnonzero((bus[:, BUS_TYPE] == REF) & energized)
    assert len(refs) == 1, f"{len(refs)} reference buses"
    return Model(ids, energized, int(refs[0]), B, Bf, Pfinj, P, base)


def solve(mdl: Model, va_ref_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """The DC solution: angles (degrees, NaN at isolated buses) and from-end
    flows (MW) per branch row."""
    keep = np.flatnonzero(mdl.energized & (np.arange(len(mdl.ids)) != mdl.ref))
    theta = np.full(len(mdl.ids), np.nan)
    theta[mdl.ref] = np.deg2rad(va_ref_deg)
    rhs = mdl.P[keep] - mdl.B[keep][:, [mdl.ref]].toarray().ravel() * theta[mdl.ref]
    theta[keep] = spsolve(mdl.B[keep][:, keep].tocsc(), rhs)
    pf = (mdl.Bf @ np.nan_to_num(theta) + mdl.Pfinj) * mdl.base
    return np.rad2deg(theta), pf


def _positions(mdl: Model, bus_ids: list[str] | None, branch_rows: list[int], m: int):
    """Where each bus (bus row order) and each branch row is in a tool's
    result: an index, or -1 for not reported. None for no angles at all."""
    bpos = None
    if bus_ids is not None:
        at = {b: i for i, b in enumerate(bus_ids)}
        bpos = np.array([at.get(b, -1) for b in mdl.ids])
    at = {int(r): i for i, r in enumerate(branch_rows)}
    return bpos, np.array([at.get(r, -1) for r in range(m)])


def _grade(mdl: Model, pf_ref: np.ndarray, bpos: np.ndarray | None, va_deg: np.ndarray | None, rpos: np.ndarray,
           pf_mw: np.ndarray, tol_mw: float) -> dict:
    """One solution against the model and its reference flows."""
    n = len(mdl.ids)
    angles = bpos is not None
    rows = np.flatnonzero(mdl.energized & (np.arange(n) != mdl.ref))
    if angles:
        theta, reported = np.zeros(n), np.zeros(n, bool)
        vals = np.where(bpos >= 0, va_deg[np.maximum(bpos, 0)], np.nan)
        reported = np.isfinite(vals)
        theta[reported] = np.deg2rad(vals[reported])
        balance = np.abs(mdl.B @ theta - mdl.P)[rows] * mdl.base
    live = np.asarray(abs(mdl.Bf).sum(axis=1)).ravel() > 0   # in service, both ends energized
    flow = np.where(rpos >= 0, pf_mw[np.maximum(rpos, 0)], np.nan)
    have = np.isfinite(flow)
    dflow = np.where(have, np.abs(flow - pf_ref), np.where(live, np.inf, 0.0))
    out = {
        "dc_angles": angles,
        "dc_max_balance_mw": (float(balance.max()) if len(balance) else 0.0) if angles else None,
        "dc_max_dflow_mw": float(dflow.max()) if len(dflow) else 0.0,
        "dc_worst_bus": (mdl.ids[rows[int(np.argmax(balance))]] if len(rows) else "") if angles else None,
        "dc_worst_branch": int(np.argmax(dflow)) if len(dflow) else -1,
        "dc_n_buses": int(mdl.energized.sum()),
        "dc_n_reported_buses": int(reported[mdl.energized].sum()) if angles else None,
        "dc_n_branches": int(live.sum()), "dc_n_reported_branches": int((live & have).sum()),
    }
    ok_buses = not angles or out["dc_n_reported_buses"] == out["dc_n_buses"]
    out["oracle_ok"] = bool(ok_buses and out["dc_n_reported_branches"] == out["dc_n_branches"]
                            and (not angles or out["dc_max_balance_mw"] <= tol_mw) and out["dc_max_dflow_mw"] <= tol_mw)
    return out


def check(mpc: dict, bus_ids: list[str] | None, va_deg: np.ndarray | None, branch_rows: list[int],
          pf_mw: np.ndarray, tol_mw: float) -> dict:
    """One DC solution: `va_deg` per `bus_ids` (MATPOWER numbers as str;
    both None for a flows-only tool), `pf_mw` per `branch_rows` (rows of the
    `.m`'s branch matrix)."""
    mdl = model(mpc)
    _, pf_ref = solve(mdl, float(mpc["bus"][mdl.ref, VA]))
    bpos, rpos = _positions(mdl, bus_ids, branch_rows, len(pf_ref))
    return _grade(mdl, pf_ref, bpos, None if va_deg is None else np.asarray(va_deg, float), rpos,
                  np.asarray(pf_mw, float), tol_mw)


def outage_models(mpc: dict, rows):
    """For every branch row in `rows`, the DC model of the case with that
    branch out of service and its reference flows (MW), from the base case's
    factorization and a rank-one update (see the module docstring)."""
    mdl = model(mpc)
    n = len(mdl.ids)
    keep = np.flatnonzero(mdl.energized & (np.arange(n) != mdl.ref))
    where = np.full(n, -1)
    where[keep] = np.arange(len(keep))
    lu = splu(mdl.B[keep][:, keep].tocsc())
    th_ref = np.deg2rad(float(mpc["bus"][mdl.ref, VA]))
    x0 = lu.solve(mdl.P[keep] - mdl.B[keep][:, [mdl.ref]].toarray().ravel() * th_ref)
    Bf = mdl.Bf.tocsr()
    row = {int(b): i for i, b in enumerate(mpc["bus"][:, BUS_I])}
    for r in rows:
        r = int(r)
        f, t = row[int(mpc["branch"][r, F_BUS])], row[int(mpc["branch"][r, T_BUS])]
        b = float(Bf[r, f])   # signed: a negative reactance gives a negative b
        assert b != 0 and Bf[r, t] == -b, f"branch row {r} is not in service"
        c = np.zeros(n)
        c[f], c[t] = 1.0, -1.0
        B_k = mdl.B - sp.csr_matrix((b * np.array([1.0, -1.0, -1.0, 1.0]), ([f, f, t, t], [f, t, f, t])), shape=(n, n))
        P_k = mdl.P + mdl.Pfinj[r] * c
        Pfinj_k = mdl.Pfinj.copy()
        Pfinj_k[r] = 0.0
        Bf_k = Bf.copy()
        Bf_k.data[Bf_k.indptr[r]:Bf_k.indptr[r + 1]] = 0.0
        Bf_k.eliminate_zeros()
        u = np.zeros(len(keep))
        for j, sign in ((f, 1.0), (t, -1.0)):
            if where[j] >= 0:
                u[where[j]] = sign
        w = lu.solve(u)
        z = x0 + (mdl.Pfinj[r] + b * c[mdl.ref] * th_ref) * w
        theta = np.full(n, np.nan)
        theta[mdl.ref] = th_ref
        theta[keep] = z + b * w * (u @ z) / (1.0 - b * (u @ w))
        pf_ref = (Bf_k @ np.nan_to_num(theta) + Pfinj_k) * mdl.base
        yield Model(mdl.ids, mdl.energized, mdl.ref, B_k, Bf_k, Pfinj_k, P_k, mdl.base), theta, pf_ref


def check_outages(mpc: dict, rows, bus_ids: list[str] | None, va_deg: np.ndarray | None, branch_rows: list[int],
                  pf_mw: np.ndarray, tol_mw: float) -> dict:
    """DC N-1: one solution per outage (outages x buses, outages x branch
    rows), outage k the case with branch row `rows[k]` out; the maxima over
    all of them, the number that failed and the worst."""
    n = len(rows)
    angles = bus_ids is not None
    pf_mw = np.asarray(pf_mw, float)
    assert (not angles or np.shape(va_deg) == (n, len(bus_ids))) and pf_mw.shape == (n, len(branch_rows)), \
        "one row per outage"
    worst, failed, worst_k, worst_v = None, 0, 0, -1.0
    agg = {"dc_max_balance_mw": 0.0 if angles else None, "dc_max_dflow_mw": 0.0}
    bpos = rpos = None
    for k, (mdl_k, _, pf_ref) in enumerate(outage_models(mpc, rows)):
        if rpos is None:
            bpos, rpos = _positions(mdl_k, bus_ids, branch_rows, len(pf_ref))
        r = _grade(mdl_k, pf_ref, bpos, np.asarray(va_deg[k], float) if angles else None, rpos, pf_mw[k], tol_mw)
        failed += not r["oracle_ok"]
        for f in agg:
            if r[f] is not None:
                agg[f] = max(agg[f], r[f])
        v = max(r["dc_max_balance_mw"] or 0.0, r["dc_max_dflow_mw"])
        if worst is None or v > worst_v:
            worst, worst_k, worst_v = r, k, v
    return {**agg, **{f: worst[f] for f in ("dc_angles", "dc_worst_bus", "dc_worst_branch", "dc_n_buses",
                                             "dc_n_reported_buses", "dc_n_branches", "dc_n_reported_branches")},
            "scenarios": n, "scenarios_failed": failed, "worst_scenario": worst_k, "oracle_ok": failed == 0}
