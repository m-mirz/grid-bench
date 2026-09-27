"""Grades an AC optimal-power-flow solution against its case, with no tool
and no optimizer of its own.

A tool reports bus voltages and the dispatch of every online generator
(keyed by its row in the `.m`'s gen matrix). The oracle rebuilds the
network from the `.m` (`oracle.ybus`) and checks, independently of how the
tool modelled anything:

- balance: `V * conj(Ybus @ V)` at every energized bus equals the tool's
  own dispatch minus the load. This is the tier-1 residual of the power
  flow with the dispatch as the specified injection: a dispatch that the
  voltages do not carry is not a solution.
- limits, each as the largest violation over the case: |V| against
  VMIN/VMAX, generator P and Q against PMIN/PMAX, QMIN/QMAX, apparent power
  at both branch ends against RATE_A (0: unlimited), and the angle
  difference across each branch against ANGMIN/ANGMAX (MATPOWER's rule:
  unconstrained at -360/360).
- objective: the polynomial cost (`gencost` model 2) of the reported
  dispatch, recomputed here; what a tool reports as its cost is not used.
- gap: (objective - reference) / reference, the reference being PGLib's
  published AC objective (`cases.pglib`), a local optimum given to five
  significant digits.

`oracle_ok`: every energized bus and online generator reported, balance
and limits within tolerance, and the gap at most `GAP_OK`. A feasible
solution cheaper than the reference by more than `GAP_OK` also passes, and
is flagged (`opf_better_than_reference`): with a nonconvex problem, another
local optimum is possible, and cheaper-and-feasible is checked, not assumed.

Checked against an independent solve (tests/test_opf.py: SLSQP with
analytic Jacobians, written only for the tests): on pglib case14 (typical,
congested, small angle difference) and case118 it converges to 1e-12 MVA of
balance with every limit met, and lands 4e-6 to 9e-6 below PGLib's
reference, inside its five-digit rounding.
"""
from dataclasses import asdict, dataclass

import numpy as np

from cases.matpower import (ANGMAX, ANGMIN, BUS_TYPE, COST, GEN_BUS, GEN_STATUS, ISOLATED, MODEL, NCOST, PD,
                            PMAX, PMIN, QD, QMAX, QMIN, RATE_A, VMAX, VMIN)
from oracle.ybus import make_ybus, make_yft

BALANCE_OK_MVA = 1e-3     # as the power-flow oracle; tools are asked for 1e-6 p.u. (1e-4 MVA on 100 MVA)
LIMIT_OK_PU = 1e-5        # |V|, and P/Q/S limits divided by base power; interior-point solutions sit inside by ~tol
ANGLE_OK_DEG = 1e-3
GAP_OK = 1e-4             # above the 5e-5 the reference's five digits can resolve


@dataclass
class OpfCheck:
    n_buses: int
    n_reported_buses: int
    n_gens: int
    n_reported_gens: int
    max_dp_mw: float = np.nan
    max_dq_mvar: float = np.nan
    max_vm_violation_pu: float = np.nan
    max_pg_violation_mw: float = np.nan
    max_qg_violation_mvar: float = np.nan
    max_flow_violation_mva: float = np.nan
    max_angle_violation_deg: float = np.nan
    objective: float = np.nan
    reference: float = np.nan
    gap: float = np.nan

    def asdict(self) -> dict:
        return {f"opf_{k}": v for k, v in asdict(self).items()}


def _over(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.maximum(np.maximum(lo - x, x - hi), 0.0).max(initial=0.0))


def cost(gencost: np.ndarray, pg_mw: np.ndarray) -> float:
    """Sum of the polynomial costs, coefficients highest order first."""
    if len(gencost) != len(pg_mw):
        raise ValueError("gencost rows must be one per online generator (no reactive-power costs)")
    if (gencost[:, MODEL] != 2).any():
        raise ValueError("only polynomial generator costs (gencost model 2) are graded")
    total = 0.0
    for row, p in zip(gencost, pg_mw):
        n = int(row[NCOST])
        total += float(np.polyval(row[COST:COST + n], p))
    return total


def check(mpc: dict, reference: float, vm: dict[str, float], va_deg: dict[str, float],
          pg_mw: dict[str, float], qg_mvar: dict[str, float]) -> dict:
    """Flat fields for a result record's `extra_info` (see the module docstring)."""
    bus, gen, branch = mpc["bus"], mpc["gen"], mpc["branch"]
    base = float(mpc["baseMVA"])
    ids, ybus = make_ybus(mpc)
    keys = [str(b) for b in ids]
    energized = bus[:, BUS_TYPE] != ISOLATED
    on = np.flatnonzero(gen[:, GEN_STATUS] > 0)
    got_bus = [k for k, e in zip(keys, energized) if e and k in vm and k in va_deg]
    got_gen = [g for g in on if str(g) in pg_mw and str(g) in qg_mvar]
    c = OpfCheck(int(energized.sum()), len(got_bus), len(on), len(got_gen), reference=reference)
    if c.n_reported_buses < c.n_buses or c.n_reported_gens < c.n_gens:
        return c.asdict() | {"oracle_ok": False, "opf_better_than_reference": False}

    pos = {b: i for i, b in enumerate(keys)}
    v = np.array([vm[k] * np.exp(1j * np.deg2rad(va_deg[k])) if e else 0 for k, e in zip(keys, energized)])
    pg = np.array([pg_mw[str(g)] for g in on])
    qg = np.array([qg_mvar[str(g)] for g in on])
    s_spec = -(bus[:, PD] + 1j * bus[:, QD])
    np.add.at(s_spec, [pos[str(int(b))] for b in gen[on, GEN_BUS]], pg + 1j * qg)
    ds = np.where(energized, v * np.conj(ybus @ v) * base - s_spec, 0)
    c.max_dp_mw, c.max_dq_mvar = float(np.abs(ds.real).max()), float(np.abs(ds.imag).max())

    vmag = np.abs(v)[energized]
    c.max_vm_violation_pu = _over(vmag, bus[energized, VMIN], bus[energized, VMAX])
    c.max_pg_violation_mw = _over(pg, gen[on, PMIN], gen[on, PMAX])
    c.max_qg_violation_mvar = _over(qg, gen[on, QMIN], gen[on, QMAX])

    rows, f, t, yf, yt = make_yft(mpc)
    s_f = np.abs(v[f] * np.conj(yf @ v)) * base
    s_t = np.abs(v[t] * np.conj(yt @ v)) * base
    rate = branch[rows, RATE_A]
    limited = rate > 0
    c.max_flow_violation_mva = float(np.maximum(np.maximum(s_f, s_t) - rate, 0.0)[limited].max(initial=0.0))
    dtheta = np.rad2deg(np.angle(v[f] * np.conj(v[t])))
    lo, hi = branch[rows, ANGMIN], branch[rows, ANGMAX]
    bounded = (lo > -360) | (hi < 360)
    c.max_angle_violation_deg = _over(dtheta[bounded], lo[bounded], hi[bounded])

    c.objective = cost(mpc["gencost"][on], pg)
    c.gap = (c.objective - reference) / reference
    feasible = (max(c.max_dp_mw, c.max_dq_mvar) <= BALANCE_OK_MVA
                and c.max_vm_violation_pu <= LIMIT_OK_PU
                and max(c.max_pg_violation_mw, c.max_qg_violation_mvar, c.max_flow_violation_mva) <= LIMIT_OK_PU * base
                and c.max_angle_violation_deg <= ANGLE_OK_DEG)
    return c.asdict() | {"oracle_ok": bool(feasible and c.gap <= GAP_OK),
                         "opf_better_than_reference": bool(feasible and c.gap < -GAP_OK)}
