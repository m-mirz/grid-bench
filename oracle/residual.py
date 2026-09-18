"""Absolute correctness check for a solved MATPOWER case, with no tool used as
the reference.

Cross-tool comparison can only say how far tools drift from *each other*.
This instead evaluates a tool's reported voltages against the original power
flow equations, with a Ybus built independently from the `.m` file:

    dS = V * conj(Ybus @ V) - S_specified

A solution that actually solves the stated problem drives this to zero.
Which components are meaningful depends on bus type:

- PQ bus: P and Q are both specified, so both residuals must vanish.
- PV bus: only P is specified (Q is a free variable of the solve); instead,
  |V| must equal the generator setpoint.
- Slack bus: neither P nor Q is specified; only the |V| setpoint is checked.

The residual is invariant to a global rotation of all angles, so tools that
pick different angle references need no alignment.

Because a bus's injection depends only on its own voltage and its Ybus
neighbours', a tool that omits some buses (e.g. only reports the main
connected component) is still checked on every bus whose whole neighbourhood
it did report; `n_checked` says how many that was.

Ported and extended from gridoxide's `scripts/bench/check_matpower_residual.py`,
which found four real conversion bugs that a five-tool cross-comparison had
completely missed.
"""
from dataclasses import asdict, dataclass

import numpy as np

from cases.matpower import BUS_TYPE, GEN_BUS, GEN_STATUS, ISOLATED, PD, PG, PQ, PV, QD, QG, REF, VG
from oracle.ybus import make_ybus


@dataclass
class Residual:
    max_dp_mw: float        # over PQ and PV buses
    max_dq_mvar: float      # over PQ buses
    max_dvm_pu: float       # |V| vs generator setpoint, over PV and slack buses
    worst_bus: str          # bus with the largest |dP| or |dQ|
    n_checked: int          # buses whose full neighbourhood the tool reported
    n_buses: int            # energized buses in the case
    failing: dict           # {bus class: count} of buses over tolerance, see classify()

    def asdict(self) -> dict:
        return asdict(self)


def specified_injections(mpc: dict, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Net specified injection per bus in p.u., and |V| setpoint (NaN if none)."""
    bus, gen = mpc["bus"], mpc["gen"]
    base_mva = float(mpc["baseMVA"])
    pos = {b: i for i, b in enumerate(ids)}
    s = -(bus[:, PD] + 1j * bus[:, QD]) / base_mva
    vset = np.full(len(ids), np.nan)
    for g in gen[gen[:, GEN_STATUS] > 0]:
        i = pos[int(g[GEN_BUS])]
        s[i] += (g[PG] + 1j * g[QG]) / base_mva
        if np.isnan(vset[i]):
            vset[i] = g[VG]
    return s, vset


def effective_bus_types(mpc: dict, vset: np.ndarray) -> np.ndarray:
    """Bus types as MATPOWER's `bustypes.m` resolves them: a PV bus with no
    online generator is solved as PQ (case3120sp has 101 of these, and
    case2848rte 25), so its Q is specified and must be checked."""
    bus_type = mpc["bus"][:, BUS_TYPE].astype(int)
    return np.where((bus_type == PV) & np.isnan(vset), PQ, bus_type)


def classify(mpc: dict, ids: np.ndarray, vset: np.ndarray) -> np.ndarray:
    """A label per bus naming the MATPOWER bus-type rule that applies to it.
    Tools that break one of these rules fail on exactly that class of bus,
    which is what the report needs to say (e.g. "an offline generator still
    regulates voltage"), rather than only which bus was worst.

    - pv_without_online_gen: typed PV, but every generator is offline, so
      MATPOWER solves it as PQ (a tool holding its voltage keeps an offline
      generator regulating).
    - pq_with_online_gen: typed PQ with an online generator, which MATPOWER
      treats as a fixed P/Q injection (a tool holding its voltage makes the
      generator regulate against the case).
    - pq_with_offline_gen: typed PQ with only offline generators.
    - pq, pv, ref: everything else, by type.
    """
    raw = mpc["bus"][:, BUS_TYPE].astype(int)
    gen = mpc["gen"]
    pos = {b: i for i, b in enumerate(ids)}
    online = np.zeros(len(ids), bool)
    offline = np.zeros(len(ids), bool)
    for g in gen:
        (online if g[GEN_STATUS] > 0 else offline)[pos[int(g[GEN_BUS])]] = True
    label = np.select([raw == REF, raw == PV], ["ref", "pv"], "pq").astype(object)
    label[(raw == PV) & np.isnan(vset)] = "pv_without_online_gen"
    label[(raw == PQ) & online] = "pq_with_online_gen"
    label[(raw == PQ) & ~online & offline] = "pq_with_offline_gen"
    return label


def residual(mpc: dict, vm_pu: dict[str, float], va_deg: dict[str, float],
             zero_phase_shifts: bool = False, tol_mva: float = 1e-3, tol_pu: float = 1e-6) -> Residual:
    """Evaluates a solution keyed by MATPOWER bus number (as `str`)."""
    ids, ybus = make_ybus(mpc, zero_phase_shifts)
    base_mva = float(mpc["baseMVA"])
    s_spec, vset = specified_injections(mpc, ids)
    bus_type = effective_bus_types(mpc, vset)
    energized = bus_type != ISOLATED

    keys = [str(b) for b in ids]
    known = np.array([k in vm_pu and k in va_deg for k in keys])
    vm = np.array([vm_pu.get(k, np.nan) for k in keys])
    va = np.deg2rad(np.array([va_deg.get(k, np.nan) for k in keys]))
    v = np.where(known, vm * np.exp(1j * va), 0.0)

    # A bus is checkable iff it and every Ybus neighbour has a reported voltage.
    missing_neighbour = (abs(ybus) @ (~known).astype(float)) > 0
    checkable = energized & known & ~missing_neighbour

    ds = (v * np.conj(ybus @ v) - s_spec) * base_mva
    p_mask = checkable & (bus_type != REF)
    q_mask = checkable & (bus_type == PQ)
    v_mask = checkable & np.isin(bus_type, (PV, REF)) & ~np.isnan(vset)

    dp = np.where(p_mask, np.abs(ds.real), 0.0)
    dq = np.where(q_mask, np.abs(ds.imag), 0.0)
    dvm = np.where(v_mask, np.abs(vm - np.nan_to_num(vset)), 0.0)
    worst = int(np.argmax(np.maximum(dp, dq)))
    over = (np.maximum(dp, dq) > tol_mva) | (dvm > tol_pu)
    labels, counts = np.unique(classify(mpc, ids, vset)[over], return_counts=True)
    return Residual(
        max_dp_mw=float(dp.max(initial=0.0)),
        max_dq_mvar=float(dq.max(initial=0.0)),
        max_dvm_pu=float(dvm.max(initial=0.0)),
        worst_bus=keys[worst],
        n_checked=int(checkable.sum()),
        n_buses=int(energized.sum()),
        failing={str(k): int(c) for k, c in zip(labels, counts)},
    )
