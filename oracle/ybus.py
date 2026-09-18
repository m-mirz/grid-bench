"""Bus admittance matrix, following MATPOWER's `makeYbus.m` exactly.

This is the measuring stick every tool is checked against, so it depends on
numpy/scipy only and must never import a tool under test.

Conventions (MATPOWER manual, "Branch model"): pi-model per branch, series
admittance ys = 1/(r + jx), total line charging b split half to each end, a
complex tap t = ratio * exp(j*angle) on the *from* side, `ratio == 0` meaning
"no transformer" (unity tap). Bus shunts Gs/Bs are in MW/MVAr at 1 p.u.
voltage and go on the diagonal. Out-of-service branches are skipped.
"""
import numpy as np
import scipy.sparse as sp

from cases.matpower import ANGLE, BR_B, BR_R, BR_STATUS, BR_X, BS, BUS_I, F_BUS, GS, RATIO, T_BUS


def make_ybus(mpc: dict, zero_phase_shifts: bool = False) -> tuple[np.ndarray, sp.csr_matrix]:
    """Returns `(bus_ids, Ybus)`; row `i` of Ybus belongs to `bus_ids[i]`.

    `zero_phase_shifts` is a diagnostic, not a correctness option: it rebuilds
    Ybus with every branch angle forced to 0, which isolates the one known
    lossy conversion (a tool that cannot represent continuous phase shift).
    """
    bus, branch = mpc["bus"], mpc["branch"]
    base_mva = float(mpc["baseMVA"])
    ids = bus[:, BUS_I].astype(int)
    pos = {b: i for i, b in enumerate(ids)}

    br = branch[branch[:, BR_STATUS] != 0]
    f = np.array([pos[int(b)] for b in br[:, F_BUS]], dtype=int)
    t = np.array([pos[int(b)] for b in br[:, T_BUS]], dtype=int)
    ys = 1.0 / (br[:, BR_R] + 1j * br[:, BR_X])
    bc = br[:, BR_B]
    tap = np.where(br[:, RATIO] != 0, br[:, RATIO], 1.0).astype(complex)
    if not zero_phase_shifts:
        tap = tap * np.exp(1j * np.deg2rad(br[:, ANGLE]))

    ytt = ys + 1j * bc / 2
    yff = ytt / (tap * np.conj(tap))
    yft = -ys / np.conj(tap)
    ytf = -ys / tap

    n = len(ids)
    ysh = (bus[:, GS] + 1j * bus[:, BS]) / base_mva
    rows = np.concatenate([f, f, t, t, np.arange(n)])
    cols = np.concatenate([f, t, f, t, np.arange(n)])
    vals = np.concatenate([yff, yft, ytf, ytt, ysh])
    return ids, sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
