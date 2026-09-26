"""Bus admittance matrix, following MATPOWER's `makeYbus.m` exactly.

This is the measuring stick every tool is checked against, so it depends on
numpy/scipy only and must never import a tool under test.

Conventions (MATPOWER manual, "Branch model"): pi-model per branch, series
admittance ys = 1/(r + jx), total line charging b split half to each end, a
complex tap t = ratio * exp(j*angle) on the *from* side, `ratio == 0` meaning
"no transformer" (unity tap). Bus shunts Gs/Bs are in MW/MVAr at 1 p.u.
voltage and go on the diagonal. Out-of-service branches are skipped.

Branch SHIFT (column 10) is in degrees with MATPOWER's sign: t's angle is
+SHIFT, so a positive SHIFT makes the from side lead. RATIO (column 9) is the
from-side turns ratio as written, never inverted. No per-case exception, the
PEGASE cases included. Do not infer the convention from a case's bus VM/VA
columns: on case1354pegase and case9241pegase they fit "radians, opposite
sign" best but solve the equations under no reading (best fit still 39 MW /
272 MW mismatch), whereas MATPOWER's own runpf under the convention above
closes them to 1e-9 MW. Those columns are never used as a reference here.
"""
import numpy as np
import scipy.sparse as sp

from cases.matpower import ANGLE, BR_B, BR_R, BR_STATUS, BR_X, BS, BUS_I, F_BUS, GS, RATIO, T_BUS


def _branches(mpc: dict, zero_phase_shifts: bool) -> tuple:
    """In-service branches: their rows in `mpc["branch"]`, from/to bus
    positions, and the four pi-model admittances."""
    bus, branch = mpc["bus"], mpc["branch"]
    pos = {int(b): i for i, b in enumerate(bus[:, BUS_I])}
    rows = np.flatnonzero(branch[:, BR_STATUS] != 0)
    br = branch[rows]
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
    return rows, f, t, yff, yft, ytf, ytt


def make_yf(mpc: dict) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    """Returns `(branch_rows, from_pos, Yf)`: row `k` of Yf gives the current
    into in-service branch `branch_rows[k]` (its row in `mpc["branch"]`) at
    its from bus, `from_pos[k]`, as `Yf @ V`. MATPOWER's `makeYbus.m` Yf."""
    rows, f, t, yff, yft, _, _ = _branches(mpc, False)
    k = np.arange(len(rows))
    yf = sp.csr_matrix((np.r_[yff, yft], (np.r_[k, k], np.r_[f, t])), shape=(len(rows), len(mpc["bus"])))
    return rows, f, yf


def make_ybus(mpc: dict, zero_phase_shifts: bool = False) -> tuple[np.ndarray, sp.csr_matrix]:
    """Returns `(bus_ids, Ybus)`; row `i` of Ybus belongs to `bus_ids[i]`.

    `zero_phase_shifts` is a diagnostic, not a correctness option: it rebuilds
    Ybus with every branch angle forced to 0, which isolates the one known
    lossy conversion (a tool that cannot represent continuous phase shift).
    """
    bus = mpc["bus"]
    base_mva = float(mpc["baseMVA"])
    ids = bus[:, BUS_I].astype(int)
    _, f, t, yff, yft, ytf, ytt = _branches(mpc, zero_phase_shifts)

    n = len(ids)
    ysh = (bus[:, GS] + 1j * bus[:, BS]) / base_mva
    rows = np.concatenate([f, f, t, t, np.arange(n)])
    cols = np.concatenate([f, t, f, t, np.arange(n)])
    vals = np.concatenate([yff, yft, ytf, ytt, ysh])
    return ids, sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
