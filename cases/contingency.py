"""N-1 outage sets for the contingency cases, chosen from the case with no
tool involved.

A contingency case (`<case>#n1`) is a MATPOWER case plus `N_OUTAGES`
single-branch outages: each scenario is the case with one in-service branch
taken out of service. Every tool solves the base case and every outage in
one timed call (`adapters.batch_adapter`, problem "n1"), each outage
starting from the tool's own solution of the base case, as contingency
analysis is done (the one exception to the flat start of rule 3).

Which branches: in seeded random order (`numpy.random.default_rng` seeded
from the case key, as in `cases.measurements`), the first `N_OUTAGES`
in-service branches whose outage

- does not island the grid: every energized bus stays connected to the
  slack (an islanding outage is a different problem, a split network, that
  tools handle differently: skipped, masked, or an error), and
- leaves a case whose power flow converges: `cases.prep` solves each
  candidate with `cases.truth.solve_pf` from the base-case solution and
  checks it with tier 1; a candidate that does not converge is skipped
  (deterministically, and counted in the file), so every outage provably
  has a solution near the base case.

A radial grid has no outage that keeps it connected, so contingency cases
are transmission cases only.

Written as `.npz` (`contingency_path`): `branch_row` (rows of the `.m`'s
branch matrix, in scenario order), `from_bus`, `to_bus` (for each adapter
to assert its join), and `n_skipped` (candidates dropped because their
power flow did not converge).
"""
import hashlib

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from cases.matpower import BR_STATUS, BUS_I, BUS_TYPE, F_BUS, ISOLATED, REF, T_BUS

N_OUTAGES = 200


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def candidates(mpc: dict, key: str):
    """Yields the in-service branch rows whose outage keeps every energized
    bus connected to the slack, in the case's seeded order."""
    bus, branch = mpc["bus"], mpc["branch"]
    pos = {int(b): i for i, b in enumerate(bus[:, BUS_I])}
    energized = bus[:, BUS_TYPE] != ISOLATED
    live = np.flatnonzero(branch[:, BR_STATUS] > 0)
    f = np.array([pos[int(b)] for b in branch[live, F_BUS]])
    t = np.array([pos[int(b)] for b in branch[live, T_BUS]])
    slack = int(np.flatnonzero(bus[:, BUS_TYPE] == REF)[0])
    for j in np.random.default_rng(_seed(key)).permutation(len(live)):
        keep = np.ones(len(live), bool)
        keep[j] = False
        g = sp.coo_matrix((np.ones(keep.sum()), (f[keep], t[keep])), shape=(len(bus), len(bus)))
        _, label = connected_components(g, directed=False)
        if (label[energized] == label[slack]).all():
            yield int(live[j])


def outage(mpc: dict, data: dict, k: int) -> dict:
    """The case with outage k's branch out of service: the problem every
    tool is asked to solve for that scenario."""
    out = {name: (v.copy() if isinstance(v, np.ndarray) else v) for name, v in mpc.items()}
    out["branch"][int(data["branch_row"][k]), BR_STATUS] = 0
    return out


def write(data: dict, path) -> None:
    with open(path, "wb") as f:
        np.savez(f, **data)


def read(path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return dict(z)
