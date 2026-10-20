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

A full DC N-1 case (`<case>#dcn1-full`) takes every candidate instead, in
the same order and with no power-flow check (a connected DC model always
solves): thousands per case (case1354pegase 1430, case2869pegase 3804,
case9241pegase 14384). A full AC N-1 case (`<case>#n1-full`, for the GPU
tabs) takes every candidate whose power flow converges, the check above.

A radial grid has no outage that keeps it connected, so contingency cases
are transmission cases only.

Written as `.npz` (`contingency_path`): `branch_row` (rows of the `.m`'s
branch matrix, in scenario order), `from_bus`, `to_bus` (for each adapter
to assert its join), and `n_skipped` (candidates dropped because their
power flow did not converge).
"""
import hashlib

import numpy as np

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
    critical, reached = _critical_edges(len(bus), f, t, slack, energized)
    if not reached[energized].all():
        return
    for j in np.random.default_rng(_seed(key)).permutation(len(live)):
        if not critical[j]:
            yield int(live[j])


def _critical_edges(n, f, t, slack, energized):
    """One iterative low-link traversal: bridges cutting off energized buses.

    Edge identities, rather than parent vertices, distinguish parallel lines.
    A bridge to a subtree containing only isolated buses is not critical:
    removing it leaves every energized bus connected to the reference.
    """
    adjacent = [[] for _ in range(n)]
    for e, (a, b) in enumerate(zip(f, t)):
        adjacent[a].append((b, e))
        adjacent[b].append((a, e))
    entered = np.full(n, -1, dtype=int)
    low = np.zeros(n, dtype=int)
    active = energized.astype(int).copy()
    critical = np.zeros(len(f), dtype=bool)
    entered[slack] = low[slack] = 0
    clock = 1
    # Frames retain adjacency position so path depth cannot exhaust Python recursion.
    stack = [(slack, -1, -1, iter(adjacent[slack]))]
    while stack:
        v, parent, parent_edge, neighbours = stack[-1]
        nxt = next(neighbours, None)
        if nxt is None:
            stack.pop()
            if parent_edge >= 0:
                critical[parent_edge] = low[v] > entered[parent] and active[v] > 0
                low[parent] = min(low[parent], low[v])
                active[parent] += active[v]
            continue
        w, e = nxt
        if e == parent_edge:
            continue
        if entered[w] < 0:
            entered[w] = low[w] = clock
            clock += 1
            stack.append((w, v, e, iter(adjacent[w])))
        else:
            low[v] = min(low[v], entered[w])
    return critical, entered >= 0


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
