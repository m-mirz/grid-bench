"""Differential checks against independent per-outage component labelling."""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from cases.contingency import _seed, candidates
from cases.matpower import BR_STATUS, BUS_I, BUS_TYPE, F_BUS, ISOLATED, REF, T_BUS


def grid(n, edges, isolated=(), offline=()):
    bus = np.zeros((n, 13))
    bus[:, BUS_I] = np.arange(n) * 3 + 10  # identifiers need not be positions
    bus[:, BUS_TYPE] = 1
    bus[0, BUS_TYPE] = REF
    bus[list(isolated), BUS_TYPE] = ISOLATED
    branch = np.zeros((len(edges), 13))
    branch[:, BR_STATUS] = 1
    for i, (a, b) in enumerate(edges):
        branch[i, F_BUS] = bus[a, BUS_I]
        branch[i, T_BUS] = bus[b, BUS_I]
    branch[list(offline), BR_STATUS] = 0
    return {'bus': bus, 'branch': branch}


def brute(mpc, key):
    bus, branch = mpc['bus'], mpc['branch']
    pos = {int(b): i for i, b in enumerate(bus[:, BUS_I])}
    live = np.flatnonzero(branch[:, BR_STATUS] > 0)
    f = np.array([pos[int(b)] for b in branch[live, F_BUS]], dtype=int)
    t = np.array([pos[int(b)] for b in branch[live, T_BUS]], dtype=int)
    energized = bus[:, BUS_TYPE] != ISOLATED
    slack = np.flatnonzero(bus[:, BUS_TYPE] == REF)[0]
    result = []
    for j in np.random.default_rng(_seed(key)).permutation(len(live)):
        keep = np.arange(len(live)) != j
        graph = sp.coo_matrix((np.ones(keep.sum()), (f[keep], t[keep])), shape=(len(bus), len(bus)))
        _, label = connected_components(graph, directed=False)
        if (label[energized] == label[slack]).all():
            result.append(int(live[j]))
    return result


@pytest.mark.parametrize('seed', range(100))
def test_random_multigraph_matches_brute_in_exact_order(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 30))
    edges = [tuple(e) for e in rng.integers(n, size=(int(rng.integers(0, 90)), 2))]
    isolated = [i for i in range(1, n) if rng.random() < .25]
    offline = [i for i in range(len(edges)) if rng.random() < .2]
    mpc = grid(n, edges, isolated, offline)
    assert list(candidates(mpc, str(seed))) == brute(mpc, str(seed))


@pytest.mark.parametrize('n,edges,isolated,offline,expected', [
    (2, [(0, 1), (0, 1)], (), (), {0, 1}),  # parallel lines are not bridges
    (3, [(0, 1), (1, 2)], (), (), set()),
    (3, [(0, 1), (1, 2)], (2,), (), {1}),  # disconnected inactive leaf allowed
    (4, [(0, 1), (2, 3)], (2, 3), (), {1}),  # inactive component irrelevant
    (3, [(0, 1), (1, 2)], (), (1,), set()),  # initially disconnected active node
    (2, [(0, 1), (0, 0), (1, 1)], (), (), {1, 2}),  # self loops
    (2, [], (), (), set()),
])
def test_topological_boundaries(n, edges, isolated, offline, expected):
    mpc = grid(n, edges, isolated, offline)
    actual = list(candidates(mpc, 'boundary'))
    assert set(actual) == expected
    assert actual == brute(mpc, 'boundary')


def test_deep_radial_grid_does_not_use_recursion():
    n = 30000
    mpc = grid(n, [(i, i + 1) for i in range(n - 1)])
    assert list(candidates(mpc, 'deep')) == []
