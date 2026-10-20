"""Manual benchmark of candidate filtering; no power-flow solver is timed.

Run from the repository root, with data/benchmark-grids initialized:
    PYTHONPATH=. python tests/benchmark_contingency_candidates.py
Use --full to also enumerate all candidates (the old path can take minutes).
Every timed pair must return exactly the same branch rows in the same order.
"""
import argparse
import itertools
import json
import platform
import statistics
import time

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from cases.contingency import _seed, candidates
from cases.matpower import BR_STATUS, BUS_I, BUS_TYPE, F_BUS, ISOLATED, REF, T_BUS, parse_m
from cases.registry import CASES


def original_candidates(mpc, key):
    """Original per-outage component check, retained as the timing baseline."""
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
        graph = sp.coo_matrix((np.ones(keep.sum()), (f[keep], t[keep])), shape=(len(bus), len(bus)))
        _, label = connected_components(graph, directed=False)
        if (label[energized] == label[slack]).all():
            yield int(live[j])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["case14", "case1354pegase", "case9241pegase"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(json.dumps({"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}))
    for case in args.cases:
        mpc = parse_m(CASES[case]["file"])
        for limit in ([200, None] if args.full else [200]):
            timings = {"before": [], "after": []}
            methods = [("before", original_candidates), ("after", candidates)]
            for rep in range(args.repeats):
                outputs = {}
                for name, method in (methods if rep % 2 == 0 else methods[::-1]):
                    start = time.perf_counter()
                    outputs[name] = list(itertools.islice(method(mpc, case + "#n1"), limit))
                    timings[name].append(time.perf_counter() - start)
                assert outputs["before"] == outputs["after"], f"candidate/order mismatch: {case}"
            print(json.dumps({"case": case, "limit": limit, "candidates": len(outputs["after"]),
                              "same_order": True, "seconds": timings,
                              "median_seconds": {k: statistics.median(v) for k, v in timings.items()}}), flush=True)


if __name__ == "__main__":
    main()
