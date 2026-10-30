"""gridoxide (GPU): N-1 contingency analysis through
`GpuBatchSolver.solve_contingencies` (`native`, one GPU).

The batch adapter's method (adapters/gridoxide_gpu_batch_adapter.py) on a
case's outages. An outage that keeps the network connected changes only
the Y-bus values at the outaged branch's positions, not its pattern, so
every outage is a member of the same uniform batch: each carries its
changed values as a delta on the intact Y-bus, which two kernels apply to
the rows it touches. (An outage that splits the network would go to the
CPU; no outage here does, `cases.contingency`.)

Input and joins: the batch adapter's model and session. Outages are joined
by branch row: `branch_rows()` maps gridoxide's flat branch order (lines,
then transformers) to rows of the `.m`'s branch matrix, and each outage is
the flat branch at its row, one branch per contingency, in the case's
order; the row's buses are asserted against the outage file's.

Settings: the batch adapter's. The timed call is the base case from flat
start, `solve_contingencies` and `voltages()`. `capacity=1024`:
case1354pegase#n1-full is two chunks, case2869pegase's four,
case9241pegase's fifteen. An outage that does not converge raises
DidNotConverge with the count.

Results (A100-SXM4-40GB, gridoxide d105bf4, cuDSS 1.0.0; ms per outage,
the batch adapter's comparison): every outage of every case accepted, the
only GPU tool here of which that holds.

    case                     gridoxide  gpusim2grid       ExaPF (GPU)          (lightsim2grid)
    case1354pegase#n1        0.58       0.51              0.52                 (0.13)
    case2869pegase#n1        1.37       160 NaN           167 not converged    (0.25)
    case9241pegase#n1        4.83       3.65              6.86                 (1.43)
    case1354pegase#n1-full   0.31       0.11, 148 failed  0.24                 (0.091)
    case2869pegase#n1-full   0.61       0.26, 1 failed    out of memory        (0.27)
    case9241pegase#n1-full   2.24       0.86, 16 failed   out of memory        (1.26)

gpusim2grid and ExaPF.jl both run cuDSS 0.8.0, and both fail
case2869pegase#n1, which gridoxide on 1.0.0 solves whole. Where gpusim2grid
is accepted it is 1.1-2.6x faster, for the batch adapter's reasons; 30 CPU
threads are faster than gridoxide on every N-1 case (1.8-5.4x).
"""
import numpy as np

from adapters.batch_adapter import BatchSolution, ContingencyAdapter, base_case, outages
from adapters.gridoxide_gpu_adapter import CAPACITY, GridoxideGpu, session
from adapters.solver_adapter import DidNotConverge
from cases.matpower import F_BUS, T_BUS, parse_m
from cases.registry import CASES


class GridoxideGpuN1(GridoxideGpu, ContingencyAdapter):
    mode = "native"
    threaded = False
    settings = GridoxideGpu.settings | {"mode": "native", "batch_api": "GpuBatchSolver.solve_contingencies",
                                        "entry_point": "GpuBatchSolver.solve_contingencies",
                                        "start": "base-case solution", "capacity": CAPACITY}

    def load(self, case):
        base = base_case(case)
        model, gpu, ids = session(base)
        out = outages(case)
        flat = {row: k for k, row in enumerate(model.branch_rows())}
        branch = parse_m(CASES[base]["file"])["branch"]
        rows = out["branch_row"].astype(int)
        assert np.array_equal(branch[rows, F_BUS].astype(int), out["from_bus"].astype(int))
        assert np.array_equal(branch[rows, T_BUS].astype(int), out["to_bus"].astype(int))
        contingencies = [[flat[int(r)]] for r in rows]
        return {"model": model, "gpu": gpu, "ids": ids, "contingencies": contingencies, "v": None}

    def solve(self, model, threads=1):
        assert threads == 1, "one GPU: threads are not a setting"
        model["model"].solve(init="flat")
        results = model["gpu"].solve_contingencies(model["contingencies"])
        failed = sum(r.status != "converged" for r in results)
        if failed:
            raise DidNotConverge(f"{failed} of {len(results)} outages not below tolerance")
        model["v"] = model["gpu"].voltages()

    def solution(self, model, case):
        n, ids = len(model["contingencies"]), model["ids"]
        vm, va = (np.frombuffer(b).reshape(n, len(ids)) for b in model["v"])
        return BatchSolution(ids, vm, np.rad2deg(va))
