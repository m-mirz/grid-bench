"""gpusim2grid: batch power flow through `InjectionSweepGPU` (`native`, one
GPU).

gpusim2grid's injection sweep starts every scenario from the base case's
solution, the problem's start: the session takes the base case's converged
voltage at construction and starts each scenario there
(`init_from_n_powerflow`).

Input and joins: lightsim2grid's (`adapters/lightsim2grid_batch_adapter.py`,
`injections`): the same grid from `init_from_matpower`, the scenarios'
generator P and load P/Q per element, which gpusim2grid assembles into bus
injections itself (`set_injections_from_elements`, in `load`). Bus ids:
the power-flow adapter's (AC-solver numbering, asserted).

Settings:
- The timed call is lightsim2grid's solve of the base case from its flat
  start (`ac_pf(v_flat, MAX_ITERATIONS, TOLERANCE_PU)`, gpusim2grid's own
  first step), then `compute` and the voltages' copy to the host. The
  session's scenarios start from that same solution, computed the same way
  when the session was built (`init_from_n_powerflow=True`, its default),
  since it takes its start only then; repeating the solve in the call
  counts the base case, as the problem asks.
- `nb_iter=4` (its default): a fixed number of Newton steps per scenario,
  with no convergence test of its own. Not tuned per case: the oracle
  grades every scenario, and one that 4 steps do not bring to the
  tolerance shows there.
- `batch_size=512` (its default): every sweep here (100 scenarios) is one
  chunk on the device.
- FP64, default cuDSS settings, no damping, distributed-slack formulation
  with the case's one slack: the power-flow adapter's.
- Thread count: not a setting (one GPU).

Results (A100, 416ae9f, 100 scenarios): every scenario accepted on the
transmission sweeps, 0.96 ms per scenario on case1354pegase, 2.1 ms on
case2869pegase, 7.0 ms on case9241pegase (ExaPF.jl's batch on the same GPU
0.86, 2.0 and 10.2 ms; lightsim2grid on 30 threads of the same machine
0.27, 0.65 and 3.3 ms), and on mvlv10616 (6.4 ms). mvlv1004#sweep
is rejected: 4 Newton steps from the base case leave 25 of its 100
scenarios at up to 3.3e-3 MVA, above the oracle's 1e-3 (heavily loaded
feeders, down to 0.67 p.u.): the fixed step count, reported, not tuned.

Beyond 100 scenarios (a separate experiment, the same adapter, sweeps of
1 000 to 50 000 scenarios, every one graded): the time per scenario falls
3.7 to 4.4x by 1 000 scenarios and then stays (case9241pegase 1.72 ms at
50 000; lightsim2grid on 30 threads 2.0 ms from the base case at 10 000).
On case1354pegase, clean at 100 scenarios, 21 to 29 % of the scenarios
come back NaN from 1 000 on, the batch_size=512 chunks hitting the cuDSS
fault documented in the N-1 adapter; mvlv10616 and case9241pegase stay clean.
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case
from adapters.gpusim2grid_adapter import Gpusim2gridAdapter, bus_ids, solved_grid
from adapters.lightsim2grid_batch_adapter import injections
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge

NB_ITER = 4
BATCH_SIZE = 512


class Gpusim2gridBatch(BatchAdapter):
    name = Gpusim2gridAdapter.name
    display_name = Gpusim2gridAdapter.display_name
    color = Gpusim2gridAdapter.color
    package = Gpusim2gridAdapter.package
    language = Gpusim2gridAdapter.language
    modules = Gpusim2gridAdapter.modules
    mode = "native"
    threaded = False
    settings = Gpusim2gridAdapter.settings | {"mode": "native", "batch_api": "InjectionSweepGPU.compute",
                                              "entry_point": "InjectionSweepGPU.compute", "init": "base-case solution",
                                              "start": "base-case solution", "nb_iter": NB_ITER,
                                              "batch_size": BATCH_SIZE}
    dependencies = Gpusim2gridAdapter.dependencies

    def load(self, case):
        from gpusim2grid import InjectionSweepGPU
        base = base_case(case)
        grid, v_flat = solved_grid(base)
        gen_p, load_p, load_q, _ = injections(case, grid)
        sweep = InjectionSweepGPU(grid, nb_iter=NB_ITER, max_iter_base=MAX_ITERATIONS, tol_base=TOLERANCE_PU)
        sweep.set_injections_from_elements(load_p, load_q, gen_p)
        return {"grid": grid, "sweep": sweep, "v_flat": v_flat, "n": len(gen_p), "ids": bus_ids(base, grid), "v": None}

    def solve(self, model, threads=1):
        assert threads == 1, "one GPU: threads are not a setting"
        if model["grid"].ac_pf(model["v_flat"].copy(), MAX_ITERATIONS, TOLERANCE_PU).shape[0] == 0:
            raise DidNotConverge("lightsim2grid's base case did not converge")
        sweep = model["sweep"]
        sweep.compute(batch_size=BATCH_SIZE)
        model["v"] = sweep.V_results.to_numpy().reshape(model["n"], -1)

    def solution(self, model, case):
        v, ids = model["v"], model["ids"]
        assert v.shape == (model["n"], len(ids))
        return BatchSolution(ids, np.abs(v), np.angle(v, deg=True))
