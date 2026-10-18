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
- `batch_size=2048`, not its default of 512: the chunk at which an A100
  is full (below, issue #19). Every `#sweep` and `#sweep-1k` is one chunk on
  the device, a `#sweep-10k` five. It is the setting for a 40 GB device:
  case9241pegase's peaks at 20 GB, and 8192 runs out of memory.
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

Where the time goes (gpusim2grid's own phase times, A100, case9241pegase,
one timed call of 705 ms): the cuDSS factorizations and triangular solves
549 ms (78 %), its symbolic analysis 73 ms (10 %), redone on every call
though the pattern never changes (the session rebuilds its solver in
`run`), lightsim2grid's base case on the CPU about 57 ms, the voltages'
copy to the host (100 x 9241) 11 ms, upload and CPU preprocessing 11 ms,
every other Newton kernel together 3 ms. With a fixed step count there is
no convergence test, so the host never waits on the device between steps:
GPU time and wall time agree. Not communication but the sparse LU is the
cost.

Batch size (issue #19: at its default of 512 an A100 is far from full).
A separate experiment, the same adapter with only `batch_size` changed, a
process per point, every scenario graded, peak device memory from
nvidia-smi; seconds per call of 10 000 scenarios (14 374 outages for
case9241pegase#n1-full), NaN scenarios in parentheses:

    batch_size              512          1024         2048     4096     8192
    case9241pegase#sweep    17.5         12.3         9.6      9.9      out of memory
    mvlv10616#sweep         17.8         12.5         9.7      9.8      out of memory
    case2869pegase#sweep    5.1          3.6          2.8      2.9      2.9 (6)
    case1354pegase#sweep    2.3 (2 907)  1.6 (5 060)  1.3      1.3      1.3 (1 259)
    case9241pegase#n1-full  23.5         16.0         12.4     12.6     out of memory

Each doubling to 2048 saves about 30 %, and nothing after: the device is
full there, at 20 GB for the 9241- and 10616-bus sweeps (30 GB at 4096;
8192 runs out of the 40 GB, as cudaErrorMemoryAllocation or cuDSS's
factorize status 2). The NaN loss depends on the chunk, not on the run:
three repeats of case1354pegase#sweep-10k gave 2 775 to 2 938 NaN at 512
and none at 2048, every time; but not monotonically (it is back at 8192,
and at 2 900 with all 10 000 in one chunk), so 2048 avoids it on that
sweep without the cause being known. The rest of the loss is in the N-1
adapter's docstring.

The large sweeps in the benchmark (`results-docker/gpu/`, A100, lightsim2grid
1.1.0, batch_size 2048; ms per scenario, lightsim2grid on 30 threads of the
same machine in parentheses):

    case            1 000 scenarios              10 000 scenarios
    case1354pegase  0.15, 593 NaN (0.12)         0.13, all accepted (0.13)
    case2869pegase  0.37, all accepted (0.43)    0.28, all accepted (0.33)
    case9241pegase  1.37, all accepted (1.79)    0.96, all accepted (1.84)
    mvlv10616       1.32, all accepted (0.92)    0.97, all accepted (0.90)

So among the accepted sweeps gpusim2grid is ahead of 30 cores on
case2869pegase (1.2x at both sizes) and case9241pegase (1.3x, and 1.9x at
10 000), level on case1354pegase#sweep-10k (0.128 against 0.130 ms), and
behind on the feeder. case1354pegase#sweep-1k, one chunk of 1 000, loses 59 % of
its scenarios to NaN. With batch_size 512 the 10 000-scenario sweeps took
1.8x as long, and case2869pegase's lost 92 % of its scenarios in that
earlier run (as did its 1 000), where every run today kept them all
at both sizes, cause unknown (the same pins; possibly the host's driver,
CUDA 13.3 now).
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case
from adapters.gpusim2grid_adapter import Gpusim2gridAdapter, bus_ids, solved_grid
from adapters.lightsim2grid_batch_adapter import injections
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge

NB_ITER = 4
BATCH_SIZE = 2048


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
