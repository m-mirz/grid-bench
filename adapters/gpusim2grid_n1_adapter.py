"""gpusim2grid: N-1 contingency analysis through `ContingencyAnalysisGPU`
(`native`, one GPU).

gpusim2grid's contingency analysis is its most mature batch path: the base
case solved on the CPU by lightsim2grid, then every outage on the GPU from
that solution, one cuDSS symbolic analysis reused across the batch. That is
the N-1 problem's start (every outage from the tool's own base-case
solution).

Input and joins: lightsim2grid's (`adapters/lightsim2grid_n1_adapter.py`,
`contingency_ids`): the same grid from `init_from_matpower` and the same
contingency ids (lines, then transformers, each element's buses asserted
against its branch row), given to `add_contingencies_by_branch_id` one
branch per contingency, in the case's outage order. Bus ids: the
power-flow adapter's (AC-solver numbering, asserted).

Settings:
- The timed call is lightsim2grid's solve of the base case from its flat
  start (`ac_pf(v_flat, MAX_ITERATIONS, TOLERANCE_PU)`), then `compute` and
  the voltages' copy to the host. The session's outages start from that
  same solution, computed the same way when the session was built
  (`init_from_n_powerflow=True`, its default), since it takes its start
  only then; repeating the solve in the call counts the base case, as the
  problem asks.
- `nb_iter=4` (its default): a fixed number of Newton steps per outage,
  with no convergence test of its own. Not tuned per case: the oracle
  grades every outage, and one that 4 steps do not bring to the tolerance
  shows there.
- `batch_size=2048`, the batch adapter's (its docstring: the chunk at
  which an A100 is full): every `#n1` case (at most 200 outages) is one
  chunk on the device, case1354pegase#n1-full one, case2869pegase's two,
  case9241pegase's eight.
- `handle_disconnected_grid=False` (its default): no outage here islands
  the grid (`cases.contingency`).
- FP64, default cuDSS settings, no damping, distributed-slack formulation
  with the case's one slack: the power-flow adapter's.
- Thread count: not a setting (one GPU).

Results (A100, e950f8b, `results-docker/gpu/`): every outage accepted on
case14, case1354pegase (0.52 ms per outage) and case9241pegase (3.7 ms;
ExaPF.jl on the same GPU 6.9 ms, lightsim2grid 28 ms on one thread of the
same machine and 1.4 ms on 30). Known loss, reported rather than worked
around: case2869pegase#n1 fails, 129 to 167 of its 200 outages NaN, a
different set on every run of the same session. ExaPF.jl's GPU N-1 fails
the same case the same way (adapters/exapf_gpu_n1_adapter.py); the two
share no code but cuDSS 0.8.0's batched factorization and solve, which
points there.

Full N-1 (`#n1-full`, every outage that converges; A100, batch_size 2048,
the batch adapter's experiment for the choice; lightsim2grid on 30 threads
of the same machine, which accepts every outage of all three, in
parentheses):
- case1354pegase (1 428 outages, one chunk, 0.11 ms each (0.092)): 247
  NaN (128 in an earlier run), every finite outage but one accepted; 144 to 232 NaN over five runs
  of the experiment at one chunk, against 436 to 494 at 512 (chunks of
  512). Not cuDSS's batch size alone: ExaPF.jl's GPU N-1 solves the same
  1 428 outages in one batch of 1 429 blocks with every outage accepted
  (adapters/exapf_gpu_n1_adapter.py).
- case2869pegase (3 804 outages, 0.25 ms each (0.26)): no NaN, 1 outage
  finite and rejected. At batch_size 512, 0.50 ms each, an earlier run lost
  3 324 to 3 380 (87 to 89 %); every run today kept them all at every size,
  cause unknown (see the batch adapter), while case2869pegase#n1 above
  still loses 167 of its 200.
- case9241pegase (14 374 outages, 0.84 ms each (1.24), 1.6 ms at 512): no
  NaN, but 16 outages finite and wrong at every batch size (worst: outage
  5002, 50 MW off, a bus at 0.29 p.u.): `nb_iter=4` does not bring the
  hardest outages to the tolerance, which none of the default 200 needed.
"""
import numpy as np

from adapters.batch_adapter import BatchSolution, ContingencyAdapter, base_case
from adapters.gpusim2grid_adapter import Gpusim2gridAdapter, bus_ids, solved_grid
from adapters.gpusim2grid_batch_adapter import BATCH_SIZE, NB_ITER
from adapters.lightsim2grid_n1_adapter import contingency_ids
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge


class Gpusim2gridN1(ContingencyAdapter):
    name = Gpusim2gridAdapter.name
    display_name = Gpusim2gridAdapter.display_name
    color = Gpusim2gridAdapter.color
    package = Gpusim2gridAdapter.package
    language = Gpusim2gridAdapter.language
    modules = Gpusim2gridAdapter.modules
    mode = "native"
    threaded = False
    settings = Gpusim2gridAdapter.settings | {"mode": "native", "batch_api": "ContingencyAnalysisGPU.compute",
                                              "entry_point": "ContingencyAnalysisGPU.compute",
                                              "start": "base-case solution", "nb_iter": NB_ITER,
                                              "batch_size": BATCH_SIZE}
    dependencies = Gpusim2gridAdapter.dependencies

    def load(self, case):
        from gpusim2grid import ContingencyAnalysisGPU
        base = base_case(case)
        grid, v_flat = solved_grid(base)
        ids, _ = contingency_ids(case, grid)
        ca = ContingencyAnalysisGPU(grid, nb_iter=NB_ITER, max_iter_base=MAX_ITERATIONS, tol_base=TOLERANCE_PU)
        ca.add_contingencies_by_branch_id([[int(i)] for i in ids])
        return {"grid": grid, "ca": ca, "v_flat": v_flat, "n": len(ids), "ids": bus_ids(base, grid), "v": None}

    def solve(self, model, threads=1):
        assert threads == 1, "one GPU: threads are not a setting"
        if model["grid"].ac_pf(model["v_flat"].copy(), MAX_ITERATIONS, TOLERANCE_PU).shape[0] == 0:
            raise DidNotConverge("lightsim2grid's base case did not converge")
        ca = model["ca"]
        ca.compute(batch_size=BATCH_SIZE)
        model["v"] = ca.V_results.to_numpy().reshape(model["n"], -1)

    def solution(self, model, case):
        v, ids = model["v"], model["ids"]
        assert v.shape == (model["n"], len(ids))
        return BatchSolution(ids, np.abs(v), np.angle(v, deg=True))
