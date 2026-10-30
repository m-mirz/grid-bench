"""gridoxide (GPU): batch power flow through `GpuBatchSolver.solve_batch`
(`native`, one GPU).

gridoxide's batched Newton-Raphson on an NVIDIA GPU (its `gpu` feature, the
`gridoxide-gpu` crate): the Jacobians, mismatches and updates of every
scenario in CUDA kernels, the linear systems as one uniform batch of cuDSS
1.0 (one symbolic analysis for the session, then masked refactorizations and
solves), and every scenario stopped at its own convergence: after each step
the host reads back one ‖F‖∞ per scenario and drops the converged ones from
the masks of the next. Its own image (tool-configs/gridoxide_gpu), built
from a pinned commit; results go to `results-docker/gpu/`.

Input: `PowerFlowModel.from_matpower` on `cases.prep`'s `.m` (`m_path`),
gridoxide's own MATPOWER reader, with `tol=TOLERANCE_PU` and
`max_iter=MAX_ITERATIONS`. Bus ids: `bus_numbers()`, one per MATPOWER bus
(asserted). Scenarios: each bus's injection is the base case's
(`bus_spec()`, per unit, net generation minus load) plus the scenario's
change from the case, `(ΔPg − ΔPd, −ΔQd) / baseMVA`, generators summed by
`gen_bus` and loads by `load_bus`, joined by bus number: the change, not a
rebuilt injection, so that whatever gridoxide's importer makes of the case
is left as it is (rule 4). Written into the session in `load`
(`set_scenarios`, the conversion of the Python lists not timed, as
gpusim2grid's `set_injections_from_elements`).

Settings:
- The timed call is gridoxide's solve of the base case from flat start
  (`solve(init="flat")`), then `solve_batch()` and the voltages' copy to the
  host (`voltages()`, two float64 buffers). The session's scenarios start
  from that same solution, computed the same way in `load`, since a session
  takes its start when it is built (as gpusim2grid's); repeating the solve
  in the call counts the base case, as the problem asks.
- Adaptive stopping, gridoxide's default: every scenario to `TOLERANCE_PU`
  (‖F‖∞ in per unit) or `MAX_ITERATIONS`, never a fixed step count. A
  scenario that does not converge raises DidNotConverge with the count.
- `capacity=1024`: scenarios per chunk on the device. Every `#sweep` and
  `#sweep-1k` is one chunk, a `#sweep-10k` ten. Larger chunks do not help
  on an A100: at 1024 one step keeps the device 99.8 % busy
  (plans/GPU_BATCH_PLAN.md in gridoxide, phase 5), and capacity 2048 timed
  the same on case9241pegase's 10 000.
- Factor reuse and the CPU hand-off off (`reuse_below=None`,
  `cpu_handoff=None`, the defaults): Newton-Raphson as stated, every active
  scenario refactored every step.
- FP64, cuDSS's default settings (reordering, pivoting), no damping, the
  case's one slack.
- Thread count: not a setting (one GPU); cuDSS's analysis runs on the host
  in `load`.
- One GPU, the one docker-compose passes in (`GRID_BENCH_GPU`);
  `dependencies` records it with the driver, the CUDA runtime and cuDSS.

Results (A100-SXM4-40GB, gridoxide d105bf4, cuDSS 1.0.0,
`results-docker/gpu/`; ms per scenario, gpusim2grid and ExaPF.jl's GPU
batch run the same day on the same machine, lightsim2grid on its 30
threads in parentheses): every scenario of every sweep accepted, the only
GPU tool here of which that holds.

    case                     gridoxide  gpusim2grid      ExaPF (GPU)    (lightsim2grid)
    case1354pegase#sweep     0.70       0.96             0.86           (0.15)
    case2869pegase#sweep     1.32       2.08             2.02           (0.37)
    case9241pegase#sweep     4.43       6.94             10.2           (2.04)
    mvlv1004#sweep           0.57       25 rejected      0.81           (0.10)
    case1354pegase#sweep-1k  0.27       0.15, 507 NaN    0.22           (0.12)
    case2869pegase#sweep-1k  0.44       0.37             0.55           (0.43)
    case9241pegase#sweep-1k  1.63       1.37             out of memory  (1.79)
    case1354pegase#sweep-10k 0.30       0.13             out of memory  (0.13)
    case2869pegase#sweep-10k 0.51       0.28             out of memory  (0.33)
    case9241pegase#sweep-10k 1.52       0.95             out of memory  (1.84)
    mvlv10616#sweep-10k      1.34       0.96             out of memory  (0.90)

On 100 scenarios it is the fastest GPU tool (1.4-1.6x gpusim2grid, which
redoes cuDSS's analysis and its session's setup on every call, where a
session here keeps them; the analysis alone is 10 % of gpusim2grid's call,
so that is not the whole difference). On large
batches gpusim2grid is 1.2-2.3x faster: it runs cuDSS 0.8.0, whose
factorization and solve cost 1.75-3.7x less per system than 1.0.0's at large
batches, and four fixed steps where gridoxide takes each scenario to 1e-8
p.u. But 0.8.0 returns NaN in a uniform batch (case1354pegase#sweep-1k,
here; gridoxide's plans/GPU_BATCH_PLAN.md, "Correction: A fails on 0.8.0"),
which is why gridoxide-gpu refuses it. Against 30 CPU threads it is ahead
only on case9241pegase's large sweeps (1.1-1.2x).
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case, scenarios
from adapters.gridoxide_gpu_adapter import CAPACITY, GridoxideGpu, session
from adapters.solver_adapter import DidNotConverge
from cases.matpower import BUS_I, GEN_BUS, PD, PG, QD, parse_m
from cases.registry import CASES


def injections(case: str, model) -> list[list[tuple[int, float, float]]]:
    """Every scenario's per-unit injections, as `solve_batch` takes them:
    the base case's plus the scenario's change, at every bus a load or an
    online generator of the sweep sits at."""
    mpc = parse_m(CASES[base_case(case)]["file"])
    sweep = scenarios(case)
    index = {int(b): i for i, b in enumerate(model.bus_numbers())}
    _, p_base, q_base = model.bus_spec()
    pd_case = dict(zip(mpc["bus"][:, BUS_I].astype(int), mpc["bus"][:, PD]))
    qd_case = dict(zip(mpc["bus"][:, BUS_I].astype(int), mpc["bus"][:, QD]))
    load = np.array([index[int(b)] for b in sweep["load_bus"]], dtype=np.int64)
    gen = np.array([index[int(b)] for b in sweep["gen_bus"]], dtype=np.int64)
    assert np.array_equal(mpc["gen"][sweep["gen_row"], GEN_BUS].astype(int), sweep["gen_bus"].astype(int))
    pd0 = np.array([pd_case[int(b)] for b in sweep["load_bus"]])
    qd0 = np.array([qd_case[int(b)] for b in sweep["load_bus"]])
    pg0 = mpc["gen"][sweep["gen_row"], PG]
    buses = np.union1d(load, gen)
    base_mva = mpc["baseMVA"]
    out = []
    for k in range(len(sweep["scale"])):
        dp = np.zeros(len(p_base))
        dq = np.zeros(len(p_base))
        np.add.at(dp, gen, sweep["pg"][k] - pg0)
        np.add.at(dp, load, -(sweep["pd"][k] - pd0))
        np.add.at(dq, load, -(sweep["qd"][k] - qd0))
        p = np.asarray(p_base)[buses] + dp[buses] / base_mva
        q = np.asarray(q_base)[buses] + dq[buses] / base_mva
        out.append(list(zip(buses.tolist(), p.tolist(), q.tolist())))
    return out


class GridoxideGpuBatch(GridoxideGpu, BatchAdapter):
    mode = "native"
    threaded = False
    settings = GridoxideGpu.settings | {"mode": "native", "batch_api": "GpuBatchSolver.solve_batch",
                                        "entry_point": "GpuBatchSolver.solve_batch",
                                        "start": "base-case solution", "capacity": CAPACITY}

    def load(self, case):
        model, gpu, ids = session(base_case(case))
        gpu.set_scenarios(injections(case, model))
        return {"model": model, "gpu": gpu, "ids": ids, "n": len(scenarios(case)["scale"]), "v": None}

    def solve(self, model, threads=1):
        assert threads == 1, "one GPU: threads are not a setting"
        model["model"].solve(init="flat")
        results = model["gpu"].solve_batch()
        failed = sum(not r.converged for r in results)
        if failed:
            raise DidNotConverge(f"{failed} of {len(results)} scenarios not below tolerance")
        model["v"] = model["gpu"].voltages()

    def solution(self, model, case):
        n, ids = model["n"], model["ids"]
        vm, va = (np.frombuffer(b).reshape(n, len(ids)) for b in model["v"])
        return BatchSolution(ids, vm, np.rad2deg(va))
