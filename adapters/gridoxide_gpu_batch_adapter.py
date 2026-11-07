"""gridoxide (GPU): batch power flow through `GpuBatchSolver.solve_sweep`
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
- The first step on one Jacobian: every scenario starts from the same
  voltages on the same network, so it is one matrix, to the bit. It is
  factored once for the session and the step is one solve with a column per
  scenario, reading one factor where the batch reads one per system.
- `chord_steps=12`, `chord_below=0.9`: up to 12 more steps on those factors,
  each a column of the same solve, then Newton-Raphson. So not Newton-Raphson
  throughout: a chord method first, linear where Newton is quadratic, left
  early by a scenario whose mismatch falls by less than a tenth in a step.
  Every scenario still stops on the same mismatch at the same tolerance, and
  chord steps do not count towards `max_iteration`. On the distribution
  feeders the scenarios leave them at once, so they gain nothing.
- Factor reuse and the CPU hand-off off (`reuse_below=None`,
  `cpu_handoff=None`, the defaults).
- FP64, cuDSS's multiblock factorization and six levels of nested dissection
  (gridoxide's own choice, its scripts/bench/README.md, section 11: 1.3-2.7x
  at 1024 scenarios, half the memory on case9241pegase, the same steps and
  accuracy), no damping, the case's one slack.
- Thread count: not a setting (one GPU); cuDSS's analysis runs on the host
  in `load`.
- One GPU, the one docker-compose passes in (`GRID_BENCH_GPU`);
  `dependencies` records it with the driver, the CUDA runtime and cuDSS.

Results (A100-SXM4-40GB, cuDSS 1.0.0, `results-docker/gpu/`; s a call,
gpusim2grid and lightsim2grid on its 30 threads as of their last run on the
same machine, "rejected" where the oracle does): every scenario of every sweep
accepted, the only GPU tool here of which that holds.

    case                     gridoxide  gpusim2grid      (lightsim2grid)
    case1354pegase#sweep     0.024      0.096            (0.015)
    case2869pegase#sweep     0.047      0.207            (0.036)
    case9241pegase#sweep     0.191      0.693            (0.198)
    mvlv10616#sweep          0.226      0.637            (0.093)
    case1354pegase#sweep-1k  0.061      0.145, rejected  (0.156)
    case2869pegase#sweep-1k  0.131      0.367            (0.330)
    case9241pegase#sweep-1k  0.554      1.330            (1.813)
    mvlv10616#sweep-1k       0.625      1.273            (1.052)
    case1354pegase#sweep-10k 0.617      1.220            (1.279)
    case2869pegase#sweep-10k 1.284      2.703            (3.335)
    case9241pegase#sweep-10k 5.899      9.209            (17.984)
    mvlv10616#sweep-10k      6.844      9.226            (9.031)

Faster than gpusim2grid on every sweep, 1.3-2.8x on the large ones, and than
lightsim2grid on every sweep of 1000 scenarios or more. gpusim2grid runs cuDSS
0.8.0, cheaper per system than 1.0.0 but NaN in a uniform batch (gridoxide's
plans/GPU_BATCH_PLAN.md, "Correction: A fails on 0.8.0"), which is why
gridoxide-gpu refuses it, and four fixed steps. Here a scenario takes 3 to 12
steps, most of them chord steps, a column of a shared solve each.
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case, scenarios
from adapters.gridoxide_gpu_adapter import CAPACITY, CHORD_BELOW, CHORD_STEPS, GridoxideGpu, session
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
    settings = GridoxideGpu.settings | {"mode": "native", "batch_api": "GpuBatchSolver.solve_sweep",
                                        "entry_point": "GpuBatchSolver.solve_sweep",
                                        "chord_steps": CHORD_STEPS, "chord_below": CHORD_BELOW,
                                        "start": "base-case solution", "capacity": CAPACITY}

    def load(self, case):
        model, gpu, ids = session(base_case(case))
        gpu.set_scenarios(injections(case, model))
        return {"model": model, "gpu": gpu, "ids": ids, "n": len(scenarios(case)["scale"]), "v": None}

    def solve(self, model, threads=1):
        assert threads == 1, "one GPU: threads are not a setting"
        model["model"].solve(init="flat")
        results = model["results"] = model["gpu"].solve_sweep()
        failed = sum(not r.converged for r in results)
        if failed:
            raise DidNotConverge(f"{failed} of {len(results)} scenarios not below tolerance")
        model["v"] = model["gpu"].voltages()

    def solution(self, model, case):
        n, ids = model["n"], model["ids"]
        vm, va = (np.frombuffer(b).reshape(n, len(ids)) for b in model["v"])
        return BatchSolution(ids, vm, np.rad2deg(va))
