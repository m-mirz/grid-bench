"""gridoxide (GPU): DC N-1 through `GpuDcSolver.solve_contingencies`
(`native`, one GPU).

gridoxide's DC N-1 on an NVIDIA GPU (its `gpu` feature, gridoxide-gpu's
`GpuDcSolver`): the reduced susceptance matrix factored by cuDSS once per
session; per chunk of outages one solve with a right-hand side per outage
(the angle response to a unit transfer across the outaged branch), then
kernels for each outage's transfer and its angles and flows, the same
arithmetic as the CPU's `dc_contingencies` (adapters/gridoxide_dc_adapter.py,
PR #28). The chunks run through page-locked staging over two streams, so a
chunk's copy to the host overlaps the next one's compute: on
case9241pegase#dcn1-full the angles and flows are 2.9 GB, and bringing them
to the host is most of the time.

Input and joins: the CPU adapter's. `PowerFlowModel.from_matpower` with
`method="dc"` on the case's `.m` (MATPOWER's DC model, shunt conductance as
load), buses by MATPOWER bus number (`bus_numbers()`), branches by row of
the branch matrix (`branch_rows()`), both asserted; the outages by branch
row, each row's buses asserted against the outage file's.

Settings:
- The timed call is `solve_contingencies(outages)`: the base case's DC
  power flow on the CPU (one sparse solve, the session's `DcSolver`), every
  outage on the device, and the angles and flows on the host as read-only
  float64 buffers, which numpy views without a copy. The session (cuDSS's
  analysis and factorization, the device buffers) is built in `load`, as the
  CPU adapter's factorization is not: it is rebuilt by `reset()` in that
  adapter's timed call, about 35 ms on case9241pegase.
- `capacity=1024` outages per chunk: 512 and 2048 timed slower on
  case9241pegase#dcn1-full (gridoxide's plans/GPU_DC_PLAN.md).
- FP64, cuDSS's default settings.
- An outage the device reports unsolved (a radial branch) fails the case;
  none of these outages islands the grid (`cases.contingency`).
- Thread count: not a setting (one GPU); the host copies use every core.

Results (A100-SXM4-40GB, gridoxide 5d14426, cuDSS 1.0.0; ms per call, the
CPU adapter on 30 threads and ToOp on the same machine): every case
accepted, angles graded, within 8.2e-10 MW.

    case                       gridoxide (GPU)  gridoxide (CPU, 30)  ToOp (GPU, flows only)
    case9241pegase#dcn1        9.8              74                   51, rejected
    case1354pegase#dcn1-full   7.3              17                   55
    case2869pegase#dcn1-full   53               65                   112, rejected
    case9241pegase#dcn1-full   529              617                  893, rejected

The call includes freeing the previous call's result when the adapter
replaces it, about 300 ms of the 529 (2.9 GB) and the same in the CPU
adapter's 617; in gridoxide's own bench, which frees outside the clock,
234 against 324 ms. The CPU adapter also rebuilds its factorization in the
call (about 35 ms there), which this one keeps in its session.
"""
import numpy as np

from adapters.batch_adapter import base_case, outages
from adapters.dc_adapter import DcContingencyAdapter, DcSolution
from adapters.gridoxide_gpu_adapter import GridoxideGpu
from adapters.solver_adapter import DidNotConverge
from cases.matpower import BUS_I, F_BUS, T_BUS, parse_m
from cases.registry import CASES

CAPACITY = 1024


class GridoxideGpuDcN1(GridoxideGpu, DcContingencyAdapter):
    mode = "native"
    threaded = False
    settings = {"method": "LODF, angles and flows, on the device", "api": "GpuDcSolver.solve_contingencies",
                "linear_solver": "cudss", "precision": "fp64", "capacity": CAPACITY,
                "reader": "from_matpower (.m)"}

    def load(self, case):
        import gridoxide
        path = CASES[base_case(case)]["file"]
        mpc = parse_m(path)
        model = gridoxide.PowerFlowModel.from_matpower(str(path), method="dc")
        numbers = model.bus_numbers()
        assert numbers == [int(b) for b in mpc["bus"][:, BUS_I]], "one bus per row of mpc.bus, in order"
        rows = model.branch_rows()
        assert len(set(rows)) == len(rows) == model.n_branches, "one flat branch per in-service row"
        flat = {row: k for k, row in enumerate(rows)}
        out = outages(case)
        branch = mpc["branch"]
        for row, f, t in zip(out["branch_row"], out["from_bus"], out["to_bus"]):
            assert (int(branch[row, F_BUS]), int(branch[row, T_BUS])) == (int(f), int(t)), f"row {row}"
        return {"gpu": gridoxide.GpuDcSolver(model, capacity=CAPACITY), "bus_ids": [str(b) for b in numbers],
                "rows": rows, "base": float(mpc["baseMVA"]),
                "outages": [flat[int(row)] for row in out["branch_row"]], "va": None, "pf": None}

    def solve(self, model, threads=1):
        assert threads == 1, "one GPU: threads are not a setting"
        angles, flows, solved = model["gpu"].solve_contingencies(model["outages"])
        if not all(solved):
            k = solved.index(False)
            raise DidNotConverge(f"outage {k} (branch {model['outages'][k]}) is radial: no post-outage state")
        n = len(model["outages"])
        model["va"] = np.frombuffer(angles).reshape(n, len(model["bus_ids"]))
        model["pf"] = np.frombuffer(flows).reshape(n, len(model["rows"]))

    def solution(self, model, case):
        return DcSolution(model["bus_ids"], np.rad2deg(model["va"]), model["rows"], model["pf"] * model["base"])
