"""ExaPF: batch power flow as ExaPF.jl's block formulation (`native`, one
thread, CPU backend).

`BlockPolarForm(polar, k)` duplicates the network k times and solves all
scenarios as one Newton-Raphson on a block-diagonal Jacobian
(`BatchJacobian`, computed by ForwardDiff with one colouring for every
block): one factorization of the whole block diagonal per step (KLU on the
CPU, cuDSS on a GPU), one residual over all scenarios. This is the API
ExaPF has for many operating points of one network, and the one its GPU
backend is built for; on the CPU the blocks are processed one after
another, so the thread count is not a setting.

Input and settings: the power-flow adapter's (`adapters/exapf_adapter.py`):
ExaPF's MATPOWER parser on the `.m`, KLU, every scenario restored to the
flat start before the solve, `TOLERANCE_PU`, `MAX_ITERATIONS`. Its losses
stand (PQ buses with a generator solved as PV; no import without gencost).

Scenarios: ExaPF keeps loads (`pload`, `qload`) per bus and dispatch
(`pgen`) per online generator, block after block; `GridBenchExaPF.load_batch`
writes each scenario's `Pd`, `Qd` and `Pg` into its block, joined by
MATPOWER bus number (`bus_to_indexes`) and by gen row through the online
generators' order, asserted against their buses. `set_params!` takes loads
only; dispatch is the stack's `pgen`, which every block keeps its own.

Convergence: ExaPF stops when the 2-norm of the mismatch over every
scenario is below `TOLERANCE_PU`, so each scenario's own mismatch is below
it too (stricter than the single solve's criterion), and every scenario
takes as many Newton steps as the hardest one: the steps are independent
per block (block-diagonal Jacobian), so an extra step on a converged
scenario only costs time, which is the API's cost and is timed. If the
batch does not converge, the scenarios whose own 2-norm is not below the
tolerance are counted in the failure.

Results (CPU, 100 scenarios), the oracle accepting every scenario it ran:
- On the CPU the block formulation is slower per scenario than ExaPF's own
  single warm solve on transmission cases: 5.3 ms against 2.0 ms
  (case1354pegase), 13.2 ms against 7.0 ms (case2869pegase). Every
  intermediate of every block is a ForwardDiff dual with one partial per
  Jacobian colour (25 for case1354pegase), so a Jacobian evaluation does
  that much more arithmetic than an assembled one. The formulation is built
  for a GPU, where those blocks run in parallel.
- Memory grows the same way: `BatchJacobian` keeps those duals for every
  block, 0.55 GB of case1354pegase's 0.72 GB, and 15.3 GB live for
  case9241pegase (about 90 colours with ExaPF's natural-order greedy
  colouring). That case peaks at 16 GB, which is why the benchmark template
  never holds two models at once (see its docstring); its 106 ms per
  scenario compare with 93 ms for ExaPF's single solve.
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case, scenarios
from adapters.exapf_adapter import ExapfAdapter, _gb
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.matpower import GEN_STATUS, parse_m
from cases.registry import CASES


class ExapfBatch(BatchAdapter):
    name = ExapfAdapter.name
    display_name = ExapfAdapter.display_name
    color = ExapfAdapter.color
    package = ExapfAdapter.package
    language = ExapfAdapter.language
    modules = ExapfAdapter.modules
    mode = "native"
    threaded = False
    settings = ExapfAdapter.settings | {"mode": "native", "batch_api": "BlockPolarForm",
                                        "tolerance_norm": "2, over all scenarios"}
    version = ExapfAdapter.version
    dependencies = ExapfAdapter.dependencies

    def load(self, case):
        base = base_case(case)
        mpc, sweep = parse_m(CASES[base]["file"]), scenarios(case)
        base_mva = float(mpc["baseMVA"])
        online = np.flatnonzero(mpc["gen"][:, GEN_STATUS] > 0)
        gen_pos = np.searchsorted(online, sweep["gen_row"])
        assert (online[gen_pos] == sweep["gen_row"]).all()
        batch = _gb().load_batch(str(CASES[base]["file"]), TOLERANCE_PU, MAX_ITERATIONS,
                                 sweep["load_bus"].tolist(), sweep["pd"] / base_mva, sweep["qd"] / base_mva,
                                 (gen_pos + 1).tolist(), sweep["gen_bus"].tolist(), sweep["pg"] / base_mva)
        return {"batch": batch, "n": len(sweep["scale"])}

    def solve(self, model, threads=1):
        assert threads == 1, "ExaPF's CPU backend solves the blocks on one thread"
        converged, bad = _gb().solve_batch_b(model["batch"])   # solve_batch!
        if not converged:
            raise DidNotConverge(f"{int(bad)} of {model['n']} scenarios not below tolerance "
                                 f"after {MAX_ITERATIONS} iterations")

    def solution(self, model, case):
        numbers, vm, va = _gb().batch_solution(model["batch"])
        return BatchSolution([str(int(b)) for b in numbers], np.asarray(vm).T, np.asarray(va).T)
