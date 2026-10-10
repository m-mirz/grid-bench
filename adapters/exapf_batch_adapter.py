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
ExaPF's MATPOWER parser on the `.m`, KLU, `TOLERANCE_PU`, `MAX_ITERATIONS`.
Start: the timed call solves the base case with the power-flow adapter's
model from its flat start, then fills every block with that solution and
solves the blocks from there, as the N-1 adapter does. Its losses
stand (PQ buses with a generator solved as PV), and it reads the same
prepared `.m` (`m_path`, a zero gencost where the case has none).

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

Results (CPU, 100 scenarios from the base case, AMD EPYC 7J13), the oracle
accepting every scenario: 6.8 ms per scenario on case1354pegase, 18.6 ms on
case2869pegase, 127 ms on case9241pegase.
- The block formulation is slower per scenario than ExaPF's own single warm
  solve on transmission cases (from the flat start, before the batch
  started from the base case, on another machine: 5.3 ms against 2.0 ms
  on case1354pegase, 13.2 ms against 7.0 ms on case2869pegase). Every
  intermediate of every block is a ForwardDiff dual with one partial per
  Jacobian colour (25 for case1354pegase), so a Jacobian evaluation does
  that much more arithmetic than an assembled one. The formulation is built
  for a GPU, where those blocks run in parallel.
- Memory grows the same way: `BatchJacobian` keeps those duals for every
  block, 0.55 GB of case1354pegase's 0.72 GB, and 15.3 GB live for
  case9241pegase (about 90 colours with ExaPF's natural-order greedy
  colouring). That case peaks at 16 GB, which is why the benchmark template
  never holds two models at once (see its docstring); with the flat start
  its 106 ms per scenario compared with 93 ms for ExaPF's single solve.
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case, scenarios
from adapters.exapf_adapter import ExapfAdapter
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.matpower import GEN_STATUS, parse_m
from cases.registry import CASES, m_path


class ExapfBatch(BatchAdapter):
    name = ExapfAdapter.name
    display_name = ExapfAdapter.display_name
    color = ExapfAdapter.color
    package = ExapfAdapter.package
    language = ExapfAdapter.language
    julia = ExapfAdapter.julia
    modules = ExapfAdapter.modules
    mode = "native"
    threaded = False
    settings = ExapfAdapter.settings | {"mode": "native", "batch_api": "BlockPolarForm",
                                        "tolerance_norm": "2, over all scenarios", "start": "base-case solution"}
    _jl = ExapfAdapter._jl
    version = ExapfAdapter.version
    dependencies = ExapfAdapter.dependencies

    def load(self, case):
        base = base_case(case)
        mpc, sweep = parse_m(CASES[base]["file"]), scenarios(case)
        base_mva = float(mpc["baseMVA"])
        online = np.flatnonzero(mpc["gen"][:, GEN_STATUS] > 0)
        gen_pos = np.searchsorted(online, sweep["gen_row"])
        assert (online[gen_pos] == sweep["gen_row"]).all()
        jl = self._jl()
        batch = jl.GB.load_batch(str(m_path(base)), jl.BACKEND, jl.FACTORIZATION, TOLERANCE_PU, MAX_ITERATIONS,
                                 sweep["load_bus"].tolist(), sweep["pd"] / base_mva, sweep["qd"] / base_mva,
                                 (gen_pos + 1).tolist(), sweep["gen_bus"].tolist(), sweep["pg"] / base_mva)
        return {"batch": batch, "n": len(sweep["scale"])}

    def solve(self, model, threads=1):
        assert threads == 1, "ExaPF solves all blocks in one call: threads are not a setting"
        base_converged, bad = self._jl().GB.solve_batch_b(model["batch"])   # solve_batch!
        if not base_converged:
            raise DidNotConverge(f"base case did not converge in {MAX_ITERATIONS} iterations")
        if bad:
            raise DidNotConverge(f"{int(bad)} of {model['n']} scenarios not below tolerance "
                                 f"after {MAX_ITERATIONS} iterations")

    def solution(self, model, case):
        numbers, vm, va = self._jl().GB.batch_solution(model["batch"])
        return BatchSolution([str(int(b)) for b in numbers], np.asarray(vm).T, np.asarray(va).T)
