"""ExaPF: N-1 contingency analysis through ExaPF.jl's line-contingency
formulation (`native`, one thread, CPU backend).

`PowerFlowBalance(blk, contingencies)` on a `BlockPolarForm` of 1 + n
blocks: block 1 is the base case, block 1 + j the network with outage j's
line admittances zeroed (`PS.drop_line`), all solved as one Newton-Raphson
on a block-diagonal Jacobian (`BatchJacobian`, one KLU of the whole block
diagonal), as in the batch adapter. ExaPF's own tests solve this power flow
(test/Polar/api.jl); it is also the constraint block of its
security-constrained OPF.

One timed call (`GridBenchExaPF.solve_n1!`): the base case from flat start
with the power-flow adapter's model, then every block started from that
solution, as the problem asks. Block 1 is the base case again and starts
converged.

Joins: a `LineContingency` is the 1-based row of the `.m`'s branch matrix
(ExaPF keeps out-of-service rows, with zero admittance), and each outage's
from and to bus are asserted against that row.

Input, settings and losses: the power-flow adapter's
(`adapters/exapf_adapter.py`). Convergence as in the batch adapter: one
2-norm over every block, so each outage's own mismatch is below
`TOLERANCE_PU` and every outage takes as many steps as the hardest; the
outages whose own mismatch is not below it are counted in a failure.

Memory: every block keeps the Jacobian's ForwardDiff duals (one partial
per colour), so 201 blocks need about twice the batch's 100.
`load` computes a lower bound from ExaPF's layout (`dual_bytes`: those
duals alone) before building anything, and raises `MemoryError` if it
exceeds the memory the kernel reports available (`MemAvailable`;
`available_bytes` of the backend's Julia module, see the GPU adapter): a
system that certainly does not fit would otherwise have the kernel kill
the run, and with it every other case's record. This refuses
case9241pegase#n1 on a 28 GB machine (at least 26.4 GB; the batch measured
15.3 GB live for 100 blocks); a machine with enough memory runs it
unchanged.

Results (CPU, 200 outages): every outage accepted. 920 ms for
case1354pegase and 2.7 s for case2869pegase, between lightsim2grid's
contingency engine (137 ms, 443 ms) and Sparlectra's loop (1.4 s, 4.0 s);
peak memory 2.2 GB and 5.1 GB.
"""
import numpy as np

from adapters.batch_adapter import BatchSolution, ContingencyAdapter, base_case, outages
from adapters.exapf_adapter import ExapfAdapter
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.registry import m_path


class ExapfN1(ContingencyAdapter):
    name = ExapfAdapter.name
    display_name = ExapfAdapter.display_name
    color = ExapfAdapter.color
    package = ExapfAdapter.package
    language = ExapfAdapter.language
    julia = ExapfAdapter.julia
    modules = ExapfAdapter.modules
    mode = "native"
    threaded = False
    settings = ExapfAdapter.settings | {"mode": "native", "batch_api": "PowerFlowBalance(BlockPolarForm, LineContingency)",
                                        "start": "base-case solution", "tolerance_norm": "2, over all blocks"}
    _jl = ExapfAdapter._jl
    version = ExapfAdapter.version
    dependencies = ExapfAdapter.dependencies

    def load(self, case):
        out = outages(case)
        path = str(m_path(base_case(case)))
        jl = self._jl()
        base = jl.GB.load(path, jl.BACKEND, jl.FACTORIZATION, TOLERANCE_PU, MAX_ITERATIONS)
        need, available = int(jl.GB.dual_bytes(base, len(out["branch_row"]) + 1)), jl.available_bytes()
        if need > available:
            raise MemoryError(f"ExaPF's contingency system needs at least {need / 1e9:.1f} GB "
                              f"(its Jacobian duals alone), {available / 1e9:.1f} GB available")
        n1 = jl.GB.load_n1(base, out["branch_row"].tolist(), out["from_bus"].tolist(), out["to_bus"].tolist())
        return {"n1": n1, "n": len(out["branch_row"])}

    def solve(self, model, threads=1):
        assert threads == 1, "ExaPF solves all blocks in one call: threads are not a setting"
        base_converged, bad = self._jl().GB.solve_n1_b(model["n1"])   # solve_n1!
        if not base_converged:
            raise DidNotConverge(f"base case did not converge in {MAX_ITERATIONS} iterations")
        if bad:
            raise DidNotConverge(f"{int(bad)} of {model['n']} outages not below tolerance "
                                 f"after {MAX_ITERATIONS} iterations")

    def solution(self, model, case):
        numbers, vm, va = self._jl().GB.n1_solution(model["n1"])
        return BatchSolution([str(int(b)) for b in numbers], np.asarray(vm).T, np.asarray(va).T)
