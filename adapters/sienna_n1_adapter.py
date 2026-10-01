"""Sienna: N-1 contingency analysis as a loop of PowerFlows' solve (`loop`).

PowerFlows.jl has no AC contingency analysis, and its AC solve takes no
modified Ybus (PowerNetworkMatrices can apply an outage to one, but only its
linear tools use that). A branch is taken out through the system: made
unavailable, a new `PowerFlowData` built, solved, made available again
(`GridBenchSienna.solve_n1!`); building the data per outage is part of the
time. The base case is the power-flow adapter's solve from the common flat
start (`adapters/sienna_adapter.py`, its settings and losses: single-precision
Ybus, transformer charging on one end, PEGASE phase shifters that do not
parse); every outage starts from the base solution, by bus number. One call
into Julia for the whole loop; Julia runs single-threaded.

Join: PowerSystems' MATPOWER parser names every branch `...-i_<row + 1>`
(PowerModels' branch index), asserted by its from and to bus.
"""
import numpy as np

from adapters.batch_adapter import BatchSolution, ContingencyAdapter, base_case, outages
from adapters.sienna_adapter import SiennaAdapter, _gb
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.registry import CASES


class SiennaN1(ContingencyAdapter):
    name = SiennaAdapter.name
    display_name = SiennaAdapter.display_name
    color = SiennaAdapter.color
    package = SiennaAdapter.package
    language = SiennaAdapter.language
    modules = SiennaAdapter.modules
    mode = "loop"
    settings = SiennaAdapter.settings | {"mode": "loop", "start": "base-case solution",
                                         "update": "branch unavailable, PowerFlowData rebuilt"}
    version = SiennaAdapter.version
    dependencies = SiennaAdapter.dependencies

    def load(self, case):
        out = outages(case)
        return {"n1": _gb().load_n1(str(CASES[base_case(case)]["file"]), out["branch_row"].tolist(),
                                    out["from_bus"].tolist(), out["to_bus"].tolist()), "n": len(out["branch_row"])}

    def solve(self, model, threads=1):
        assert threads == 1, "a loop runs on one thread"
        failed = int(_gb().solve_n1_b(model["n1"], TOLERANCE_PU, MAX_ITERATIONS))   # solve_n1!
        if failed < 0:
            raise DidNotConverge(f"base case: NR did not converge in {MAX_ITERATIONS} iterations")
        if failed:
            raise DidNotConverge(f"{failed} of {model['n']} outages did not converge")

    def solution(self, model, case):
        numbers, vm, va = _gb().n1_solution(model["n1"])
        return BatchSolution([str(int(b)) for b in numbers], np.asarray(vm).T, np.asarray(va).T)
