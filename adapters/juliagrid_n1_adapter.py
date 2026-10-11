"""JuliaGrid.jl: N-1 contingency analysis as a loop on the power-flow
adapter's model (`loop`). JuliaGrid has no contingency analysis: the base
case from the flat start, then for every outage its branch's status set to 0
on the analysis (`updateBranch!`, which updates the admittance model the next
Newton step reads), Newton-Raphson from the base solution, and the branch
put back.

Input, settings, join: the power-flow adapter's (`adapters/juliagrid_adapter.py`).
An outage is the `.m`'s branch row, JuliaGrid's branch of the same index
(the row order asserted by the power-flow adapter's join), its from and to
bus asserted again here. One thread (Julia single-threaded).
"""
import numpy as np

from adapters.batch_adapter import LoopContingencyAdapter, base_case, outages
from adapters.juliagrid_adapter import JuliagridAdapter, _gb
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.matpower import F_BUS, T_BUS, parse_m
from cases.registry import CASES


class JuliagridN1(LoopContingencyAdapter):
    name = JuliagridAdapter.name
    display_name = JuliagridAdapter.display_name
    color = JuliagridAdapter.color
    package = JuliagridAdapter.package
    language = JuliagridAdapter.language
    modules = JuliagridAdapter.modules
    settings = JuliagridAdapter.settings | {"mode": "loop", "update": "updateBranch!(status = 0)",
                                            "start": "base-case solution"}
    single = JuliagridAdapter()
    version = JuliagridAdapter.version
    dependencies = JuliagridAdapter.dependencies

    def load(self, case):
        base = base_case(case)
        model = self.single.load(base)
        out = outages(case)
        branch = parse_m(CASES[base]["file"])["branch"]
        for r, f, t in zip(out["branch_row"], out["from_bus"], out["to_bus"]):
            assert (int(branch[r, F_BUS]), int(branch[r, T_BUS])) == (int(f), int(t))
        jl = _gb().load_outages(model["jl"], [int(r) + 1 for r in out["branch_row"]])
        return {"single": model, "outages": out, "jl": jl, "bus_ids": model["bus_ids"]}

    def solve_base(self, model):
        if _gb().solve_n1_base_b(model["single"]["jl"], model["jl"], TOLERANCE_PU, MAX_ITERATIONS) < 0:
            raise DidNotConverge(f"base case: NR did not converge in {MAX_ITERATIONS} iterations")

    def solve_outage(self, model, k):
        if _gb().solve_outage_b(model["single"]["jl"], model["jl"], k + 1, TOLERANCE_PU, MAX_ITERATIONS) < 0:
            raise DidNotConverge(f"outage {k}: NR did not converge in {MAX_ITERATIONS} iterations")

    def voltages(self, model):
        vm, va = _gb().solution(model["single"]["jl"])
        return np.asarray(vm), np.asarray(va)
