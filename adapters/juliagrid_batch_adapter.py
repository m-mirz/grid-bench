"""JuliaGrid.jl: batch power flow as a loop on the power-flow adapter's
model (`loop`). JuliaGrid has no batch power flow; a sweep is the base case,
then every scenario written into the analysis (`updateBus!` for demand,
`updateGenerator!` for dispatch, its documented way to change an analysis
in place) and solved from the base solution.

Input, settings, join: the power-flow adapter's (`adapters/juliagrid_adapter.py`,
KLU Newton-Raphson, flat start for the base case, `TOLERANCE_PU`,
`MAX_ITERATIONS`). Scenarios: demand per loaded bus (MATPOWER number, through
the asserted bus order) and the output of every online generator (gen
row), in p.u. on the case's base power; the base values for the base solve.
Every scenario starts from the base solution (the analysis' voltages set to
it), never from the scenario before it. One thread (Julia single-threaded).
"""
import numpy as np

from adapters.batch_adapter import LoopBatchAdapter, base_case, scenarios
from adapters.juliagrid_adapter import JuliagridAdapter, _gb
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.matpower import parse_m
from cases.registry import CASES


class JuliagridBatch(LoopBatchAdapter):
    name = JuliagridAdapter.name
    display_name = JuliagridAdapter.display_name
    color = JuliagridAdapter.color
    package = JuliagridAdapter.package
    language = JuliagridAdapter.language
    modules = JuliagridAdapter.modules
    settings = JuliagridAdapter.settings | {"mode": "loop", "update": "updateBus!, updateGenerator!, then Newton-Raphson",
                                            "start": "base-case solution"}
    single = JuliagridAdapter()
    version = JuliagridAdapter.version
    dependencies = JuliagridAdapter.dependencies

    def load(self, case):
        base = base_case(case)
        model = self.single.load(base)
        sweep = scenarios(case)
        base_mva = float(parse_m(CASES[base]["file"])["baseMVA"])
        index = {int(b): i + 1 for i, b in enumerate(model["bus_ids"])}
        jl = _gb().load_sweep(model["jl"], [index[int(b)] for b in sweep["load_bus"]],
                              [int(r) + 1 for r in sweep["gen_row"]],
                              sweep["pd"] / base_mva, sweep["qd"] / base_mva, sweep["pg"] / base_mva)
        return {"single": model, "sweep": sweep, "jl": jl, "bus_ids": model["bus_ids"]}

    def solve_base(self, model):
        if _gb().solve_sweep_base_b(model["single"]["jl"], model["jl"], TOLERANCE_PU, MAX_ITERATIONS) < 0:
            raise DidNotConverge(f"base case: NR did not converge in {MAX_ITERATIONS} iterations")

    def solve_scenario(self, model, k):
        if _gb().solve_scenario_b(model["single"]["jl"], model["jl"], k + 1, TOLERANCE_PU, MAX_ITERATIONS) < 0:
            raise DidNotConverge(f"scenario {k}: NR did not converge in {MAX_ITERATIONS} iterations")

    def voltages(self, model):
        vm, va = _gb().solution(model["single"]["jl"])
        return np.asarray(vm), np.asarray(va)
