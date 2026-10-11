"""JuliaGrid.jl: AC optimal power flow (`acOptimalPowerFlow`), a JuMP model
solved by Ipopt (Julia, through juliacall).

Input: the case's `.m` through JuliaGrid's reader with its OPF data
(`optimal=true`: generator limits, `gencost`), with the power-flow adapter's
asserted row join (`adapters/juliagrid_adapter.py`). JuliaGrid builds
MATPOWER's problem: polar voltages with the slack angle fixed, polynomial
(or piecewise) cost, generator P and Q limits, branch flow limits from
RATE_A at both ends (apparent power, the `.m`'s rating type), angle-
difference limits, voltage limits and nodal balance.

Settings:
- Ipopt (1.16.0, Ipopt 3.14.19, MUMPS: the PowerModels image's), `tol =
  OPF_TOLERANCE`, `max_iter = OPF_MAX_ITERATIONS`, silent.
- Flat start: every bus at 1 p.u. and 0 rad, every generator at the middle
  of its P and Q ranges, written into the system at load; `setInitialPoint!`
  restores it before every solve and drops the previous solve's duals,
  which JuliaGrid would otherwise pass to Ipopt as a warm start.
- A solve that does not end LOCALLY_SOLVED fails.

Known loss, reported rather than worked around: JuliaGrid builds each
branch limit as the interval 0 <= |S|^2 <= rate^2 (`checkLimit` raises a
squared limit's lower bound to 0, and it is kept as a bound), where
MATPOWER and PowerModels.jl state the upper side only. The bound removes no
feasible point, but Ipopt carries it on every branch, and it slows the
interior point 6 to 13 times: 146 iterations on pglib_opf_case118_ieee and
402 on case300 (with `max_iter` raised), against 23 and 31 with the
upper sides alone, the same objective, PowerModels' 24 and 30 (checked by
rebuilding those limits as upper bounds). So pglib_opf_case118_ieee__api,
case300 (all three conditions), case1354_pegase and case2869_pegase do not
finish in `OPF_MAX_ITERATIONS`; the cases that do are accepted, with
PowerModels' gaps.
"""
from adapters.juliagrid_adapter import JuliagridAdapter, _gb, join, readable
from adapters.optimizer_adapter import OPF_MAX_ITERATIONS, OPF_TOLERANCE, OptimizerAdapter
from adapters.solver_adapter import DidNotConverge, Solution
from cases.matpower import GEN_STATUS, parse_m
from cases.registry import CASES


class JuliagridOptimizer(OptimizerAdapter):
    name = JuliagridAdapter.name
    display_name = JuliagridAdapter.display_name
    color = JuliagridAdapter.color
    package = JuliagridAdapter.package
    language = JuliagridAdapter.language
    modules = JuliagridAdapter.modules
    settings = {"formulation": "acOptimalPowerFlow (polar)", "solver": "Ipopt", "linear_solver": "mumps",
                "init": "flat", "tol": OPF_TOLERANCE, "max_iter": OPF_MAX_ITERATIONS}
    version = JuliagridAdapter.version
    dependencies = JuliagridAdapter.dependencies

    def load(self, case):
        opt = _gb().load_opf(readable(CASES[case]["file"]), OPF_TOLERANCE, OPF_MAX_ITERATIONS)
        mpc = parse_m(CASES[case]["file"])
        return {"jl": opt, "bus_ids": join(case, opt.system), "base": float(mpc["baseMVA"]),
                "online": [r for r in range(len(mpc["gen"])) if mpc["gen"][r, GEN_STATUS] > 0], "iterations": None}

    def solve(self, model):
        ok, iterations = _gb().solve_opf_b(model["jl"])   # solve_opf!
        if not ok:
            raise DidNotConverge(f"Ipopt: not locally solved in {OPF_MAX_ITERATIONS} iterations")
        model["iterations"] = int(iterations)

    def solution(self, model, case):
        vm, va, pg, qg, _ = _gb().opf_solution(model["jl"])
        ids, base = model["bus_ids"], model["base"]
        return Solution(dict(zip(ids, map(float, vm))), dict(zip(ids, map(float, va))), model["iterations"],
                        pg_mw={str(r): float(pg[r]) * base for r in model["online"]},
                        qg_mvar={str(r): float(qg[r]) * base for r in model["online"]})
