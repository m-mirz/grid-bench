"""PowerModels.jl: AC-OPF in the polar formulation (`ACPPowerModel`,
`build_opf`), a JuMP model solved by Ipopt (Julia, through juliacall, which
runs Julia inside the benchmark process; the Julia half is
`GridBenchPowerModels` in tool-configs/powermodels). PGLib-OPF's published
reference objectives were computed with this formulation and solver
(PowerModels.jl 0.19.9, Ipopt 3.14.4 with HSL ma27), so it is drawn as a
reference line (dotted, in MATPOWER's neutral ink), and graded like every
other tool.

Input: `PowerModels.parse_file` on the `.m` (PowerModels' own MATPOWER
parser). It checks and corrects the data as it parses (e.g. angle-difference
limits wider than 60 degrees, thermal limits of 0), logging each change; the
PGLib cases need none of the corrections that change a limit, and the oracle
grades against the raw `.m` either way. Joins: buses by number (`bus_i`);
generators by `index`, the row in the `.m`'s gen matrix (1-based there,
0-based here), asserted against the row's bus.

Settings:
- `ACPPowerModel`, `build_opf`: MATPOWER's AC-OPF in polar form: polynomial
  cost, voltage, generator, branch apparent-power (`rate_a`, both ends) and
  angle-difference limits.
- Flat start: `vm_start = 1`, `va_start = 0` on every bus, `pg_start`,
  `qg_start` at the middle of each generator's range, written into the data
  before the model is built; JuMP passes these start values to Ipopt on
  every `optimize!`, so every solve starts there.
- The JuMP model is built once in `load` (timed as import) and re-optimized
  by `solve`: the persistent model, as in every other adapter.
- Ipopt: `tol = OPF_TOLERANCE` (Ipopt's own default is 1e-8, on its scaled
  NLP error), `max_iter = OPF_MAX_ITERATIONS`, `linear_solver = "mumps"`
  (the one Ipopt_jll ships; PGLib's reference times used HSL ma27, which
  needs a licence), `print_level = 0`.
- Julia runs single-threaded (`PYTHON_JULIACALL_THREADS=1`); PowerModels'
  logging is silenced.

Result: the optimum on every default case, within the reference's rounding
(case2869_pegase and case300_ieee__sad included, where MATPOWER's MIPS
stops early), in 14 to 45 Ipopt iterations.
"""
from importlib.metadata import version

from adapters.optimizer_adapter import OPF_MAX_ITERATIONS, OPF_TOLERANCE, OptimizerAdapter
from adapters.solver_adapter import DidNotConverge, Solution
from cases.matpower import GEN_BUS, parse_m
from cases.registry import CASES


def _gb():
    from adapters.powermodels_julia import GB   # starts Julia on first use
    return GB


class PowermodelsOptimizer(OptimizerAdapter):
    name = "powermodels"
    display_name = "PowerModels.jl (Ipopt)"
    color = "#52514f"   # tools/palette.py REFERENCE_DOTTED: MATPOWER's neutral ink, dotted
    package = "juliacall"
    modules = ("adapters.powermodels_julia",)
    language = "julia"
    settings = {"formulation": "ACPPowerModel", "solver": "Ipopt", "linear_solver": "mumps", "init": "flat",
                "tol": OPF_TOLERANCE, "max_iter": OPF_MAX_ITERATIONS}

    def version(self):
        return self.dependencies()["PowerModels"]

    def dependencies(self):
        from adapters.powermodels_julia import JULIA_VERSION
        return {k: str(v) for k, v in _gb().versions().items()} | {"julia": JULIA_VERSION,
                                                                    "juliacall": version("juliacall")}

    def load(self, case):
        return _gb().load(str(CASES[case]["file"]), OPF_TOLERANCE, OPF_MAX_ITERATIONS)

    def solve(self, model):
        iterations = _gb().solve_b(model)   # juliacall spells solve! as solve_b
        if iterations < 0:
            raise DidNotConverge(f"Ipopt: not locally solved in {OPF_MAX_ITERATIONS} iterations")
        self._iterations = int(iterations)

    def solution(self, model, case):
        buses, vm, va, rows, gen_buses, pg, qg = _gb().solution(model)
        gen = parse_m(CASES[case]["file"])["gen"]
        rows = [int(r) for r in rows]
        assert all(int(gen[r, GEN_BUS]) == int(b) for r, b in zip(rows, gen_buses))
        buses = [str(b) for b in buses]
        return Solution(dict(zip(buses, map(float, vm))), dict(zip(buses, map(float, va))), self._iterations,
                        pg_mw={str(r): float(p) for r, p in zip(rows, pg)},
                        qg_mvar={str(r): float(q) for r, q in zip(rows, qg)})
