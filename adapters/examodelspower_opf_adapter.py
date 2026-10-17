"""ExaModelsPower.jl: AC-OPF in polar form (`ac_opf_model`, `form = :polar`),
an ExaModels model (SIMD-friendly algebraic modelling with its own automatic
differentiation, no JuMP) solved by MadNLP (Julia, through juliacall, which
runs Julia inside the benchmark process; the Julia half is
`GridBenchExaModelsPower` in tool-configs/examodelspower). Both come from the
group behind ExaPF.jl, and are built to run on GPUs; this is the CPU backend
(`backend = nothing`), as every other OPF tool here runs.

Input: the `.m`, read by ExaPowerIO (ExaModelsPower's parser), then the
same two calls `ac_opf_model` makes (`parse_ac_power_data`,
`build_polar_opf`), kept apart so the parsed data is at hand for the joins.
Joins: buses by number (`bus_i`); generators by `i`, which is not the row
in the `.m`'s gen matrix but the position among its in-service rows
(ExaPowerIO drops the others as it parses and numbers the rest), so it is
mapped back to the row through the `.m`'s status column, asserted by count
and against the row's bus.

What the model reads differently from MATPOWER (none of it arises on the
PGLib cases, which have only quadratic costs, no zero ratings and nothing
out of service; the oracle grades against the raw `.m` either way):
- costs: the first three polynomial coefficients, always taken as
  quadratic, linear, constant (a `gencost` with n = 2 would be misread);
- `rateA = 0` is a limit of 0 MVA, not unlimited;
- out-of-service generators and branches are dropped as parsed (the
  join above maps generators back to their rows);
- bus types are rewritten from the generators: a PQ bus with an in-service
  generator becomes PV, a PV or reference bus without one becomes PQ, and a
  reference bus so lost moves to the bus of the largest generator.

Settings:
- Polar form: MATPOWER's AC-OPF, polynomial cost, voltage, generator,
  branch apparent-power (`rateA`, both ends, as p^2 + q^2 <= rateA^2) and
  angle-difference limits; the reference bus's angle fixed at 0.
- Flat start: every bus at 1 p.u. and 0 rad (ExaModelsPower's own start),
  every generator at the middle of its P and Q ranges, written into the
  model's starting point (`meta.x0`) in `load`; branch flows start at 0, as
  in ExaModelsPower and PowerModels.
- The model is built once in `load` (timed as import) and solved by `solve`
  with `madnlp`, which makes a new solver each time: nothing (iterate,
  duals, barrier parameter) carries over between solves, so every solve
  starts flat. MadNLP's own re-solve (`solve!` on the same solver) would
  keep the duals, a warm start.
- MadNLP: `tol = OPF_TOLERANCE` (its default is 1e-8, on the scaled NLP
  error, as Ipopt's), `max_iter = OPF_MAX_ITERATIONS`, `linear_solver =
  MumpsSolver` (MadNLP's default sparse solver, set so a version bump cannot
  change it silently; the same Ipopt uses in the PowerModels image), its
  defaults otherwise: the full sparse KKT system, NLP scaling on.
  `print_level = ERROR`. Only `SOLVE_SUCCEEDED` counts as solved; any other
  status (`SOLVED_TO_ACCEPTABLE_LEVEL` included) is raised by name.
- Julia runs single-threaded (`PYTHON_JULIACALL_THREADS=1`); ExaPowerIO's
  info log on every parse is silenced.

What it deliberately does not do: the GPU backend (CUDA, MadNLPGPU and
cuDSS with the condensed KKT system, what the package is built for) is the
examodelspower_gpu tool (adapters/examodelspower_gpu_opf_adapter.py), the same
adapter and Julia code on another backend, in the GPU tabs.
"""
from importlib import import_module
from importlib.metadata import version

import numpy as np

from adapters.optimizer_adapter import OPF_MAX_ITERATIONS, OPF_TOLERANCE, OptimizerAdapter
from adapters.solver_adapter import DidNotConverge, Solution
from cases.matpower import GEN_BUS, GEN_STATUS, parse_m
from cases.registry import CASES


class ExamodelspowerOptimizer(OptimizerAdapter):
    name = "examodelspower"
    display_name = "ExaModelsPower.jl (MadNLP)"
    color = "#e3494b"   # tools/palette.py EXAMODELSPOWER: slot 8's red, dashed (ExaPF's makers)
    package = "juliacall"
    julia = "adapters.examodelspower_julia"   # the backend: load, solve, versions
    modules = (julia,)
    language = "julia"
    settings = {"formulation": "polar", "solver": "MadNLP", "linear_solver": "mumps", "backend": "cpu",
                "init": "flat", "tol": OPF_TOLERANCE, "max_iter": OPF_MAX_ITERATIONS}

    def _jl(self):
        return import_module(self.julia)   # starts Julia on first use

    def version(self):
        return self.dependencies()["ExaModelsPower"]

    def dependencies(self):
        return self._jl().versions() | {"juliacall": version("juliacall")}

    def load(self, case):
        return self._jl().load(str(CASES[case]["file"]), OPF_TOLERANCE, OPF_MAX_ITERATIONS)

    def solve(self, model):
        iterations, status = self._jl().solve(model)
        if status != "SOLVE_SUCCEEDED":
            raise DidNotConverge(f"MadNLP: {status} after {iterations} iterations")
        self._iterations = int(iterations)

    def solution(self, model, case):
        buses, vm, va, rows, gen_buses, pg, qg = self._jl().GB.solution(model)
        gen = parse_m(CASES[case]["file"])["gen"]
        in_service = np.flatnonzero(gen[:, GEN_STATUS] != 0)   # ExaPowerIO drops status == 0
        assert len(in_service) == len(rows)
        rows = [int(in_service[int(r)]) for r in rows]
        assert all(int(gen[r, GEN_BUS]) == int(b) for r, b in zip(rows, gen_buses))
        buses = [str(b) for b in buses]
        return Solution(dict(zip(buses, map(float, vm))), dict(zip(buses, map(float, va))), self._iterations,
                        pg_mw={str(r): float(p) for r, p in zip(rows, pg)},
                        qg_mvar={str(r): float(q) for r, q in zip(rows, qg)})
