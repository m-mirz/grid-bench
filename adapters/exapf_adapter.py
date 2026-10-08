"""ExaPF: ExaPF.jl's polar Newton-Raphson (Argonne's Exanauts, Julia),
here on its CPU backend (KernelAbstractions `CPU()`, KLU sparse LU), driven
from Python through juliacall, which runs Julia inside the benchmark process:
a call costs microseconds, so the timing is Julia's. ExaPF is written for
GPUs (CUDA with cuDSS, ROCm); the CPU backend runs the same code, so this
adapter checks its model before any GPU number exists.

Input: `ExaPF.PowerFlowProblem(<case>.m, CPU(), :polar)` on the original
`.m` (ExaPF's own MATPOWER parser and a port of MATPOWER's makeYbus), the
persistent model every solve reuses: the Jacobian's sparsity pattern and
colouring (ExaPF differentiates the power balance with ForwardDiff, not by
hand) and the KLU factorization, refactored in place each step. ExaPF
indexes buses by their row in the bus matrix; `bus_to_indexes` maps the
MATPOWER bus number to it, asserted in `solution`. No CGMES importer, so the
cgmes families are not run.

The Julia side is `GridBenchExaPF` (tool-configs/exapf/julia/), compiled
into the image with a precompile workload, as for Sienna.

Known losses, reported by the oracle rather than hidden (v0.13.0):
- A PQ bus with an online generator is solved as PV at that generator's Vg
  (`PS.bustypeindex`); MATPOWER keeps it PQ and injects the generator's Qg,
  which no ExaPF equation holds (the balance has loads as constants and
  only Pg as a generator term). case2848rte: the residual fails exactly its
  48 such buses (236 MVAr), P is within 4e-7 MW. Like MATPOWER and
  Sparlectra, ExaPF also reaches the case's low-voltage root from flat
  start there (0.020 p.u.), which the oracle's voltage floor rejects too.
  case1888rte and case6495rte have such buses too, but diverge first, as
  they do for every tool except power-grid-model.
- `.m` files without `mpc.gencost` fail to import (`KeyError: key
  "gencost" not found`, parse_mat.jl): the parser requires it, so the
  "no costs" fallback in `PowerNetwork` is never reached. A power flow never
  uses costs; case4_dist and the mvlv grids have none.
- A PV bus without an online generator is solved as PQ: MATPOWER's own
  `bustypes` rule and the one the oracle applies, so no loss.

Settings (`PowerFlowProblem`, `NewtonRaphson`):
- `:polar`, one scenario: the single power flow. `:block_polar` is the
  batch formulation, for the batch benchmark.
- Linear solver: ExaPF's CPU default, `DirectSolver` on KLU (asserted at
  load). No Krylov solver, which ExaPF offers for GPUs.
- Flat start on every solve: ExaPF starts from the voltages in its stack,
  i.e. the previous solution, so `solve!` first restores every angle to 0,
  slack and PV magnitudes to their generators' Vg (what ExaPF itself
  initialises them to) and every other bus to 1 p.u. ExaPF's own start is
  the case's VM/VA columns, a hot start for a solved `.m`.
- `rtol=TOLERANCE_PU`: ExaPF stops when the 2-norm of the mismatch vector
  (P at PV and PQ buses, Q at PQ buses, p.u.) is below it. That is at least
  the infinity norm the other tools use, so the criterion is slightly
  stricter, never looser. ExaPF checks at the top of each of its `maxiter`
  passes, so it gets `MAX_ITERATIONS + 1` for `MAX_ITERATIONS` Newton steps.
- Single slack, no reactive limits and no outer loop: ExaPF's power flow
  has none of these.
- Julia runs single-threaded (`PYTHON_JULIACALL_THREADS=1`); info and
  warning logs off: the parser logs at info level on every field outside
  MATPOWER's core matrices, which would be timed. Errors still print.
"""
from importlib.metadata import version

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import CASES


def _gb():
    from adapters.exapf_julia import GB   # starts Julia on first use
    return GB


class ExapfAdapter(SolverAdapter):
    name = "exapf"
    display_name = "ExaPF.jl"
    color = "#e34949"   # tools/palette.py EXAPF: Sienna's red, dotted
    package = "juliacall"
    modules = ("adapters.exapf_julia",)
    language = "julia"
    families = ("matpower", "distribution")
    settings = {"solver": "NewtonRaphson", "formulation": "polar", "backend": "cpu", "linear_solver": "klu",
                "init": "flat", "reactive_limits": False, "distributed_slack": False,
                "tolerance_pu": TOLERANCE_PU, "tolerance_norm": 2, "max_iteration": MAX_ITERATIONS}

    def version(self):
        return self.dependencies()["ExaPF"]

    def dependencies(self):
        from adapters.exapf_julia import JULIA_VERSION
        return {k: str(v) for k, v in _gb().versions().items()} | {"julia": JULIA_VERSION,
                                                                    "juliacall": version("juliacall")}

    def load(self, case):
        return _gb().load(str(CASES[case]["file"]), TOLERANCE_PU, MAX_ITERATIONS)

    def solve(self, model):
        converged, steps = _gb().solve_b(model)   # juliacall spells solve! as solve_b
        if not converged:
            raise DidNotConverge(f"NR did not converge in {MAX_ITERATIONS} iterations")
        self._iterations = int(steps)

    def solution(self, model, case):
        numbers, vm, va = _gb().solution(model)
        ids = [str(int(n)) for n in numbers]
        return Solution(dict(zip(ids, map(float, vm))), dict(zip(ids, map(float, va))), self._iterations)
