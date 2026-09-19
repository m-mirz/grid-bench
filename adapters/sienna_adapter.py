"""Sienna: PowerFlows.jl's AC Newton-Raphson on a PowerSystems.jl `System`
(NREL's Sienna platform, Julia, KLU sparse LU), driven from Python through
juliacall, which runs Julia inside the benchmark process: a call costs
microseconds, so the timing is Julia's.

Input: `PowerSystems.System(<case>.m)` on the original `.m` (PowerSystems'
own MATPOWER parser), then `PowerFlows.PowerFlowData`, the persistent model
every solve reuses. PowerSystems keeps the MATPOWER bus numbers, and
PowerFlows' bus lookup is keyed by them. No CGMES importer, so the cgmes
families are not run.

The Julia side is `GridBenchSienna` (tool-configs/sienna/julia/), compiled
into the image with a precompile workload: without it, every fresh process
spends ~95 s in the JIT on its first case, and the memory measurement would
be the compiler's.

Known losses, reported by the oracle rather than hidden:
- Ybus is single precision: PowerNetworkMatrices declares
  `const YBUS_ELTYPE = ComplexF32` (v0.24.3). PowerFlows converges to
  `TOLERANCE_PU` on that rounded matrix, which leaves residuals of 1e-5 to
  1e-3 MW against the exact one (1e-9 for the other tools), growing with the
  size of the grid: mvlv29840 misses the 1e-3 MVA acceptance limit on this
  alone.
- Transformer line charging: PowerSystems' MATPOWER parser keeps only the
  from end's half (`b_fr`) as the transformer's `primary_shunt` and drops
  the to end's (power_models_data.jl). case118 (branch 68-116: 8.3 MVAr at
  bus 68), case300 (18 such transformers), case3120sp (52) and case2848rte
  (2,731: every branch is a transformer) are solved without it; each case's
  worst bus is at such a transformer.
- Pure phase shifters (tap ratio 0, shift not 0, in every PEGASE case) fail
  to parse: `KeyError: key "base_voltage_from" not found`; the parser only
  sets that key on branches it has already classified as transformers.

Settings (`ACPowerFlow{NewtonRaphsonACPowerFlow}`):
- `check_reactive_power_limits=false` (default), no participation factors
  (default: the reference bus takes all slack): no reactive limits, single
  slack.
- `correct_bustypes=true`: a PV bus without an available generator is
  solved as PQ, MATPOWER's own `bustypes` rule and the one the oracle
  applies. With the default (false) PowerFlows refuses such a case outright
  ("No available sources at bus ... Please change the bus type to PQ"),
  e.g. case3120sp and every RTE case.
- `enhanced_flat_start=false`: PowerFlows otherwise replaces a flat start
  with a heuristic one when the initial mismatch is large.
  `robust_power_flow=false` (default): no DC power flow fallback start.
- Flat start on every solve: PowerFlows starts from the voltages held in its
  data, i.e. the previous solution, so `solve!` first restores PQ buses to
  1 p.u. and every angle to 0 (PV and slack magnitudes are their setpoints).
- `tol=TOLERANCE_PU`: PowerFlows stops when the infinity norm of the power
  mismatch in p.u. is below it, the same criterion as the others.
  `MAX_ITERATIONS` Newton steps.
- Julia runs single-threaded (`PYTHON_JULIACALL_THREADS=1`); info and
  warning logs are off, since PowerFlows logs on every solve.
"""
from importlib.metadata import version

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import CASES


def _gb():
    from adapters.sienna_julia import GB   # starts Julia on first use
    return GB


class SiennaAdapter(SolverAdapter):
    name = "sienna"
    display_name = "Sienna (PowerFlows.jl)"
    color = "#e34948"
    package = "juliacall"
    modules = ("adapters.sienna_julia",)
    language = "julia"
    families = ("matpower", "distribution")
    settings = {"solver": "NewtonRaphsonACPowerFlow", "init": "flat", "enhanced_flat_start": False,
                "reactive_limits": False, "distributed_slack": False, "tolerance_pu": TOLERANCE_PU,
                "max_iteration": MAX_ITERATIONS}

    def version(self):
        return self.dependencies()["PowerFlows"]

    def dependencies(self):
        from adapters.sienna_julia import JULIA_VERSION
        return {k: str(v) for k, v in _gb().versions().items()} | {"julia": JULIA_VERSION,
                                                                    "juliacall": version("juliacall")}

    def load(self, case):
        return _gb().load(str(CASES[case]["file"]))

    def solve(self, model):
        if not _gb().solve_b(model, TOLERANCE_PU, MAX_ITERATIONS):   # juliacall spells solve! as solve_b
            raise DidNotConverge(f"NR did not converge in {MAX_ITERATIONS} iterations")

    def solution(self, model, case):
        numbers, vm, va = _gb().solution(model)
        ids = [str(int(n)) for n in numbers]
        return Solution(dict(zip(ids, map(float, vm))), dict(zip(ids, map(float, va))))
