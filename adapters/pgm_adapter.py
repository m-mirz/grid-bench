"""power-grid-model: C++ Newton-Raphson.

Input: PGM JSON produced by `cases.prep` with `gridoxide.matpower.convert`
(PGM has no MATPOWER importer). The conversion is exact: a phase-shifting
branch becomes a `generic_branch`, whose pi-model is MATPOWER's with a
continuous shift (a `transformer` clock would round it to 60 degrees, a
residual of 20 to 60,000 MW on the PEGASE and RTE cases).
The slack is a PGM `source` (an ideal voltage behind an impedance) at the
slack generator's `Vg` with sk = 1e18 VA, which makes it the ideal slack
MATPOWER defines. gridoxide 0.0.2 used the bus's `Vm` column (case4_dist and
case18: 0.05 p.u. low) and sk = 1e10 VA, 0.01 p.u. on a 100 MVA base, which
held every slack off its setpoint (up to 0.029 p.u. on the MV/LV grids) and
made case118 diverge. The vendored copy is corrected (lines marked
"grid-bench:" in cases/gridoxide_matpower.py): the input then states the
case's problem; it is not a fix of PGM.
No CGMES importer, so the cgmes family is not run.

Settings:
- Generators are PV buses through PGM's `voltage_regulator` component,
  which PGM gates behind `experimental_features="enabled"`, available only on
  the private `_calculate_power_flow` (the public `calculate_power_flow` does
  not accept that flag). Without regulators every generator is a PQ
  injection, a different problem.
- Reactive limits are stripped from the regulators by `cases.prep`.
- `error_tolerance=TOLERANCE_PU` (PGM's tolerance is on the voltage update,
  not the power mismatch), `max_iterations=MAX_ITERATIONS`.
- No flat start: PGM's Newton-Raphson has no initialization option and
  always starts from its own linear guess (newton_raphson_pf_solver.hpp,
  `initialize_derived_solver`, 1.13.172): one linear solve with every load
  and generator as the admittance -conj(S) at 1 p.u. (a generator is a
  negative conductance; a regulated one keeps only its P), then PV buses at
  their setpoint with the guess's angle. This breaks the common flat-start
  rule, and it cannot be configured.

Result: every case PGM converges on is accepted (case14, case118, all
default distribution cases). From case300 upward it fails, and the start
is the cause. The conversion is exact there: at known voltages, PGM's own
branch flows (state estimation with every voltage measured) equal
MATPOWER's to 1e-10 MVA on case300, case1354pegase and case2869pegase. A
textbook polar Newton-Raphson on the oracle's Ybus reproduces PGM's outcome
on all 16 cases when started from PGM's linear guess, and converges on all
of them from a flat start: it diverges on case300, case3120sp, case2848rte
and case1888rte (PGM: IterationDiverge), and hits an exactly singular
Jacobian on the three PEGASE cases and case6495rte (PGM: SparseMatrixError).
Removing the regulators is no control: with generators at Q = 0 even
case118 has no solution. With a flat start (a power-grid-model build with
that option, m-mirz/power-grid-model branch feature/newton-raphson-flat-start),
PGM is accepted on every case but case6495rte, which no tool solves.
"""
import numpy as np

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import pgm_json_path


class PgmAdapter(SolverAdapter):
    name = "pgm"
    display_name = "power-grid-model"
    color = "#eda100"
    package = "power-grid-model"
    modules = ("power_grid_model", "power_grid_model.utils", "power_grid_model.errors")
    language = "c++"
    families = ("matpower", "distribution")
    settings = {"calculation_method": "newton_raphson", "voltage_regulators": "experimental",
                "reactive_limits": False, "tolerance_pu": TOLERANCE_PU, "max_iteration": MAX_ITERATIONS}

    def load(self, case):
        from power_grid_model import PowerGridModel
        from power_grid_model.utils import json_deserialize
        dataset = json_deserialize(pgm_json_path(case).read_text())
        return {"model": PowerGridModel(dataset), "node_ids": dataset["node"]["id"], "result": None}

    def solve(self, model):
        from power_grid_model import CalculationMethod
        from power_grid_model.errors import PowerGridError
        try:
            model["result"] = model["model"]._calculate_power_flow(  # noqa: SLF001, see docstring
                calculation_method=CalculationMethod.newton_raphson, symmetric=True,
                error_tolerance=TOLERANCE_PU, max_iterations=MAX_ITERATIONS, experimental_features="enabled")
        except PowerGridError as e:
            raise DidNotConverge(f"{type(e).__name__}: {str(e).splitlines()[0]}") from e

    def solution(self, model, case):
        node = model["result"]["node"]
        ids = [str(i) for i in model["node_ids"]]
        return Solution(dict(zip(ids, node["u_pu"])), dict(zip(ids, np.rad2deg(node["u_angle"]))))
