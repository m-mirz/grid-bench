"""power-grid-model: C++ Newton-Raphson.

Input: PGM JSON produced by `cases.prep` with `gridoxide.matpower.convert`
(PGM has no MATPOWER importer). Known loss of that conversion, which the
oracle reports rather than hides:
- PGM's transformer `clock` cannot hold a continuous phase shift, so every
  MATPOWER phase shift is rounded to zero (PEGASE and RTE cases).
The slack is a PGM `source` (an ideal voltage behind an impedance) at the
slack generator's `Vg` with sk = 1e15 VA, which makes it the ideal slack
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
- PGM always initializes from its own flat start.

Result: every case PGM converges on is accepted (case14, case118, all
default distribution cases). From case300 upward it diverges or reports a
singular matrix at any source sk from 1e8 to 1e15 and also without voltage
regulators, so neither the slack nor the experimental PV support explains
that; not yet traced.
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
