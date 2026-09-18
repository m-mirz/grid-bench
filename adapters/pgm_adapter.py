"""power-grid-model: C++ Newton-Raphson.

Input: PGM JSON produced by `cases.prep` with `gridoxide.matpower.convert`
(PGM has no MATPOWER importer). Known losses of that conversion, both of
which the oracle reports rather than hides:
- PGM's transformer `clock` cannot hold a continuous phase shift, so every
  MATPOWER phase shift is rounded to zero (PEGASE and RTE cases).
- The slack is a PGM `source`: an ideal voltage behind a tiny impedance
  (sk = 1e10), not an ideal slack bus, so the slack voltage lands slightly
  off its setpoint (visible as `max_dvm_pu`).
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

Result: PGM's experimental PV support diverges from case118 upward (same as
gridoxide's bench records for power-grid-model 1.13.120); it converges on
case14 and case_illinois200.
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
    families = ("matpower",)
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
