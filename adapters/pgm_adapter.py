"""power-grid-model: C++ Newton-Raphson.

Input: PGM JSON produced by `cases.prep` with `gridoxide.matpower.convert`
(PGM has no MATPOWER importer). Its known loss, which the oracle reports
rather than hides: PGM's transformer `clock` cannot hold a continuous phase
shift, so every MATPOWER phase shift is rounded to zero (PEGASE and RTE
cases).

The slack is a PGM `source`, an ideal voltage behind an impedance, which
`cases.prep` (`_ideal_source`) makes the `.m`'s slack. The converter's sk =
1e10 VA left it 1e-3 to 0.4 p.u. off its setpoint (0.4 on case9241pegase,
whose slack carries 2.6 GW): sk is raised to 1e40, an ideal slack to machine
precision with nothing else changed (checked from 1e10 to 1e40). The
converter took `u_ref` from the bus's `Vm` column, which MATPOWER never
reads at a generator bus: it is set to the generators' `Vg` (case4_dist and
case18 were solved 0.05 p.u. low, case3120sp 0.04). PGM's own power balance is exact on
  every distribution case (1e-11 MW); it is the conversion that fails.
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
- `calculation_initialization="flat"` (since 1.13.185): every node at 1
  p.u. with the source's angle plus its topological phase shift, regulated
  nodes at their `u_ref`: the common flat start. PGM's default starts Newton-
  Raphson from a linear voltage guess (every load and generator as a constant
  admittance), which diverges on meshed transmission grids that a flat
  start solves.

Result: from the flat start PGM converges on every case (from the linear
guess, on none above case14 but case_illinois200), and with the ideal slack
it is exact (1e-13 to 1e-9 MVA) on every case without phase shifts. The
PEGASE and RTE cases are rejected on exactly the buses next to the shifts
dropped by the converter (the zero-shift residual is 1e-9 MVA).
`average_source` converges on the same cases.
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
                "init": "flat (calculation_initialization)",
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
                error_tolerance=TOLERANCE_PU, max_iterations=MAX_ITERATIONS, calculation_initialization="flat",
                experimental_features="enabled")
        except PowerGridError as e:
            raise DidNotConverge(f"{type(e).__name__}: {str(e).splitlines()[0]}") from e

    def solution(self, model, case):
        node = model["result"]["node"]
        ids = [str(i) for i in model["node_ids"]]
        return Solution(dict(zip(ids, node["u_pu"])), dict(zip(ids, np.rad2deg(node["u_angle"]))))
