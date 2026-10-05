"""power-grid-model: C++ Newton-Raphson.

Input: PGM JSON produced by `cases.prep` with `gridoxide.matpower.convert`
(PGM has no MATPOWER importer). Known losses of that conversion, both of
which the oracle reports rather than hides:
- PGM's transformer `clock` cannot hold a continuous phase shift, so every
  MATPOWER phase shift is rounded to zero (PEGASE and RTE cases).
- The slack is a PGM `source`: an ideal voltage behind an impedance
  (sk = 1e10 VA), not an ideal slack bus, so the slack voltage lands off its
  setpoint (visible as `max_dvm_pu`): slightly on transmission cases, by
  2-3% on the heavily loaded 150 kV slack of the generated MV/LV grids
  (2.25 ohm carrying ~1 kA).
- The source's `u_ref` is the slack bus's `Vm` column, where MATPOWER's
  setpoint is the generator's `Vg`: case4_dist and case18 (Vm 1, Vg 1.05)
  are solved 0.05 p.u. low throughout. PGM's own power balance is exact on
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
guess, on none above case14 but case_illinois200); every rejection is the
conversion's (the slack off its setpoint, phase shifts dropped: the
zero-shift residual is 1e-9 MVA on PEGASE and RTE). `average_source`
converges on the same cases and agrees except on case6495rte, where its
slack lands at its setpoint and the flat start's at 0.659 p.u.
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
