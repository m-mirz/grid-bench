"""VeraGrid: `StateEstimationDriver`, Levenberg-Marquardt WLS.

Input: the `.m` through `parse_matpower_file`, as the power-flow adapter
reads it, plus VeraGrid measurement objects added to the MultiCircuit.
Joins:
- bus: `Bus.code` = MATPOWER bus number;
- branch: `parse_matpower_file` names each line or transformer
  `branch <row>` after its row in the `.m`; the join is that name, asserted
  against the branch's from and to bus codes.
Measurements: `VmMeasurement` (p.u.), `PiMeasurement` / `QiMeasurement`
(bus injection, MW / MVAr, positive into the network), `PfMeasurement` /
`QfMeasurement` (from end), each with the case's sigma as `uncertainty`.

Settings:
- `solver=SolverType.LM`: Gauss-Newton on the normal equations with
  Levenberg-Marquardt damping, converging on the state update, the closest
  of VeraGrid's three WLS solvers to the common problem. None of the three
  is both right and says so; measured at `MAX_ITERATIONS` (oracle verdict,
  and whether the solver claims convergence):
  - `LM`: the optimum on case14 and case118~noisy in 5-6 iterations; from
    case118~exact up it declares convergence 0.3-1 p.u./rad away from the
    optimum, as the damping shrinks the step below the tolerance. Those are
    the ✗ results.
  - `GN`: adds a Tikhonov term of 1e-6 of the gain matrix's largest
    diagonal to every step, which makes it converge linearly: 104
    iterations on case118~noisy, so everything past case14 is a failure.
  - `NR`: reaches the optimum on every case above except
    case1354pegase~exact, but tests the gradient norm, in weighted per
    unit, against the tolerance, which it never meets (not even in 2000
    iterations): every case would be recorded as not converged.
- Flat start: `Vm0 = 1`, `Va0 = 0` on every bus, since the solver starts
  from the bus voltages it compiles, and `parse_matpower_file` fills them
  from the `.m`'s VM/VA columns (often a power-flow solution).
- `fixed_slack=False`: the slack's |V| is a state like every other bus's,
  and the measurements at the slack are kept (True drops them).
- `run_observability_analyis=False`, `add_pseudo_measurements=False`,
  `run_measurement_profiling=False`: no measurements added.
- Bad data: the LM solver has none (its b-test is commented out in 6.5.29);
  `c_threshold` and `prefer_correct` are therefore unused.

Result: a case whose `baseMVA` is not 100 (every distribution feeder) is
estimated wrongly by every solver (case33bw~exact: J = 8395 where the
optimum is 0), the parser defect the power-flow adapter documents.
- `tol=SE_TOLERANCE`, `max_iter=MAX_ITERATIONS`.
"""
import numpy as np

from adapters.estimator_adapter import SE_TOLERANCE, EstimatorAdapter
from adapters.solver_adapter import MAX_ITERATIONS, DidNotConverge, Solution
from adapters.veragrid_adapter import VeragridAdapter
from cases import measurements
from cases.registry import CASES, measurements_path


class VeragridEstimator(EstimatorAdapter):
    name = VeragridAdapter.name
    display_name = VeragridAdapter.display_name
    color = VeragridAdapter.color
    package = VeragridAdapter.package
    language = VeragridAdapter.language
    modules = ("VeraGridEngine",)
    settings = {"solver": "LM", "init": "flat", "fixed_slack": False, "observability_analysis": False,
                "pseudo_measurements": False, "bad_data": False, "tol": SE_TOLERANCE, "max_iter": MAX_ITERATIONS}

    def load(self, case):
        import VeraGridEngine as vg
        grid, _ = vg.parse_matpower_file(str(CASES[case]["file"]))
        for bus in grid.buses:
            bus.Vm0, bus.Va0 = 1.0, 0.0
        bus_of = {int(b.code): b for b in grid.buses}
        branch_of = {int(b.name.removeprefix("branch ")): b
                     for b in grid.get_branches(add_vsc=False, add_hvdc=False, add_switch=True)}
        make = {"vm": (vg.VmMeasurement, grid.add_vm_measurement),
                "p_inj": (vg.PiMeasurement, grid.add_pi_measurement),
                "q_inj": (vg.QiMeasurement, grid.add_qi_measurement),
                "p_from": (vg.PfMeasurement, grid.add_pf_measurement),
                "q_from": (vg.QfMeasurement, grid.add_qf_measurement)}
        for m in measurements.read(measurements_path(case))["measurements"]:
            if "bus" in m:
                obj = bus_of[m["bus"]]
            else:
                obj = branch_of[m["branch_row"]]
                assert (int(obj.bus_from.code), int(obj.bus_to.code)) == (m["from_bus"], m["to_bus"])
            cls, add = make[m["kind"]]
            add(cls(value=m["value"], uncertainty=m["sigma"], api_obj=obj))
        options = vg.StateEstimationOptions(solver=vg.SolverType.LM, tol=SE_TOLERANCE, max_iter=MAX_ITERATIONS,
                                            fixed_slack=False, run_observability_analyis=False,
                                            add_pseudo_measurements=False, run_measurement_profiling=False)
        return {"driver": vg.StateEstimationDriver(grid, options), "codes": [str(b.code) for b in grid.buses]}

    def solve(self, model):
        model["driver"].run()
        results = model["driver"].results
        if not results.converged:
            raise DidNotConverge(f"Levenberg-Marquardt: not converged in {MAX_ITERATIONS} iterations")

    def solution(self, model, case):
        v = model["driver"].results.voltage
        return Solution(dict(zip(model["codes"], np.abs(v))), dict(zip(model["codes"], np.rad2deg(np.angle(v)))))
