"""Sparlectra: `runse!`'s weighted least squares (Julia, through juliacall;
the Julia half is `GridBenchSparlectra` `se.jl` in tool-configs/sparlectra).

Input: the `.m` exactly as the power-flow adapter imports it
(`load_matpower`, every import option explicit), plus one Sparlectra
`Measurement` per row of the case's measurement set: `VmMeas` (p.u.),
`PinjMeas` / `QinjMeas` (MW / MVAr, positive into the network), `PflowMeas`
/ `QflowMeas` with `direction = :from`, each with the case's sigma. Joins:
buses by MATPOWER number (`busOrigIdxDict`); branches by `.m` row, which the
importer adds one per row in order, so row r is `branchVec[r + 1]`, asserted
against the measurement's from and to bus.

Settings (`StateEstimationConfig`, passed explicitly, so no configuration
file can change them):
- `method = :wls`: plain weighted least squares, Gauss-Newton.
- `flatstart = true`, and every node reset to 1 p.u., 0 degrees before each
  estimate: the flat start takes the slack's magnitude from its node, which
  `update_net` overwrote with the previous estimate.
- `max_eliminations = 0`, `robust_mode = :off`, `robust = false`: no bad-data
  elimination (the default removes up to 3 rows) and no down-weighting.
- `update_shunts = false`, `update_taps = false`: no parameter estimation.
- `topology_precheck = false`: advisory only, it would log per run.
- `tol = SE_TOLERANCE`, `max_iter = MAX_ITERATIONS`. Sparlectra builds its
  Jacobian by forward finite differences (`jac_eps = 1e-6`, its default)
  and raises any tolerance below that step to it, with a log line: the
  fixed point is only located to about 1e-6, at the oracle's threshold
  (`oracle.wls.STEP_OK`), so this tool is graded right at its own limit.
"""
import numpy as np

from adapters.estimator_adapter import SE_TOLERANCE, EstimatorAdapter
from adapters.solver_adapter import MAX_ITERATIONS, DidNotConverge, Solution
from adapters.sparlectra_adapter import SparlectraAdapter, _gb
from cases import measurements
from cases.registry import CASES, measurements_path

KINDS = ("vm", "p_inj", "q_inj", "p_from", "q_from")   # se.jl's SE_TYPE, by position


class SparlectraEstimator(EstimatorAdapter):
    name = SparlectraAdapter.name
    display_name = SparlectraAdapter.display_name
    color = SparlectraAdapter.color
    package = SparlectraAdapter.package
    language = SparlectraAdapter.language
    modules = SparlectraAdapter.modules
    settings = {"method": "wls", "init": "flat", "bad_data": False, "robust": False, "parameter_estimation": False,
                "tol": SE_TOLERANCE, "jac": "finite differences, jac_eps=1e-6", "max_iter": MAX_ITERATIONS}
    version = SparlectraAdapter.version
    dependencies = SparlectraAdapter.dependencies

    def load(self, case):
        ms = measurements.read(measurements_path(case))["measurements"]
        column = lambda key, default: np.array([m.get(key, default) for m in ms], dtype=np.int64)
        return _gb().load_se(
            str(CASES[case]["file"]), np.array([KINDS.index(m["kind"]) for m in ms], dtype=np.int64),
            column("bus", 0), column("branch_row", -1), column("from_bus", 0), column("to_bus", 0),
            np.array([m["value"] for m in ms]), np.array([m["sigma"] for m in ms]), SE_TOLERANCE, MAX_ITERATIONS)

    def solve(self, model):
        iterations = _gb().estimate_b(model)   # juliacall spells estimate! as estimate_b
        if iterations < 0:
            raise DidNotConverge(f"runse! did not converge in {MAX_ITERATIONS} iterations")
        self._iterations = int(iterations)

    def solution(self, model, case):
        ids, vm, va = _gb().se_solution(model)
        ids = [str(i) for i in ids]
        return Solution(dict(zip(ids, map(float, vm))), dict(zip(ids, map(float, va))), self._iterations)
