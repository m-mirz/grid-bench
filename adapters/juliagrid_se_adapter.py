"""JuliaGrid.jl: AC weighted least-squares state estimation, Gauss-Newton on
the normal equations (`gaussNewton(monitoring, KLU)`).

Input: the case's `.m` through JuliaGrid's reader, as the power-flow adapter
(`adapters/juliagrid_adapter.py`), with its asserted row join. Measurements
(`cases.measurements`): |V| as `addVoltmeter!`, P and Q injections as
`addWattmeter!` / `addVarmeter!` at the bus (JuliaGrid's bus meters measure
the injection, positive into the network, as the set does), P and Q
from-end flows as the same meters `from` the branch (its row, asserted by
its from and to bus). Values and sigmas in p.u. on the case's base power
(the set's MW and MVAr divided by `baseMVA`); JuliaGrid takes each
measurement's variance, sigma².

Settings:
- Gauss-Newton on the normal equations, KLU for the gain matrix: plain
  WLS, every measurement weighted by 1/variance.
- Flat start: every bus at 1 p.u. and 0 rad (the system's voltages, set
  flat at load; `setInitialPoint!` restores them before every estimate).
- The slack bus angle is the only reference (JuliaGrid fixes the slack's
  angle increment to 0 and estimates its magnitude like any other's).
- The loop of JuliaGrid's `stateEstimation!`, written out for its status:
  stop when the largest state increment is below `SE_TOLERANCE`, at most
  `MAX_ITERATIONS` steps.
- No pseudo-measurements, no bad-data test (`residualTest!` is not called),
  no observability analysis.

Results (first run): the oracle accepts every estimate, both scenarios on
every case (case14~exact: J 1.5e-20, step 1e-13).
"""
import numpy as np

from adapters.estimator_adapter import SE_TOLERANCE, EstimatorAdapter
from adapters.juliagrid_adapter import JuliagridAdapter, _gb, join, readable
from adapters.solver_adapter import MAX_ITERATIONS, DidNotConverge, Solution
from cases import measurements
from cases.matpower import F_BUS, T_BUS, parse_m
from cases.registry import CASES, measurements_path

KINDS = ("vm", "p_inj", "q_inj", "p_from", "q_from")


class JuliagridEstimator(EstimatorAdapter):
    name = JuliagridAdapter.name
    display_name = JuliagridAdapter.display_name
    color = JuliagridAdapter.color
    package = JuliagridAdapter.package
    language = JuliagridAdapter.language
    modules = JuliagridAdapter.modules
    settings = {"method": "wls", "solver": "gaussNewton (normal equations)", "linear_solver": "klu", "init": "flat",
                "bad_data": False, "pseudo_measurements": False, "tol": SE_TOLERANCE, "max_iter": MAX_ITERATIONS}
    version = JuliagridAdapter.version
    dependencies = JuliagridAdapter.dependencies

    def load(self, case):
        data = measurements.read(measurements_path(case))
        base, ms = float(data["baseMVA"]), data["measurements"]
        mpc = parse_m(CASES[case]["file"])
        row_of = {int(b): i + 1 for i, b in enumerate(mpc["bus"][:, 0])}
        kind, bus, branch, value, sigma = [], [], [], [], []
        for m in ms:
            k = KINDS.index(m["kind"])
            scale = 1.0 if k == 0 else base
            kind.append(k)
            bus.append(row_of[m["bus"]] if "bus" in m else 0)
            if "branch_row" in m:
                r = m["branch_row"]
                assert (int(mpc["branch"][r, F_BUS]), int(mpc["branch"][r, T_BUS])) == (m["from_bus"], m["to_bus"])
                branch.append(r + 1)
            else:
                branch.append(0)
            value.append(m["value"] / scale)
            sigma.append(m["sigma"] / scale)
        est = _gb().load_se(readable(CASES[case]["file"]), np.array(kind), np.array(bus), np.array(branch),
                            np.array(value), np.array(sigma), SE_TOLERANCE, MAX_ITERATIONS)
        return {"jl": est, "bus_ids": join(case, est.system), "iterations": None}

    def solve(self, model):
        steps = _gb().estimate_b(model["jl"])   # estimate!
        if steps < 0:
            raise DidNotConverge(f"Gauss-Newton did not converge in {MAX_ITERATIONS} iterations")
        model["iterations"] = int(steps)

    def solution(self, model, case):
        vm, va = _gb().se_solution(model["jl"])
        ids = model["bus_ids"]
        return Solution(dict(zip(ids, map(float, vm))), dict(zip(ids, map(float, va))), model["iterations"])
