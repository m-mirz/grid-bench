"""VeraGrid: `OptimalPowerFlowDriver` with `SolverType.NONLINEAR_OPF`,
AC-OPF with VeraGrid's own interior-point solver (pure Python, numba), no
external solver.

Input: the `.m` through `parse_matpower_file`, as the power-flow adapter
reads it; it imports the quadratic costs (`Cost0`, `Cost`, `Cost2`) and
the P and Q limits. Joins: buses by `Bus.code` (MATPOWER number);
generators by name, which the parser sets to `gen <row>` after the row in
the `.m`, asserted against the generator's bus.

How VeraGrid states the case's constraints (from its formulation,
`Simulations/OPF/Formulations/ac_opf_problem.py`), reported by the oracle
where it differs, not changed:
- Branch limits: |S|^2 <= rate^2 at both ends, on monitored branches (the
  parser monitors every branch): the case's MVA limit.
- Angle-difference limits: none (the formulation keeps bus angle limits
  and notes they are "not really used"), so `angmin`/`angmax` are not
  enforced (the `__sad` cases bind them).

Settings (`OptimalPowerFlowOptions`):
- `solver=NONLINEAR_OPF`, `ips_method=NR`: the AC-OPF, Newton steps.
- `acopf_mode=ACOPFstd`: hard limits. `ACOPFslacks` would soften branch and
  voltage limits with penalised slack variables, a different problem.
- `ips_init_with_pf=False`: the interior start the formulation describes as
  MATPOWER's (flat angles, variables at the middle of their bounds), not a
  power-flow solution.
- `ips_control_q_limits=False`: generator Q limits are always constraints;
  True adds a capability curve (P^2 + Q^2 <= (V * Inom)^2) the case has
  not got.
- `ips_tolerance=OPF_TOLERANCE` (its default is 1e-4),
  `ips_iterations=OPF_MAX_ITERATIONS`.

Result: the optimum on case14 (typical and congested) and case118, to the
reference's rounding. The `__sad` cases break their angle limits (by about
1 degree on case14, 22 % cheaper than the reference). On case300 it
reports convergence at the common tolerance with 0.17 MVA of power
imbalance at a few generator buses; at 1e-9 the imbalance drops to 3e-4
MVA and at 1e-11 to 3e-6, and the cost to the reference's: its tolerance
is not on the power mismatch as MATPOWER's is, and at 1e-6 it stops early.
Its power flow solves case300 exactly, so the network is read correctly.
"""
import numpy as np

from adapters.optimizer_adapter import OPF_MAX_ITERATIONS, OPF_TOLERANCE, OptimizerAdapter
from adapters.solver_adapter import DidNotConverge, Solution
from adapters.veragrid_adapter import VeragridAdapter
from cases.matpower import GEN_BUS, GEN_STATUS, parse_m
from cases.registry import CASES


class VeragridOptimizer(OptimizerAdapter):
    name = VeragridAdapter.name
    display_name = VeragridAdapter.display_name
    color = VeragridAdapter.color
    package = VeragridAdapter.package
    language = VeragridAdapter.language
    modules = ("VeraGridEngine",)
    settings = {"solver": "NONLINEAR_OPF", "ips_method": "NR", "acopf_mode": "ACOPFstd", "init": "interior (flat)",
                "q_capability_curve": False, "angle_limits": False, "ips_tolerance": OPF_TOLERANCE,
                "ips_iterations": OPF_MAX_ITERATIONS}

    def load(self, case):
        import VeraGridEngine as vg
        from VeraGridEngine.enumerations import AcOpfMode
        path = CASES[case]["file"]
        grid, _ = vg.parse_matpower_file(str(path))
        gen = parse_m(path)["gen"]
        index = {g.name: k for k, g in enumerate(grid.generators)}
        rows = []
        for row in np.flatnonzero(gen[:, GEN_STATUS] > 0):
            k = index[f"gen {row}"]
            assert int(grid.generators[k].bus.code) == int(gen[row, GEN_BUS])
            rows.append((str(row), k))
        options = vg.OptimalPowerFlowOptions(
            solver=vg.SolverType.NONLINEAR_OPF, ips_method=vg.SolverType.NR, acopf_mode=AcOpfMode.ACOPFstd,
            ips_init_with_pf=False, ips_control_q_limits=False, ips_tolerance=OPF_TOLERANCE,
            ips_iterations=OPF_MAX_ITERATIONS)
        return {"driver": vg.OptimalPowerFlowDriver(grid, options), "codes": [str(b.code) for b in grid.buses],
                "gens": rows}

    def solve(self, model):
        model["driver"].run()
        if not model["driver"].results.converged:
            raise DidNotConverge(f"NONLINEAR_OPF did not converge in {OPF_MAX_ITERATIONS} iterations")

    def solution(self, model, case):
        r = model["driver"].results
        v = np.asarray(r.voltage)
        p, q = np.asarray(r.generator_power), np.asarray(r.generator_reactive_power)
        return Solution(dict(zip(model["codes"], np.abs(v))), dict(zip(model["codes"], np.rad2deg(np.angle(v)))),
                        pg_mw={row: float(p[k]) for row, k in model["gens"]},
                        qg_mvar={row: float(q[k]) for row, k in model["gens"]})
