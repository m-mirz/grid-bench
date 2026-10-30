"""gridoxide: AC optimal power flow (`OptimalPowerFlowModel`), its own
NLP model solved by IPOPT through gridoxide's FFI, in Rust, driven through
its Python bindings.

Input: `OptimalPowerFlowModel.from_matpower` on the raw `.m`, gridoxide's
own MATPOWER reader (the power-flow adapter's, `adapters/gridoxide_adapter.py`)
with `mpc.gencost` and the OPF columns. It builds MATPOWER's AC-OPF in polar
form: polynomial cost, voltage limits, generator P and Q limits, `RATE_A` as
an apparent-power limit at both branch ends (0 unlimited, written squared,
upper side only) and `ANGMIN`/`ANGMAX` as linear angle-difference rows (a
side ignored at 0 or beyond +-360 degrees, as MATPOWER's `makeAang`). Isolated
buses and out-of-service equipment are left out. Joins: buses by number
(`nodes()`), generators by their row in the `.m`'s gen matrix (`dispatch()`),
never by position.

No converter stands in between: gridoxide's PGM route (`gridoxide.matpower`
plus `ac_opf`) puts the reference bus behind a source impedance and rounds
phase shifts to 30 degrees, so it is not used.

Settings:
- `solver="ipopt"`: Ipopt 3.14.19 (JuliaGrid's), built from source in the
  image with sequential MUMPS (ThirdParty-Mumps 3.0.9) and the reference
  BLAS/LAPACK (tool-configs/gridoxide/Dockerfile; bookworm's package is
  3.11.9, older than gridoxide's bindings), linked by gridoxide's
  `opf-ipopt` feature. gridoxide supplies the exact Jacobian and Hessian.
  `tol = OPF_TOLERANCE`, `max_iter = OPF_MAX_ITERATIONS`, silent.
- The model is built once in `load` (timed as import) and solved by `solve`.
- Flat start, `solve(init="flat")` on every solve: |V| = 1 p.u. (clamped
  into its limits, a no-op on these cases) and zero angle on every bus, every
  generator's P and Q at the middle of its range. gridoxide's default instead
  spreads the load over the generators' P ranges; IPOPT needs it on none of
  these cases, gridoxide's own solver does.
- A solve IPOPT does not report as solved (or solved to an acceptable
  level) fails.

Result: the optimum on every default case, within the reference's rounding
and with PowerModels' gaps, in 14 to 49 Ipopt iterations (PowerModels: 14
to 45); first run in the commit message that added this adapter.

gridoxide's own interior point (`solver="nlp"`) is not benchmarked: it
stalls at `OPF_MAX_ITERATIONS` on case300_ieee, its `__sad` variant,
case118_ieee__api and case1354_pegase.

Found and fixed in gridoxide on the way here (branch ac-opf-matpower): no
angle-difference limits at all; no way to reach IPOPT from Python; the IPOPT
callbacks rebuilt a hash map of the sparsity pattern on every evaluation, and
iterations were not reported.
"""
import os

import numpy as np

from adapters.gridoxide_adapter import GridoxideAdapter
from adapters.optimizer_adapter import OPF_MAX_ITERATIONS, OPF_TOLERANCE, OptimizerAdapter
from adapters.solver_adapter import DidNotConverge, Solution
from cases.registry import CASES


class GridoxideOptimizer(OptimizerAdapter):
    name = GridoxideAdapter.name
    display_name = "gridoxide (Ipopt)"
    color = GridoxideAdapter.color
    package = GridoxideAdapter.package
    language = GridoxideAdapter.language
    modules = GridoxideAdapter.modules
    settings = {"formulation": "polar, from_matpower (.m)", "solver": "Ipopt 3.14.19", "linear_solver": "mumps",
                "init": "flat", "tol": OPF_TOLERANCE,
                "max_iter": OPF_MAX_ITERATIONS, "commit": os.environ.get("GRIDOXIDE_COMMIT", "unknown")}

    def load(self, case):
        import gridoxide
        model = gridoxide.OptimalPowerFlowModel.from_matpower(
            str(CASES[case]["file"]), solver="ipopt", tol=OPF_TOLERANCE, max_iter=OPF_MAX_ITERATIONS)
        return {"model": model}

    def solve(self, model):
        try:
            model["model"].solve(init="flat")
        except RuntimeError as e:
            raise DidNotConverge(str(e)) from e

    def solution(self, model, case):
        m = model["model"]
        vm, va = m.voltage_mag(), np.rad2deg(m.voltage_ang())
        nodes = m.nodes()
        dispatch = m.dispatch()
        return Solution({str(b): vm[i] for b, i in nodes}, {str(b): va[i] for b, i in nodes}, m.iterations,
                        pg_mw={str(row): p for row, p, _ in dispatch},
                        qg_mvar={str(row): q for row, _, q in dispatch})
