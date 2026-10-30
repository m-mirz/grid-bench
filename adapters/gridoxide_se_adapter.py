"""gridoxide: `StateEstimationModel`, Gauss-Newton on the normal equations
(`newton_raphson`), in Rust, driven through its Python bindings.

Input: `StateEstimationModel.from_matpower` on the raw `.m`, gridoxide's own
MATPOWER reader, the one the power-flow adapter (`adapters/gridoxide_adapter.py`)
uses, and the case's measurement set (`cases.measurements`) as given: each
measurement is `(kind, at, value, sigma)` with the set's own kind, the bus
by its MATPOWER number or the branch by its row in the `.m`'s branch matrix,
and the value and sigma in the set's units (|V| p.u., MW, MVAr). gridoxide
reads the same conventions the set is written in: an injection is
generation less load, positive into the network, with the shunt part of the
network; a from-end flow is positive out of the from bus. The join is by bus
number and branch row, never by position; gridoxide refuses a measurement
naming a bus or a row it does not have, and `nodes()` returns each bus
number with its index for the solution.

No converter stands in between, so the network is exactly the case's: phase
shifts unrounded, transformer line charging where the case puts it.
power-grid-model's estimator, given the PGM conversion, loses both (see
`adapters/pgm_se_adapter.py`).

Settings:
- `method="newton_raphson"`: WLS Gauss-Newton on the nonlinear measurement
  functions.
- The case's reference bus is the angle reference, its magnitude estimated
  like any other bus's. No bus is held at zero injection: a bus without load
  or generator injects what its measurements say (gridoxide's PGM loader,
  by default, follows power-grid-model and constrains such buses).
- `tol=SE_TOLERANCE` on the largest state update, `max_iter=MAX_ITERATIONS`.
- `backend`: the power-flow adapter's (`GRIDOXIDE_BACKEND`, default `klu`).
  The estimator keeps its symbolic factorization between solves.
- Start (hardcoded, not settable from Python), on every `solve()`: of a
  linear solve of the network and the flat state, each with and without the
  measured magnitudes applied, the one the measurements fit best among those
  keeping every bus within 0.5 to 1.5 p.u.; not always flat.

Iterations are not reported: the Python model does not expose them for a
single solve.

Results: see the benchmark JSON; first host run in the commit message that
switched this adapter to the `.m`.

Found and fixed in gridoxide on the way here (branch se-node-injection),
while gridoxide read power-grid-model's document: a node-injection sensor
counted the bus's shunt as injected; the power derivative's leading-factor
term was added once per coefficient on the measured bus; the start, built
from the document's power-flow schedule and the measured magnitudes, began
case118~noisy at |V| = 0.67 p.u. and case3120sp~noisy and
case9241pegase~noisy where Gauss-Newton diverged. On that document gridoxide
needed two new options to solve the benchmark's problem, `zero_injection` and
`ideal_sources`, and with the zero-injection constraints on it factors
`[HᵀWH Cᵀ; C 0]`, which KLU orders badly (case9241pegase~exact 2.9 s against
0.5 s without).
"""
import os

import numpy as np

from adapters.estimator_adapter import SE_TOLERANCE, EstimatorAdapter
from adapters.gridoxide_adapter import GridoxideAdapter
from adapters.solver_adapter import MAX_ITERATIONS, DidNotConverge, Solution
from cases import measurements
from cases.registry import CASES, measurements_path


class GridoxideEstimator(EstimatorAdapter):
    name = GridoxideAdapter.name
    display_name = GridoxideAdapter.display_name
    color = GridoxideAdapter.color
    package = GridoxideAdapter.package
    language = GridoxideAdapter.language
    modules = GridoxideAdapter.modules
    backend = GridoxideAdapter.backend
    settings = {"method": "newton_raphson", "backend": backend, "reader": "from_matpower (.m)",
                "init": "best plausible of linear and flat (hardcoded)", "tol": SE_TOLERANCE,
                "max_iter": MAX_ITERATIONS, "commit": os.environ.get("GRIDOXIDE_COMMIT", "unknown")}

    def load(self, case):
        import gridoxide
        ms = [(m["kind"], m["bus"] if "bus" in m else m["branch_row"], m["value"], m["sigma"])
              for m in measurements.read(measurements_path(case))["measurements"]]
        model = gridoxide.StateEstimationModel.from_matpower(
            str(CASES[case]["file"]), ms, backend=self.backend, method="newton_raphson",
            tol=SE_TOLERANCE, max_iter=MAX_ITERATIONS)
        return {"model": model}

    def solve(self, model):
        try:
            model["model"].solve()
        except RuntimeError as e:
            raise DidNotConverge(str(e)) from e

    def solution(self, model, case):
        m = model["model"]
        vm, va = m.voltage_mag(), np.rad2deg(m.voltage_ang())
        nodes = m.nodes()
        return Solution({str(i): vm[k] for i, k in nodes}, {str(i): va[k] for i, k in nodes})
