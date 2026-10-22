"""gridoxide: Rust Newton-Raphson with SuiteSparse KLU, driven through its
Python bindings.

Input: `PowerFlowModel.from_matpower` on the raw `.m`, gridoxide's own
MATPOWER reader (`src/matpower.rs`), which solves the problem MATPOWER's
`runpf` does: reference and PV buses only with an online generator, the
first PV bus as reference if none remains, regulated buses at their
generators' `Vg`, isolated buses and out-of-service equipment left out.
gridoxide also reads MATPOWER through PGM JSON (`gridoxide.matpower`, what
power-grid-model is given here); that path puts the reference bus behind a
source impedance and rounds phase shifts to 30 degrees, so it is not used.
The distribution feeders are read from their plain-data copies: the reader
refuses a case that edits its matrices with MATLAB code rather than solve
the unconverted numbers. gridoxide also reads CGMES; that family is not run
yet, since this adapter was added for the MATPOWER comparison only.

Settings:
- `backend="klu"`: SuiteSparse KLU through FFI, the solver lightsim2grid and
  p3s factor with, so the columns differ by their code. `GRIDOXIDE_BACKEND`
  picks another of gridoxide's backends for a comparison run (`klu_native`,
  its Rust port of KLU). The symbolic
  factorization is cached on the model and reused by every solve, as
  lightsim2grid reuses its own.
- `solve(init="flat")`: |V| = 1 p.u. and zero angle at every PQ bus, the
  setpoints elsewhere, on every solve. gridoxide's default starts from a
  constant-admittance linear solve instead.
- `tol=TOLERANCE_PU`: gridoxide's tolerance is the largest active or
  reactive power mismatch, per unit on the case's own `baseMVA`, the
  quantity `TOLERANCE_PU` is defined in. `max_iter=MAX_ITERATIONS`.
- No reactive limits, tap control, remote voltage control or distributed
  slack: `solve`'s outer loops are all off by default and left off.
- Built from a gridoxide commit, not a release (tool-configs/gridoxide/
  Dockerfile): the reader and the flat start are newer than the last
  release. The commit is in `settings`. Release profile, no target-cpu.
- Iterations are not reported: the Python model does not expose them.
"""
import os

import numpy as np

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import CASES


class GridoxideAdapter(SolverAdapter):
    name = "gridoxide"
    display_name = "gridoxide (KLU)"
    color = "#8b3201"   # tools/palette.py: Rust brown, base
    package = "gridoxide"
    modules = ("gridoxide",)
    language = "rust"
    families = ("matpower", "distribution")
    backend = os.environ.get("GRIDOXIDE_BACKEND", "klu")
    settings = {"backend": backend, "init": "flat", "reader": "from_matpower (.m)",
                "tolerance_pu": TOLERANCE_PU, "max_iteration": MAX_ITERATIONS,
                "commit": os.environ.get("GRIDOXIDE_COMMIT", "unknown"),
                "build": "release profile, rustc 1.95, SuiteSparse KLU by cc at opt-level 3, no target-cpu"}

    def load(self, case):
        import gridoxide
        model = gridoxide.PowerFlowModel.from_matpower(
            str(CASES[case]["file"]), backend=self.backend, tol=TOLERANCE_PU, max_iter=MAX_ITERATIONS)
        return {"model": model}

    def solve(self, model):
        try:
            model["model"].solve(init="flat")
        except RuntimeError as e:
            raise DidNotConverge(str(e)) from e

    def solution(self, model, case):
        m = model["model"]
        ids = [str(b) for b in m.bus_numbers()]
        vm, va = m.voltage_mag(), m.voltage_ang()
        if len(vm) != len(ids):
            raise RuntimeError(f"gridoxide has {len(vm)} buses, the case {len(ids)}")
        return Solution(dict(zip(ids, vm)), dict(zip(ids, np.rad2deg(va))))
