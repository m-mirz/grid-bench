"""PyPSA: `Network.pf()`, full AC Newton-Raphson (scipy sparse LU).

Input: the prepared `.mat`, read with scipy and passed to
`import_from_pypower_ppc`. PyPSA has no MATPOWER file reader, so the
`loadmat` call is part of its timed import. No CGMES importer, so the cgmes
family is not run.

Settings:
- `transformers.model = "pi"`: PyPSA imports transformers with its default
  T-model, while MATPOWER's branch model is a pi-model. With the T-model the
  oracle shows P residuals of several MW on case300; with "pi" it is at
  solver tolerance. This selects the problem the case defines, not a solver
  tweak.
- `pf(x_tol=TOLERANCE_PU, use_seed=False)`: the common tolerance, and a flat
  start on every solve rather than seeding from the previous result.
- PyPSA's AC power flow has no reactive limits or distributed slack to
  disable in this call.
"""
import logging

import numpy as np

from adapters.solver_adapter import TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import mat_path


class PypsaAdapter(SolverAdapter):
    name = "pypsa"
    display_name = "PyPSA"
    color = "#1baf7a"
    package = "pypsa"
    modules = ("pypsa", "scipy.io")
    language = "python"
    families = ("matpower",)
    settings = {"algorithm": "nr", "transformer_model": "pi", "init": "flat", "tolerance_pu": TOLERANCE_PU}

    def load(self, case):
        import pypsa
        import scipy.io
        logging.getLogger("pypsa").setLevel(logging.ERROR)
        mpc = scipy.io.loadmat(str(mat_path(case)), simplify_cells=True)["mpc"]
        n = pypsa.Network()
        n.import_from_pypower_ppc(mpc, overwrite_zero_s_nom=1e4)
        n.transformers["model"] = "pi"
        return {"network": n, "iterations": None}

    def solve(self, model):
        res = model["network"].pf(x_tol=TOLERANCE_PU, use_seed=False)
        if not bool(np.all(res.converged.values)):
            raise DidNotConverge(f"pf did not converge after {int(np.max(res.n_iter.values))} iterations")
        model["iterations"] = int(np.max(res.n_iter.values))

    def solution(self, model, case):
        n = model["network"]
        vm = n.buses_t.v_mag_pu.iloc[0]
        va = np.rad2deg(n.buses_t.v_ang.iloc[0])
        return Solution({str(k): float(v) for k, v in vm.items()}, {str(k): float(v) for k, v in va.items()},
                        model["iterations"])
