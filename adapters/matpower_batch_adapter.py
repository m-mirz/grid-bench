"""MATPOWER: batch power flow as a loop of `runpf` (`loop`), inside Octave.

MATPOWER has no batch power flow: `runpf` solves one case struct (MOST,
its multi-period tool, is an optimal power flow). A sweep is the loop a
MATPOWER user writes: `runpf` on the base case, then for every scenario put
it into the case struct, `runpf` from the base solution, keep the voltages
(adapters/matpower_octave/gb_sweep_solve.m). The whole loop
runs in Octave and is timed there, as the power-flow adapter's solves are,
so the bridge adds nothing; one thread (Octave inherits the batch run's
`OPENBLAS_NUM_THREADS=1`, and `runpf` has nothing to parallelise over
scenarios).

Every `runpf` is the power-flow adapter's (`adapters/matpower_adapter.py`):
its options, MATPOWER as shipped (MP-Core). The base case starts flat; every
scenario starts from the base solution, written into the struct's VM and
VA columns (from which `runpf` starts), as the N-1 adapter does.

Scenarios: written in `load` to a `.mat` in the container's /tmp (the
checkout is read-only) as rows of the case's own bus and gen matrices
(1-based), then read by Octave as part of the timed load. Rows are the
case's, so the join is exact; `runpf`'s `bus(:, 1)` gives the bus numbers
of the result.
"""
import tempfile
from pathlib import Path

import numpy as np

from adapters.batch_adapter import BatchSolution, base_case, scenarios
from adapters.matpower_adapter import MatpowerAdapter, MatpowerModel
from adapters.solver_adapter import DidNotConverge
from cases.matpower import BUS_I, parse_m
from cases.registry import CASES


class MatpowerBatch(MatpowerAdapter):
    """The power-flow adapter's Octave session, driving the sweep loop
    (a subclass of it, as the OPF adapter is, with the batch problem's
    attributes)."""
    problem = "batch"
    families = ("sweep-matpower", "sweep-distribution")
    mode = "loop"
    settings = MatpowerAdapter.settings | {"mode": "loop", "update": "case struct, then runpf (in Octave)",
                                           "start": "base-case solution"}

    def tags(self):
        return ["powerflow", "ac", "batch", self.mode, self.language]

    def thread_counts(self):
        return (1,)

    def load(self, case):
        import scipy.io
        base = base_case(case)
        mpc, sweep = parse_m(CASES[base]["file"]), scenarios(case)
        row = {int(b): i for i, b in enumerate(mpc["bus"][:, BUS_I])}
        path = Path(tempfile.gettempdir()) / f"gb-{case.replace('#', '_')}.sweep.mat"
        scipy.io.savemat(str(path), {"load_rows": np.array([row[int(b)] + 1 for b in sweep["load_bus"]], float),
                                     "gen_rows": sweep["gen_row"].astype(float) + 1,
                                     "pd": sweep["pd"], "qd": sweep["qd"], "pg": sweep["pg"]})
        session = self._octave()
        (line,) = session.eval(f"gb_sweep_load('{CASES[base]['file']}', '{path}')")
        model_id, seconds = line.split()
        self._seconds += float(seconds)
        model = MatpowerModel(session, int(model_id))
        model.n = len(sweep["scale"])
        return model

    def solve(self, model, threads=1):
        assert threads == 1, "a loop runs on one thread"
        (line,) = self._octave().eval(f"gb_sweep_solve({model.id})")
        base_ok, failed, seconds = line.split()
        self._seconds += float(seconds)
        if base_ok != "1":
            raise DidNotConverge("runpf did not converge on the base case")
        if failed != "0":
            raise DidNotConverge(f"runpf did not converge on {failed} of {model.n} scenarios")

    def solution(self, model, case):
        import scipy.io
        path = Path(tempfile.gettempdir()) / f"gb-{case.replace('#', '_')}.solution.mat"
        self._octave().eval(f"gb_sweep_solution({model.id}, '{path}')")
        out = scipy.io.loadmat(str(path))
        return BatchSolution([str(int(b)) for b in out["bus"].ravel()], out["vm"].T, out["va"].T)
