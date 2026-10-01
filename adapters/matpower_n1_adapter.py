"""MATPOWER: N-1 contingency analysis as a loop of `runpf` (`loop`), inside
Octave.

MATPOWER has no AC contingency analysis (its N-1 tools are linear, through
PTDF and LODF matrices). The loop is the one a MATPOWER user writes
(adapters/matpower_octave/gb_n1_solve.m): `runpf` on the base case from
flat start, then for every outage the case struct with that branch's status
set to 0 and the base solution in its VM and VA columns, from which `runpf`
starts. The whole loop is timed in Octave, as the power-flow adapter's
solves are; one thread (see adapters/matpower_batch_adapter.py).

Every `runpf` is the power-flow adapter's (`adapters/matpower_adapter.py`).
Outages are rows of the case's own branch matrix, so the join is exact.
"""
import tempfile
from pathlib import Path

import numpy as np

from adapters.batch_adapter import BatchSolution, base_case, outages
from adapters.matpower_adapter import MatpowerAdapter, MatpowerModel
from adapters.solver_adapter import DidNotConverge
from cases.registry import CASES


class MatpowerN1(MatpowerAdapter):
    """The power-flow adapter's Octave session, driving the N-1 loop."""
    problem = "n1"
    families = ("n1-matpower",)
    mode = "loop"
    settings = MatpowerAdapter.settings | {"mode": "loop", "start": "base-case solution",
                                           "update": "branch status, then runpf (in Octave)"}

    def tags(self):
        return ["powerflow", "ac", "contingency", self.mode, self.language]

    def thread_counts(self):
        return (1,)

    def load(self, case):
        rows = " ".join(str(int(r) + 1) for r in outages(case)["branch_row"])
        session = self._octave()
        (line,) = session.eval(f"gb_n1_load('{CASES[base_case(case)]['file']}', [{rows}])")
        model_id, seconds = line.split()
        self._seconds += float(seconds)
        model = MatpowerModel(session, int(model_id))
        model.n = len(rows.split())
        return model

    def solve(self, model, threads=1):
        assert threads == 1, "a loop runs on one thread"
        (line,) = self._octave().eval(f"gb_n1_solve({model.id})")
        base_ok, failed, seconds = line.split()
        self._seconds += float(seconds)
        if base_ok != "1":
            raise DidNotConverge("runpf did not converge on the base case")
        if failed != "0":
            raise DidNotConverge(f"runpf did not converge on {failed} of {model.n} outages")

    def solution(self, model, case):
        import scipy.io
        path = Path(tempfile.gettempdir()) / f"gb-{case.replace('#', '_')}.solution.mat"
        self._octave().eval(f"gb_sweep_solution({model.id}, '{path}')")
        out = scipy.io.loadmat(str(path))
        return BatchSolution([str(int(b)) for b in out["bus"].ravel()], np.asarray(out["vm"]).T,
                             np.asarray(out["va"]).T)
