"""MATPOWER: DC power flow (`rundcpf`) and DC N-1 (a loop of `rundcpf`), in
Octave: the definition of the DC problem (`oracle/dc.py` builds MATPOWER's
`makeBdc` model), so these columns are the reference.

The power-flow adapter's Octave session and case loading
(`adapters/matpower_adapter.py`, the case struct read by `loadcase`, every
bus at 1 p.u. and 0 degrees: the reference angle is then 0, which the
oracle allows); `rundcpf` with the session's options (`verbose` 0). The
results: every bus's angle and every branch row's from-end flow (`PF`,
0 for a branch out of service), timed in Octave like the power-flow solves.

DC N-1 (adapters/matpower_octave/gb_dcn1_solve.m): `rundcpf` on the base
case, then on the case with each outage's branch status set to 0, one
thread; `rundcpf` solves anew, it uses no sensitivities.
"""
import tempfile
from pathlib import Path

import numpy as np

from adapters.batch_adapter import base_case, outages
from adapters.dc_adapter import DcSolution
from adapters.matpower_adapter import MatpowerAdapter, MatpowerModel
from adapters.solver_adapter import DidNotConverge
from cases.registry import CASES


def _solution(adapter, model, case, outage_columns: bool) -> DcSolution:
    import scipy.io
    path = Path(tempfile.gettempdir()) / f"gb-{case.replace('#', '_')}.dc.mat"
    adapter._octave().eval(f"gb_dc_solution({model.id}, '{path}')")
    out = scipy.io.loadmat(str(path))
    va, pf = np.asarray(out["va"], float), np.asarray(out["pf"], float)
    if outage_columns:
        va, pf = va.T, pf.T
    else:
        va, pf = va.ravel(), pf.ravel()
    return DcSolution([str(int(b)) for b in out["bus"].ravel()], va, list(range(pf.shape[-1])), pf)


class MatpowerDc(MatpowerAdapter):
    problem = "dc"
    families = ("dc-matpower",)
    settings = {"solver": "rundcpf", "runtime": "GNU Octave"}

    def tags(self):
        return ["powerflow", "dc", self.language]

    def load(self, case):
        return super().load(base_case(case))

    def solve(self, model):
        (line,) = self._octave().eval(f"gb_dc_solve({model.id})")
        success, seconds = line.split()
        self._seconds += float(seconds)
        if success != "1":
            raise DidNotConverge("rundcpf did not succeed")

    def solution(self, model, case):
        return _solution(self, model, case, outage_columns=False)


class MatpowerDcN1(MatpowerAdapter):
    problem = "dcn1"
    families = ("dcn1-matpower",)
    mode = "loop"
    settings = {"solver": "rundcpf per outage", "mode": "loop", "runtime": "GNU Octave"}
    too_slow = {"case9241pegase#dcn1-full": "14384 outages at about 140 ms of rundcpf each (AMD EPYC 7J13) are "
                                            "about 33 min per call; warm-up and timed round exceed the 30-minute timeout"}

    def tags(self):
        return ["powerflow", "dc", "contingency", self.mode, self.language]

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
        (line,) = self._octave().eval(f"gb_dcn1_solve({model.id})")
        base_ok, failed, seconds = line.split()
        self._seconds += float(seconds)
        if base_ok != "1":
            raise DidNotConverge("rundcpf did not succeed on the base case")
        if failed != "0":
            raise DidNotConverge(f"rundcpf did not succeed on {failed} of {model.n} outages")

    def solution(self, model, case):
        return _solution(self, model, case, outage_columns=True)
