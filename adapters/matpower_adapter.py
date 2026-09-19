"""MATPOWER: `runpf`, Newton-Raphson, run by GNU Octave (MATLAB needs a
licence, so it cannot run in a container or in CI). MATPOWER defines the
case format and the branch model the oracle follows (`makeYbus`, `bustypes`),
so it is the reference implementation, drawn as a neutral dashed line.

Input: `loadcase` on the original `.m` (MATPOWER's own reader; the
distribution feeders are read as the plain-data copies every tool gets).
Solutions come back keyed by MATPOWER bus number.

How it is driven: one persistent `octave-cli` process per benchmark
process, fed statements over stdin (adapters/octave_session.py), with the
Octave side in adapters/matpower_octave/. Every timed step is timed inside
Octave with tic/toc and reported through `clock`, so the ~1 ms the bridge
adds per call is not in any number; memory is that Octave process's
(`tool_pid`). MATPOWER has no persistent solver object: a solve is `runpf`
on the loaded case struct, which rebuilds Ybus and the indexing on every
call, as MATPOWER is used. Octave's `\\` is UMFPACK; OpenBLAS runs with its
default threads, like numpy's in the Python tools.

MATPOWER 8's `runpf` runs on its object-oriented MP-Core by default, which
Octave executes slowly: ~40 ms per call before any work (case14). The
legacy core (`exp.use_legacy_core = 1`) solves the same problem in the same
iterations faster (measured here, per solve: case14 5 vs 40 ms,
case9241pegase 255 vs 442 ms, mvlv29840 366 vs 686 ms). The benchmark runs
the default, MATPOWER as shipped; MATLAB runs classdef code much faster
than Octave, so these numbers are not MATLAB's.

Settings (`mpoption`):
- `pf.alg = 'NR'`: Newton-Raphson in polar form with power mismatches,
  MATPOWER's default.
- `pf.tol = TOLERANCE_PU` on the infinity norm of the power mismatch in
  p.u., the criterion of the other tools; `pf.nr.max_it = MAX_ITERATIONS`.
- `pf.enforce_q_lims = 0` (default): no reactive limits.
- Flat start on every solve: `gb_load` sets every bus to 1 p.u. and 0
  degrees in the case struct, and runpf starts from it (generator buses at
  their setpoints). runpf leaves the struct unchanged, so each solve starts
  flat again.
- Single slack, bus types by MATPOWER's own `bustypes`: the oracle's rules.
- `verbose = 0`, `out.all = 0`: no printing inside the timed call.
"""
import weakref
from pathlib import Path

from adapters.octave_session import OctaveSession
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import CASES

MATPOWER_DIR = "/opt/matpower8.1"
OCTAVE_CODE = Path(__file__).resolve().parent / "matpower_octave"


class MatpowerModel:
    """A case struct held in the Octave session, freed when dropped here."""

    def __init__(self, session: OctaveSession, model_id: int):
        self.id = model_id
        weakref.finalize(self, _free, session, model_id)


def _free(session: OctaveSession, model_id: int) -> None:
    if session.proc.poll() is None:
        session.eval(f"gb_free({model_id})")


class MatpowerAdapter(SolverAdapter):
    name = "matpower"
    display_name = "MATPOWER (Octave)"
    color = "#52514e"   # tools/palette.py REFERENCE: neutral ink, dashed
    package = "matpower"
    modules = ()
    language = "matlab"
    families = ("matpower", "distribution")
    settings = {"pf.alg": "NR", "init": "flat", "pf.enforce_q_lims": 0, "pf.tol": TOLERANCE_PU,
                "pf.nr.max_it": MAX_ITERATIONS, "runtime": "GNU Octave"}

    def __init__(self):
        self._session = None
        self._versions = None
        self._seconds = 0.0          # time measured inside Octave, summed

    def _octave(self) -> OctaveSession:
        if self._session is None:
            self._session = OctaveSession([str(OCTAVE_CODE)])
            (line,) = self._session.eval(f"gb_init('{MATPOWER_DIR}', {TOLERANCE_PU!r}, {MAX_ITERATIONS})")
            self._versions = dict(zip(("matpower", "octave"), line.split()))
        return self._session

    def clock(self) -> float:
        return self._seconds

    def tool_pid(self):
        return self._octave().pid

    def version(self):
        self._octave()
        return self._versions["matpower"]

    def dependencies(self):
        self._octave()
        return {"octave": self._versions["octave"]}

    def load(self, case):
        session = self._octave()
        (line,) = session.eval(f"gb_load('{CASES[case]['file']}')")
        model_id, seconds = line.split()
        self._seconds += float(seconds)
        return MatpowerModel(session, int(model_id))

    def solve(self, model):
        (line,) = self._octave().eval(f"gb_solve({model.id})")   # an Octave error raises OctaveError
        success, iterations, seconds = line.split()
        self._seconds += float(seconds)
        model.iterations = int(iterations)
        if success != "1":
            raise DidNotConverge(f"runpf did not converge in {iterations} iterations")

    def solution(self, model, case):
        vm, va = {}, {}
        for line in self._octave().eval(f"gb_solution({model.id})"):
            bus, m, a = line.split()
            vm[bus], va[bus] = float(m), float(a)
        return Solution(vm, va, model.iterations)
