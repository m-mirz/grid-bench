"""MATPOWER: `runopf`, AC-OPF with MIPS (MATPOWER's own primal-dual
interior-point solver), run by GNU Octave through the power-flow adapter's
session: same process, same timing inside Octave (`clock`), same memory
(`tool_pid`); see adapters/matpower_adapter.py.

MATPOWER defines the AC-OPF formulation every other tool is held to
(`adapters/optimizer_adapter.py`), and PGLib's cases are in its format, so
it is the reference implementation here too; its result is graded like
every other tool's, against PGLib's published objective, not taken as the
answer.

Input: `loadcase` on the `.m`. Joins: buses by number (`bus` column 1);
generators by row, as `runopf` returns them in the case's order (0-based,
as in the oracle).

Settings (`mpoption`):
- `opf.ac.solver = 'MIPS'`: MATPOWER's default AC-OPF solver, in the
  MATPOWER distribution (no external solver).
- `opf.start = 2` with a flat state written into the case (`gb_opf_load`):
  every bus at 1 p.u. and 0 degrees, every generator at the middle of its
  P and Q ranges. The default (`opf.start = 0`, i.e. 1) would ignore the
  case's state and use MIPS's own interior estimate.
- `opf.flow_lim = 'S'`: apparent-power branch limits, as the case means
  `rateA` (the default).
- `opf.ignore_angle_lim = 0`: angle-difference limits enforced (the default).
- `mips.feastol`, `gradtol`, `comptol`, `costtol` = `OPF_TOLERANCE`, which is
  MATPOWER's default for all but `feastol` (default: `opf.violation`, 5e-6);
  `mips.max_it = OPF_MAX_ITERATIONS`.
- `mips.step_control = 0` (the default): no step-size control.

Result: the optimum, within the reference's rounding, on every default case
but two, where MIPS stops early (MATPOWER's "did not converge"):
case2869_pegase after 46 iterations and case300_ieee__sad after 22. The
same with MATPOWER's own start (`opf.start = 0`), so not the flat start.
With `mips.step_control = 1` case300_ieee__sad reaches the optimum in 39
iterations; case2869_pegase still stops, after 50. MATPOWER is run as
shipped, so step control stays off.
"""
from adapters.matpower_adapter import MatpowerAdapter, MatpowerModel
from adapters.optimizer_adapter import OPF_MAX_ITERATIONS, OPF_TOLERANCE
from adapters.solver_adapter import DidNotConverge, Solution
from cases.registry import CASES


class MatpowerOptimizer(MatpowerAdapter):
    """The power-flow adapter's Octave session, driving `runopf` instead of
    `runpf` (hence a subclass of it, with the OPF problem's attributes)."""
    problem = "opf"
    families = ("opf-pglib",)
    settings = {"opf.ac.solver": "MIPS", "init": "flat (opf.start=2)", "opf.flow_lim": "S",
                "opf.ignore_angle_lim": 0, "mips.tol": OPF_TOLERANCE, "mips.max_it": OPF_MAX_ITERATIONS,
                "runtime": "GNU Octave"}

    def tags(self):
        return ["opf", "ac", "interior-point", self.language]

    def _octave(self):
        fresh = self._session is None
        session = super()._octave()
        if fresh:
            session.eval(f"gb_opf_init({OPF_TOLERANCE!r}, {OPF_MAX_ITERATIONS})")
        return session

    def load(self, case):
        session = self._octave()
        (line,) = session.eval(f"gb_opf_load('{CASES[case]['file']}')")
        model_id, seconds = line.split()
        self._seconds += float(seconds)
        return MatpowerModel(session, int(model_id))

    def solve(self, model):
        (line,) = self._octave().eval(f"gb_opf_solve({model.id})")
        success, iterations, seconds = line.split()
        self._seconds += float(seconds)
        model.iterations = int(iterations)
        if success != "1":
            raise DidNotConverge(f"runopf (MIPS) did not converge in {iterations} iterations")

    def solution(self, model, case):
        vm, va, pg, qg = {}, {}, {}, {}
        for line in self._octave().eval(f"gb_opf_solution({model.id})"):
            kind, key, a, b = line.split()
            if kind == "B":
                vm[key], va[key] = float(a), float(b)
            else:
                pg[key], qg[key] = float(a), float(b)
        return Solution(vm, va, model.iterations, pg_mw=pg, qg_mvar=qg)
