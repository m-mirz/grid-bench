"""VeraGrid (VeraGridEngine, the headless engine; formerly GridCal).

Inputs:
- matpower: `parse_matpower_file` on the original `.m` (VeraGrid reads it
  directly, and does not need the `.mat` normalization). `Bus.code` holds the
  MATPOWER bus number. Known loss, reported by the oracle: a case whose
  `baseMVA` is not 100 is solved wrongly (every MATPOWER distribution
  feeder with baseMVA 1 or 10). case33bw rewritten at baseMVA 100 with the
  same per-unit-scaled impedances passes to 1e-7 MW, as is it misses by
  0.4 MW, so the parser does not apply the file's base power consistently.
- cgmes: `IO.file_open.open_cgmes` on the profile list. (The generic
  `open_file` rejects a list of CGMES files in 6.5.x with an empty error
  log.) VeraGrid builds one bus per ConnectivityNode; `Bus.idtag` is the
  ConnectivityNode mRID without dashes, mapped to TopologicalNodes via TP
  (for a bus-branch file, the TopologicalNode itself). VeraGrid's CGMES
  import sets `tap_phase = 0` for PhaseTapChangerTabular, i.e. drops every
  phase shift of the converted cases; the oracle reports it.

Settings, every automatic control off for comparability:
- `solver_type=NR`, `retry_with_other_methods=False`: no fallback to
  another algorithm when NR fails.
- `initialize_with_existing_solution=False`: flat start every solve.
- `distributed_slack=False`, `control_q=False`, `control_taps_modules=False`,
  `control_taps_phase=False`: outer-loop controls off.
- `control_remote_voltage=True`: generator voltage regulation as the case
  defines it, including a remote regulated terminal (see the pypowsybl
  adapter for why; no effect on MATPOWER cases).
- `tolerance=TOLERANCE_PU`, `max_iter=MAX_ITERATIONS`.
- The PowerFlowDriver is built once in `load` and re-run, so the timed
  solve includes VeraGrid's per-run compilation of its numerical circuit.
  numba JIT is paid in the untimed warm-up.
"""
import numpy as np

from adapters.cgmes_ids import by_node
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import CASES, cgmes_files, is_cgmes


class VeragridAdapter(SolverAdapter):
    name = "veragrid"
    display_name = "VeraGrid"
    color = "#008300"
    package = "VeraGridEngine"
    modules = ("VeraGridEngine", "VeraGridEngine.IO.file_open")
    language = "python"
    families = ("matpower", "distribution", "cgmes", "converted-cimoxide", "converted-pypowsybl")
    settings = {"solver_type": "NR", "retry_with_other_methods": False, "init": "flat", "distributed_slack": False,
                "outer_loop_controls": "off", "remote_voltage_control": True, "tolerance_pu": TOLERANCE_PU, "max_iteration": MAX_ITERATIONS}

    def load(self, case):
        import VeraGridEngine as vg
        if not is_cgmes(case):
            grid, _ = vg.parse_matpower_file(str(CASES[case]["file"]))
        else:
            from VeraGridEngine.IO.file_open import open_cgmes
            grid, _ = open_cgmes([str(f) for f in cgmes_files(case)])
        options = vg.PowerFlowOptions(
            solver_type=vg.SolverType.NR, retry_with_other_methods=False, initialize_with_existing_solution=False,
            distributed_slack=False, control_q=False, control_taps_modules=False, control_taps_phase=False,
            control_remote_voltage=True, tolerance=TOLERANCE_PU, max_iter=MAX_ITERATIONS, verbose=0)
        return {"grid": grid, "driver": vg.PowerFlowDriver(grid, options)}

    def solve(self, model):
        model["driver"].run()
        if not model["driver"].results.converged:
            raise DidNotConverge("NR did not converge")

    def solution(self, model, case):
        results, buses = model["driver"].results, model["grid"].get_buses()
        v = results.voltage
        iterations = int(np.max(results.iterations)) if getattr(results, "iterations", None) is not None else None
        if not is_cgmes(case):
            ids = [str(b.code) for b in buses]
            return Solution(dict(zip(ids, np.abs(v))), dict(zip(ids, np.rad2deg(np.angle(v)))), iterations)
        per_cn = {b.idtag: (abs(x) * b.Vnom, float(np.rad2deg(np.angle(x)))) for b, x in zip(buses, v)}
        vm, va = by_node(case, per_cn, "connectivity_node")
        return Solution(vm, va, iterations)
