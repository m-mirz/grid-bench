"""lightsim2grid: C++ Newton-Raphson with KLU, driven without grid2op.

Input: `lightsim2grid.network.init_from_matpower` on the prepared `.mat`.
lightsim2grid 1.0 reads MATPOWER matrices directly, without building a
pandapower net first (gridoxide's bench went through pandapower's bundled
cases). One lightsim2grid bus per MATPOWER bus, in file order. Known loss,
reported by the oracle: online generators on PQ-typed buses regulate
voltage, where MATPOWER treats them as fixed P/Q injections (case2848rte:
exactly its 48 such buses). No CGMES
importer of its own (only via pypowsybl's, which would benchmark that
importer instead), so the cgmes family is not run.

Settings:
- `AlgorithmType.NRSing_KLU`: Newton-Raphson on KLU with a single slack,
  the common problem. The default `NR_KLU` is the distributed-slack Newton,
  which carries the slack's share as an extra unknown even with one slack
  bus; not used, as it solves a larger system than the problem needs.
- `ac_pf(v_init, MAX_ITERATIONS, tolerance(grid))` with `v_init` all 1.0 p.u.
  and zero angle: a flat start on every solve. lightsim2grid applies PV/slack
  setpoints itself. An empty returned vector means divergence.
- `tolerance(grid)` = `TOLERANCE_PU * sn_mva`: `ac_pf` compares the infinity
  norm of its mismatch with `tol` in MVA, not per unit (measured: the
  smallest `tol` at which Newton stops after k iterations equals the
  oracle's MVA mismatch at that iterate, on case14 and case300). Passing
  `TOLERANCE_PU` itself asked for 1e-10 p.u. at 100 MVA, 100 times the
  common tolerance, and one extra iteration on case9241pegase, mvlv10616
  and case2848rte.
- No reactive limits or controls: `ac_pf` has none to switch off.
- Built from its sdist with p3s's flags (tool-configs/lightsim2grid/Dockerfile),
  not the PyPI wheel: the same compiler and optimisation as the other
  C++/KLU Newton solver here, so the two columns differ by their code only.
"""
import numpy as np

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.matpower import BUS_I, parse_m
from cases.registry import CASES, mat_path


def tolerance(grid) -> float:
    """`TOLERANCE_PU` in the unit lightsim2grid's `tol` is in (MVA; see the
    module docstring). Shared with the batch and N-1 adapters."""
    return TOLERANCE_PU * grid.get_sn_mva()


class Lightsim2gridAdapter(SolverAdapter):
    name = "lightsim2grid"
    display_name = "lightsim2grid (KLU)"
    color = "#eda100"   # tools/palette.py: C++ yellow, base
    package = "lightsim2grid"
    modules = ("lightsim2grid", "lightsim2grid.network", "lightsim2grid.algorithm")
    language = "c++"
    families = ("matpower", "distribution")
    settings = {"algorithm": "NRSing_KLU", "init": "flat", "tolerance_pu": TOLERANCE_PU, "tolerance_unit": "MVA (tol = tolerance_pu * sn_mva)",
                "max_iteration": MAX_ITERATIONS, "build": "sdist, gcc 12, -O3 -funroll-loops -ffast-math, no -march=native"}

    def load(self, case):
        from lightsim2grid.algorithm import AlgorithmType
        from lightsim2grid.network import init_from_matpower
        grid = init_from_matpower(str(mat_path(case)))
        grid.change_algorithm(AlgorithmType.NRSing_KLU)
        n = len(grid.get_bus_vn_kv())
        return {"grid": grid, "v_init": np.full(n, grid.get_init_vm_pu(), dtype=complex), "tol": tolerance(grid), "v": None}

    def solve(self, model):
        v = model["grid"].ac_pf(model["v_init"], MAX_ITERATIONS, model["tol"])
        if v.shape[0] == 0:
            raise DidNotConverge("ac_pf returned an empty voltage vector")
        model["v"] = v

    def solution(self, model, case):
        ids = [str(int(b)) for b in parse_m(CASES[case]["file"])["bus"][:, BUS_I]]
        v = model["v"]
        if len(ids) != len(v):
            raise RuntimeError(f"lightsim2grid has {len(v)} buses, the case {len(ids)}")
        try:
            iterations = int(model["grid"].get_algo().get_nb_iter())
        except AttributeError:
            iterations = None
        return Solution(dict(zip(ids, np.abs(v))), dict(zip(ids, np.rad2deg(np.angle(v)))), iterations)
