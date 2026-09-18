"""lightsim2grid: C++ Newton-Raphson with KLU, driven without grid2op.

Input: `lightsim2grid.network.init_from_matpower` on the prepared `.mat`.
lightsim2grid 1.0 reads MATPOWER matrices directly, without building a
pandapower net first (gridoxide's bench went through pandapower's bundled
cases). One lightsim2grid bus per MATPOWER bus, in file order. No CGMES
importer of its own (only via pypowsybl's, which would benchmark that
importer instead), so the cgmes family is not run.

Settings:
- `AlgorithmType.NR_KLU`: Newton-Raphson on KLU, lightsim2grid's fastest
  generally available AC solver.
- `ac_pf(v_init, MAX_ITERATIONS, TOLERANCE_PU)` with `v_init` all 1.0 p.u.
  and zero angle: a flat start on every solve. lightsim2grid applies PV/slack
  setpoints itself. An empty returned vector means divergence.
- No reactive limits or controls: `ac_pf` has none to switch off.
"""
import numpy as np

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.matpower import BUS_I, parse_m
from cases.registry import CASES, mat_path


class Lightsim2gridAdapter(SolverAdapter):
    name = "lightsim2grid"
    display_name = "lightsim2grid (KLU)"
    color = "#eb6834"
    package = "lightsim2grid"
    modules = ("lightsim2grid", "lightsim2grid.network", "lightsim2grid.algorithm")
    language = "c++"
    families = ("matpower",)
    settings = {"algorithm": "NR_KLU", "init": "flat", "tolerance_pu": TOLERANCE_PU, "max_iteration": MAX_ITERATIONS}

    def load(self, case):
        from lightsim2grid.algorithm import AlgorithmType
        from lightsim2grid.network import init_from_matpower
        grid = init_from_matpower(str(mat_path(case)))
        grid.change_algorithm(AlgorithmType.NR_KLU)
        n = len(grid.get_bus_vn_kv())
        return {"grid": grid, "v_init": np.full(n, grid.get_init_vm_pu(), dtype=complex), "v": None}

    def solve(self, model):
        v = model["grid"].ac_pf(model["v_init"], MAX_ITERATIONS, TOLERANCE_PU)
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
