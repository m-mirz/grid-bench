"""pandapower: `pp.runpp` Newton-Raphson.

Inputs:
- matpower: `pandapower.converter.from_mpc` on the prepared `.mat`, i.e. the
  same data every other tool reads (unlike gridoxide's bench, which used
  `pandapower.networks.<case>()`, whose bundled copies provably differ from
  the `.m` files). `from_mpc` is lossy on some cases: it re-expresses
  branches as physical-unit lines/transformers/impedances, and the oracle
  shows large residuals where that conversion changes the problem (e.g.
  case300, case3120sp). That is reported as pandapower's result, not fixed.
  pandapower creates bus index = MATPOWER bus number - 1.
- cgmes: `from_cim` (cim2pp); `net.bus.cim_topnode` is the TopologicalNode.

Settings, each the closest match to the common problem definition:
- `algorithm="nr"`: Newton-Raphson, like every other tool here.
- `init="flat"`: every solve starts flat. The default `"auto"` would start
  repeated solves from the previous result, which times a warm start.
- `calculate_voltage_angles=True`: required for phase shifters.
- `enforce_q_lims=False`, `distributed_slack=False`: no reactive limits, one slack.
- `tolerance_mva = TOLERANCE_PU`: despite the name, pandapower's newtonpf
  compares it with the per-unit mismatch (checked: `1e-8 * sn_mva` left a
  6e-6 MW residual on case14, `1e-8` leaves ~1e-13).
- Not used: `recycle`, an opt-in cache of the internal model between calls.
  Stock `runpp` rebuilds its internal model every call; that is what is timed.
- numba is installed, so pandapower uses its numba-accelerated NR.
- `lightsim2grid=False`: runpp's default `"auto"` hands the NR solve to
  lightsim2grid whenever that package is importable, so "pandapower" would
  time lightsim2grid's solver in any environment that has both (measured:
  50 ms instead of 167 ms on case9241pegase). lightsim2grid has its own
  column; this one measures pandapower's solver.
"""
import numpy as np

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import CASES, cgmes_files, mat_path
from oracle.cgmes_sv import mrid


class PandapowerAdapter(SolverAdapter):
    name = "pandapower"
    display_name = "pandapower"
    color = "#2a78d6"
    package = "pandapower"
    modules = ("pandapower", "pandapower.converter.matpower.from_mpc", "pandapower.converter.cim.cim2pp.from_cim")
    language = "python"
    families = ("matpower", "cgmes")
    settings = {"algorithm": "nr", "init": "flat", "enforce_q_lims": False, "distributed_slack": False,
                "tolerance_pu": TOLERANCE_PU, "max_iteration": MAX_ITERATIONS, "numba": True,
                "lightsim2grid_backend": False}

    def load(self, case):
        if CASES[case]["family"] == "matpower":
            from pandapower.converter.matpower.from_mpc import from_mpc
            return from_mpc(str(mat_path(case)), f_hz=50)
        from pandapower.converter.cim.cim2pp.from_cim import from_cim
        return from_cim(file_list=[str(f) for f in cgmes_files(case)], cgmes_version="3.0")

    def solve(self, net):
        import pandapower as pp
        try:
            pp.runpp(net, algorithm="nr", init="flat", calculate_voltage_angles=True, enforce_q_lims=False,
                     distributed_slack=False, tolerance_mva=TOLERANCE_PU,
                     max_iteration=MAX_ITERATIONS, numba=True, lightsim2grid=False)
        except pp.LoadflowNotConverged as e:
            raise DidNotConverge(str(e)) from e

    def solution(self, net, case):
        res = net.res_bus
        iterations = int(net._ppc["iterations"]) if net._ppc and "iterations" in net._ppc else None
        if CASES[case]["family"] == "matpower":
            ids = [str(int(i) + 1) for i in res.index]
            return Solution(dict(zip(ids, res.vm_pu)), dict(zip(ids, res.va_degree)), iterations)
        v_kv = res.vm_pu * net.bus.loc[res.index, "vn_kv"]
        tn = [mrid(t) if isinstance(t, str) else None for t in net.bus.loc[res.index, "cim_topnode"]]
        pairs = [(t, v, a) for t, v, a in zip(tn, v_kv, res.va_degree) if t and np.isfinite(v)]
        return Solution({t: v for t, v, _ in pairs}, {t: a for t, _, a in pairs}, iterations)
