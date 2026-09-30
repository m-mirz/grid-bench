"""p3s (parallel pandapower solver): `NewtonPowerflowCpp`, its C++/KLU Newton-Raphson.

p3s (e2nIEE, Fraunhofer IEE) solves pandapower networks with solvers of its
own: it builds its own Ybus from the pandapower element tables and never
calls `runpp`. This column times the compiled single-grid solver (`nr_klu`,
KLU with a symbolic analysis cached per topology), not the batched or GPU
paths, which solve many operating points at once and have no counterpart in
a single warm solve.

Inputs: the same pandapower nets as the pandapower column (`from_mpc`,
`from_cim`), so an importer problem shows in both; what differs between the
two columns is p3s's Ybus and solver. p3s stamps lines, two- and
three-winding transformers, shunts and wards only: an `impedance` (which
`from_mpc` creates for some branches), an `xward`, a switch or a `dcline` is
left out of its Ybus, and bus-bus switches (CGMES) are not fused. Checked
against pandapower's own Ybus (`net._ppc["internal"]["Ybus"]`) and the
oracle's: where a net has no `impedance`, p3s's Ybus is pandapower's (case3120sp:
equal to 4e-12, both off the `.m` on the same 350 transformer rows, the
`from_mpc` loss the pandapower adapter documents); where it has one, every
row that differs is an `impedance` bus (case300: 106 rows). So every MATPOWER
case with an `impedance` does not converge (case18, case118, case300, the
pegase and RTE cases but case1354pegase, mvlv*), or would be rejected by the
oracle if it did. p3s takes
the slack only from `ext_grid`; cim2pp writes the CGMES slack as a `gen`
with `slack=True` and no `ext_grid`, so on every CGMES input p3s has no
reference bus and does not converge (checked: `busses["ref"]` is empty on
cgmes_powerflow and case14@cimoxide).

Bus ids: p3s uses a bus's pandapower index as its Ybus row, so it needs the
index to be 0..n-1; `from_mpc` sets it to MATPOWER bus number - 1, which has
gaps on most cases (case18, case300, pegase, RTE, mvlv), and p3s then fails
with an IndexError. `load` therefore renumbers the buses with pandapower's
`create_continuous_bus_index`, a relabelling that leaves the power-flow
equations untouched, and keeps the old index in `net.bus.old_index`: the
MATPOWER number is `old_index + 1`. Ybus row i is `net.bus.index[i]`
(asserted). CGMES buses join through `net.bus.cim_topnode`, as in the
pandapower adapter.

Settings, each the closest match to the common problem definition:
- `load` builds the `NewtonPowerflow` object (Ybus, bus types, injections)
  after `calculate_trafo_characteristic`, the tap table p3s requires for
  transformers; `solve` calls `calculate` on that object, which reuses its
  KLU symbolic analysis after the first call. That persistent model is how
  p3s is meant to be used, and it is what "warm solves" means here.
- `voltage` = setpoints at PV and slack buses, 1.0 p.u. and 0 degrees
  elsewhere, passed on every call, with `init="flat"`: a flat start. p3s's
  own `init="flat"` seeds PQ buses at the mean generator setpoint instead of
  1.0 (as pandapower's `init="auto"` does), and its default `init="dc"`
  starts from a DC power flow.
- `tolerance=TOLERANCE_PU`, `max_iterations=MAX_ITERATIONS`: the infinity
  norm of the per-unit power mismatch, as everywhere else.
- `voltage_band=None`: p3s by default rejects a converged solution with any
  bus outside [0.5, 1.5] p.u. as non-physical; judging solutions is the
  oracle's job, so every converged result reaches it.
- Kept at its default: `line_search=True`, p3s's Armijo-damped Newton. It is
  the tool's Newton-Raphson; on a well-conditioned case the full step is
  accepted and it is plain Newton. `flat_fallback` only applies to
  `init="dc"`.
- Not reported: iterations. `calculate` returns the voltages only.
- The C++ extension is compiled from the release's source in the image
  (tool-configs/p3s/Dockerfile): portable (no -march=native), no OpenMP
  (only the batch solve uses threads), with the release's -O3 -ffast-math.
"""
import numpy as np

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import cgmes_files, is_cgmes, mat_path
from oracle.cgmes_sv import mrid


class P3sAdapter(SolverAdapter):
    name = "p3s"
    display_name = "pandapower (p3s)"
    color = "#2a78d7"   # tools/palette.py P3S: pandapower's blue, dashed
    package = "parallel-pandapower-solver"
    modules = ("pandapower", "pandapower.converter.matpower.from_mpc", "pandapower.converter.cim.cim2pp.from_cim",
               "pandapower.toolbox", "p3s.NewtonPowerflowCpp", "p3s.calculateTrafoTapTable")
    language = "c++"
    families = ("matpower", "distribution", "cgmes", "converted-cimoxide", "converted-pypowsybl")
    settings = {"solver": "NewtonPowerflowCpp (nr_klu)", "init": "flat", "tolerance_pu": TOLERANCE_PU,
                "max_iteration": MAX_ITERATIONS, "line_search": True, "voltage_band": None,
                "continuous_bus_index": True}

    def load(self, case):
        from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
        from p3s.NewtonPowerflowCpp import NewtonPowerflow
        from pandapower.toolbox import create_continuous_bus_index
        if not is_cgmes(case):
            from pandapower.converter.matpower.from_mpc import from_mpc
            net = from_mpc(str(mat_path(case)), f_hz=50)
        else:
            from pandapower.converter.cim.cim2pp.from_cim import from_cim
            net = from_cim(file_list=[str(f) for f in cgmes_files(case)], cgmes_version="3.0")
        create_continuous_bus_index(net, store_old_index=True)
        calculate_trafo_characteristic(net, inplace=True)
        npf = NewtonPowerflow(net)
        assert (npf._reverse_lookup.values == np.arange(len(net.bus))).all()
        assert (net.bus.index.values == np.arange(len(net.bus))).all()
        v0 = np.array(npf._initial_voltage, dtype=np.complex128)
        v0[npf.busses["pq"]] = 1.0
        return {"net": net, "npf": npf, "v0": v0, "v": None}

    def solve(self, model):
        from pandapower import LoadflowNotConverged
        try:
            model["v"] = model["npf"].calculate(model["net"], init="flat", voltage=model["v0"],
                                                tolerance=TOLERANCE_PU, max_iterations=MAX_ITERATIONS,
                                                voltage_band=None)
        except LoadflowNotConverged as e:
            raise DidNotConverge(str(e)) from e

    def solution(self, model, case):
        net, v = model["net"], model["v"]
        vm, va = np.abs(v), np.angle(v, deg=True)
        if not is_cgmes(case):
            ids = [str(int(i) + 1) for i in net.bus.old_index]
            return Solution(dict(zip(ids, vm)), dict(zip(ids, va)))
        v_kv = vm * net.bus.vn_kv.values
        tn = [mrid(t) if isinstance(t, str) else None for t in net.bus.cim_topnode]
        pairs = [(t, u, a) for t, u, a in zip(tn, v_kv, va) if t and np.isfinite(u)]
        return Solution({t: u for t, u, _ in pairs}, {t: a for t, _, a in pairs})
