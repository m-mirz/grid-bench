"""p3s (parallel pandapower solver): `NewtonPowerflowCpp`, its C++/KLU Newton-Raphson.

p3s (e2nIEE, Fraunhofer IEE) solves pandapower networks with solvers of its
own: it builds its own Ybus from the pandapower element tables and never
calls `runpp`. This column times the compiled single-grid solver (`nr_klu`,
KLU with a symbolic analysis cached per topology), not the batched or GPU
paths, which solve many operating points at once and have no counterpart in
a single warm solve.

Inputs: the same pandapower nets as the pandapower column (`from_mpc`,
`from_cim`), so an importer problem shows in both; what differs between the
two columns is p3s's Ybus and solver. Since 1.2.1 p3s reproduces
pandapower on every MATPOWER case and cimoxide conversion here: where the
oracle rejects p3s (case300, case9241pegase, case3120sp, case2848rte and
their conversions), it rejects pandapower with the same residual at the
same buses, the `from_mpc`/`from_cim` losses the pandapower adapter
documents (and on case2848rte@cimoxide the same low-voltage root). 1.2.1
fixed what made p3s fail before (its tests/test_grid_bench_findings.py):
`impedance` branches stamped into its Ybus, out-of-service generators no
longer PV, the CGMES slack taken from `gen.slack`, switches fused,
transformer ratings and tap tables kept.

Known losses (1.2.1), both shared with the pandapower column: on
cgmes_microgrid_be p3s stops before the first iteration ("reached 0") on a
non-finite mismatch: `from_cim` leaves the five boundary buses without
`vn_kv` (NaN), which makes the per-unit admittance of the two lines
starting at one NaN, the importer's loss (issue #20). On cgmes_realgrid it
does not converge in `MAX_ITERATIONS`, as pandapower does not.

Bus ids: p3s uses a bus's pandapower index as its Ybus row, so it needs the
index to be 0..n-1; `from_mpc` sets it to MATPOWER bus number - 1, which has
gaps on most cases (case18, case300, pegase, RTE, mvlv), and p3s then fails
with an IndexError. `load` therefore renumbers the buses with pandapower's
`create_continuous_bus_index`, a relabelling that leaves the power-flow
equations untouched, and keeps the old index in `net.bus.old_index`: the
MATPOWER number is `old_index + 1`. p3s fuses buses joined by closed
bus-bus switches into one node (its Ybus is per node, so on the CGMES
fixtures with switches it has fewer rows than `net.bus`), and `calculate`
maps the node voltages back to one voltage per `net.bus` row, NaN on
unsupplied buses: voltage row i is `net.bus` row i (asserted). An earlier
assertion here compared p3s's node lookup with the bus rows, which raised
on every fixture with switches and was recorded as p3s failing to build its
model (issue #20). CGMES buses join through `net.bus.cim_topnode`, as in
the pandapower adapter; unsupplied buses are left out.

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
    color = "#c78600"   # tools/palette.py: C++ yellow, step 1, dashed
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
        assert len(v) == len(net.bus)
        vm, va = np.abs(v), np.angle(v, deg=True)
        if not is_cgmes(case):
            ids = [str(int(i) + 1) for i in net.bus.old_index]
            return Solution(dict(zip(ids, vm)), dict(zip(ids, va)))
        v_kv = vm * net.bus.vn_kv.values
        tn = [mrid(t) if isinstance(t, str) else None for t in net.bus.cim_topnode]
        pairs = [(t, u, a) for t, u, a in zip(tn, v_kv, va) if t and np.isfinite(u)]
        return Solution({t: u for t, u, _ in pairs}, {t: a for t, _, a in pairs})
