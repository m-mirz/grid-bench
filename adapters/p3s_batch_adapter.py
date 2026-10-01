"""p3s (parallel pandapower solver): batch power flow through its C++/KLU
`solve_batch` (`native`), parallelised with OpenMP.

This is what p3s is named for: one KLU symbolic analysis of the topology,
then every operating point an independent Newton solve, split over OpenMP
threads. The power-flow adapter (`adapters/p3s_adapter.py`) times one
`calculate`; this one times one `solve_batch` over the whole sweep.

Input and model: the power-flow adapter's `load` (the same `from_mpc` net,
renumbered with `create_continuous_bus_index`, and its `NewtonPowerflow`
object), with its documented losses: an `impedance` element is left out of
p3s's Ybus, so every sweep whose case has one does not converge.

Scenarios: the element tables of `pandapower_batch_adapter.scenario_tables`
(the same joins on the same nets, through `old_index` since the buses are
renumbered), turned into the per-bus injection matrix by p3s's own
`p3s.timeseries.build_sbus_matrix`, in `load`: it is the scenario data in
the form the solver takes, as PGM's update dataset is.

Settings:
- `solve_batch(Sbus, V0, ...)` on the `nr_klu.Solver` that p3s's own
  `calculate_timeseries_cpp` builds (same Ybus arrays, PV and PQ sets), not
  that wrapper: it starts every step from a DC power flow
  (`dc_initial_voltage`). `V0` here is the power-flow adapter's flat start
  (setpoints at PV and slack buses, 1.0 p.u. and 0 degrees elsewhere), the
  start of every scenario.
- `n_threads`: the thread count (p3s: 1 is serial; its default 0, every
  core, is never passed, so the count is always the one recorded). The
  image builds the extension with OpenMP (tool-configs/p3s/Dockerfile).
- `tol=TOLERANCE_PU`, `max_iter=MAX_ITERATIONS`, `line_search=True`, as in
  the power-flow adapter. No voltage band: `solve_batch` has none.
- A scenario that does not converge fails the batch, with the count.
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case, scenarios
from adapters.p3s_adapter import P3sAdapter
from adapters.pandapower_batch_adapter import scenario_tables
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge


class P3sBatch(BatchAdapter):
    name = P3sAdapter.name
    display_name = P3sAdapter.display_name
    color = P3sAdapter.color
    package = P3sAdapter.package
    language = P3sAdapter.language
    modules = P3sAdapter.modules + ("p3s.timeseries", "p3s.nr_klu")
    mode = "native"
    settings = P3sAdapter.settings | {"mode": "native", "batch_api": "nr_klu.Solver.solve_batch",
                                      "init": "flat (V0 per scenario)", "n_threads": "the thread count (OpenMP)"}
    single = P3sAdapter()

    def load(self, case):
        from p3s import nr_klu
        from p3s.timeseries import build_sbus_matrix
        model = self.single.load(base_case(case))
        net, npf = model["net"], model["npf"]
        tables = scenario_tables(net, scenarios(case))
        sbus = build_sbus_matrix(npf, net, {key: values.T for key, values in tables.items()})
        y = npf._YBus   # noqa: SLF001, the arrays p3s's own calculate_timeseries_cpp hands the solver
        solver = nr_klu.Solver(np.ascontiguousarray(y.indptr, dtype=np.int32),
                               np.ascontiguousarray(y.indices, dtype=np.int32),
                               np.ascontiguousarray(y.data, dtype=np.complex128),
                               np.ascontiguousarray(npf.busses["pv"], dtype=np.int32),
                               np.ascontiguousarray(npf.busses["pq"], dtype=np.int32))
        return {"solver": solver, "sbus": np.ascontiguousarray(sbus, dtype=np.complex128),
                "v0": np.ascontiguousarray(model["v0"], dtype=np.complex128),
                "bus_ids": [str(int(i) + 1) for i in net.bus.old_index], "v": None}

    def solve(self, model, threads=1):
        out = model["solver"].solve_batch(model["sbus"], model["v0"], max_iter=MAX_ITERATIONS, tol=TOLERANCE_PU,
                                          n_threads=threads, line_search=True)
        bad = int((~np.asarray(out["converged"], bool)).sum())
        if bad:
            raise DidNotConverge(f"{bad} of {model['sbus'].shape[1]} scenarios did not converge "
                                 f"in {MAX_ITERATIONS} iterations")
        model["v"] = out["V"]

    def solution(self, model, case):
        v = model["v"].T
        return BatchSolution(model["bus_ids"], np.abs(v), np.angle(v, deg=True))
