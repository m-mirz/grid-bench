"""p3s: N-1 contingency analysis through its C++/KLU `solve_batch_contingency`
(`native`), parallelised with OpenMP.

p3s's N-1 path: `p3s.contingency.case_generator` turns the net's
`outage_group` column into per-outage Ybus values (the base values minus the
outaged branch's stamp, so every outage shares one sparsity pattern and one
symbolic analysis), and the solver runs every outage as an independent
Newton solve over OpenMP threads. Its driver, `solve_contingencies_cpp`,
starts every outage from one vector (a DC power flow or flat start); the
solver itself takes a start per outage, and here that is the base case's
solution, as the problem asks. So the driver's steps are done here:

- `load`: the power-flow adapter's net and model (`adapters/p3s_adapter.py`,
  with its losses: an `impedance` is not in p3s's Ybus), one `outage_group`
  per outage on its element (named so that p3s's sorted group order is the
  outage order, asserted), `ContingencyCaseGenerator(net).build()`, and the
  `nr_klu.Solver` on its pattern, as `solve_contingencies_cpp` builds them.
  Every outage must be a `line` or `trafo` (the only branches p3s stamps)
  and leave every bus served with no re-slack (none islands the grid).
- `solve` (timed): the base case through the power-flow adapter's
  `calculate` (flat start), then `solve_batch_contingency` with that solution
  as the start of every outage and nothing pinned, `n_threads` = the thread
  count, `tol=TOLERANCE_PU`, `max_iter=MAX_ITERATIONS`, `line_search=True`.
"""
import numpy as np

from adapters.batch_adapter import BatchSolution, ContingencyAdapter, base_case, outages
from adapters.p3s_adapter import P3sAdapter
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge


class P3sN1(ContingencyAdapter):
    name = P3sAdapter.name
    display_name = P3sAdapter.display_name
    color = P3sAdapter.color
    package = P3sAdapter.package
    language = P3sAdapter.language
    modules = P3sAdapter.modules + ("p3s.contingency.case_generator", "p3s.nr_klu")
    mode = "native"
    settings = P3sAdapter.settings | {"mode": "native", "batch_api": "nr_klu.Solver.solve_batch_contingency",
                                      "start": "base-case solution", "n_threads": "the thread count (OpenMP)"}
    single = P3sAdapter()

    def load(self, case):
        from p3s import nr_klu
        from p3s.contingency.case_generator import ContingencyCaseGenerator
        model = self.single.load(base_case(case))
        net, out = model["net"], outages(case)
        lookup = net._from_ppc_lookups["branch"]   # noqa: SLF001, the converter's own join table
        names = [f"outage{k:05d}" for k in range(len(out["branch_row"]))]
        for table in ("line", "trafo"):
            net[table]["outage_group"] = None
        for name, row in zip(names, out["branch_row"]):
            table = lookup.element_type.iat[row]
            assert table in ("line", "trafo"), f"branch row {row} is a p3s-unstamped {table}"
            net[table].at[int(lookup.element.iat[row]), "outage_group"] = name
        gen = ContingencyCaseGenerator(net)
        batch = gen.build()
        assert batch.groups == names and all(c.served.all() and not c.pinned_refs for c in batch.cases)
        yx = batch.Yx_matrix
        solver = nr_klu.Solver(batch.Yp, batch.Yj, np.ascontiguousarray(batch.Yx_base, dtype=np.complex128),
                               np.ascontiguousarray(batch.pv, dtype=np.int32),
                               np.ascontiguousarray(batch.pq, dtype=np.int32))
        n = batch.n_bus
        return {"single": model, "solver": solver, "n": len(names),
                "yx_mag": np.ascontiguousarray(np.abs(yx), dtype=np.float64),
                "yx_ang": np.ascontiguousarray(np.angle(yx), dtype=np.float64),
                "sbus": np.ascontiguousarray(gen._npf._sBus, dtype=np.complex128),   # noqa: SLF001, as p3s's driver
                "pin": np.zeros((n, len(names)), dtype=np.uint8),
                "bus_ids": [str(int(i) + 1) for i in net.bus.old_index], "v": None}

    def solve(self, model, threads=1):
        self.single.solve(model["single"])
        v0 = np.ascontiguousarray(np.repeat(model["single"]["v"][:, None], model["n"], axis=1), dtype=np.complex128)
        res = model["solver"].solve_batch_contingency(model["yx_mag"], model["yx_ang"], model["sbus"], v0, model["pin"],
                                                      max_iter=MAX_ITERATIONS, tol=TOLERANCE_PU, n_threads=threads,
                                                      line_search=True)
        bad = int((~np.asarray(res["converged"], bool)).sum())
        if bad:
            raise DidNotConverge(f"{bad} of {model['n']} outages did not converge in {MAX_ITERATIONS} iterations")
        model["v"] = res["V"]

    def solution(self, model, case):
        v = model["v"].T
        return BatchSolution(model["bus_ids"], np.abs(v), np.angle(v, deg=True))
