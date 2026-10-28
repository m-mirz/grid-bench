"""gridoxide: N-1 contingency analysis through
`PowerFlowModel.solve_contingencies` (`native`), Newton-Raphson on rayon
threads, in Rust, driven through its Python bindings.

Input: the base case's `.m` through `PowerFlowModel.from_matpower`, as in the
power-flow adapter (`adapters/gridoxide_adapter.py`), with its settings
(backend, `tol=TOLERANCE_PU`, `max_iter=MAX_ITERATIONS`).

Joins, built in `load` and asserted: `branch_rows()` gives each of
gridoxide's flat branch indices (lines, then transformers) its row in the
`.m`'s branch matrix; each outage, named by its row, becomes the flat index
of that row, every row present exactly once. Buses by MATPOWER bus number
(`bus_numbers()`). Results come back in the order the outages are given,
the case's.

Settings:
- The timed `solve` is the base case from flat (`solve(init="flat")`, as the
  power-flow adapter) and then `solve_contingencies(..., start="last")`:
  every outage starts from that solution, the problem's base-case start.
- `threads`: the thread count. gridoxide hands outages to its workers one at
  a time from a shared counter. An outage that keeps the network connected
  takes it out of the Y-bus with its structural entries kept, so the
  worker's symbolic factorization carries over; none here islands the grid
  (`cases.contingency`). Results do not depend on the thread count, which
  the oracle checks per thread count.
- `solve_contingencies` builds its rayon pool and its workers' solvers on
  every call, unlike `solve_batch`, which keeps them: that cost is timed.
- A batch with an outage that did not converge fails, with the count.
"""
import numpy as np

from adapters.batch_adapter import BatchSolution, ContingencyAdapter, base_case, outages
from adapters.gridoxide_adapter import GridoxideAdapter
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.matpower import BUS_I, F_BUS, T_BUS, parse_m
from cases.registry import CASES


class GridoxideN1(ContingencyAdapter):
    name = GridoxideAdapter.name
    display_name = GridoxideAdapter.display_name
    color = GridoxideAdapter.color
    package = GridoxideAdapter.package
    language = GridoxideAdapter.language
    modules = GridoxideAdapter.modules
    backend = GridoxideAdapter.backend
    mode = "native"
    settings = GridoxideAdapter.settings | {"mode": "native", "batch_api": "PowerFlowModel.solve_contingencies",
                                            "start": "base-case solution (start=\"last\")",
                                            "threads": "the thread count"}

    def load(self, case):
        import gridoxide
        path = CASES[base_case(case)]["file"]
        model = gridoxide.PowerFlowModel.from_matpower(
            str(path), backend=self.backend, tol=TOLERANCE_PU, max_iter=MAX_ITERATIONS)
        rows = model.branch_rows()
        assert len(set(rows)) == len(rows) == model.n_branches, "one flat branch per in-service row"
        flat = {row: k for k, row in enumerate(rows)}
        out = outages(case)
        branch = parse_m(path)["branch"]
        for row, f, t in zip(out["branch_row"], out["from_bus"], out["to_bus"]):
            assert (int(branch[row, F_BUS]), int(branch[row, T_BUS])) == (int(f), int(t)), f"row {row}"
        contingencies = [[flat[int(row)]] for row in out["branch_row"]]
        numbers = model.bus_numbers()
        assert numbers == [int(b) for b in parse_m(path)["bus"][:, BUS_I]]
        return {"model": model, "contingencies": contingencies, "n": len(contingencies),
                "bus_ids": [str(b) for b in numbers], "results": None}

    def solve(self, model, threads=1):
        m = model["model"]
        try:
            m.solve(init="flat")
        except RuntimeError as e:
            raise DidNotConverge(f"base case: {e}") from e
        results = m.solve_contingencies(model["contingencies"], threads=threads, start="last")
        bad = sum(status != "converged" for status, _, _ in results)
        if bad:
            raise DidNotConverge(f"{bad} of {model['n']} outages did not converge")
        model["results"] = results

    def solution(self, model, case):
        results = model["results"]
        vm = np.array([r[1] for r in results])
        va = np.rad2deg(np.array([r[2] for r in results]))
        assert vm.shape == (model["n"], len(model["bus_ids"]))
        return BatchSolution(model["bus_ids"], vm, va)
