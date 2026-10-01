"""lightsim2grid: N-1 contingency analysis through `ContingencyAnalysisCPP`
(`native`).

lightsim2grid's contingency analysis solves one N-1 per branch it is given,
each on its own copy of the admittance matrix, split over `nb_thread` OS
threads (contiguous ranges of the list, each thread with its own solver).
Driven without grid2op, as the power-flow adapter drives `ac_pf`.

Settings:
- `change_algorithm(NR_KLU)`, as in the power-flow adapter.
- `init_from_n_powerflow=True`: `compute` first solves the base case from
  the `v_init` it is given (the power-flow adapter's flat start) and starts
  every outage from that solution: the problem's base-case start, base solve
  included in the timed call.
- `nb_thread`: the thread count; lightsim2grid documents that results do
  not depend on it, which the oracle checks per thread count.
- `handle_disconnected_grid=False` (default): no outage here islands the
  grid (`cases.contingency`).
- A batch with an outage that did not converge (`converged_mask`) fails.
  The base case's own status (`converged_n`) is only kept when limit
  violations are computed, which they are not; an outage cannot converge
  from a base case that did not, so the mask covers it.

Joins, built in `load` and asserted: `init_from_matpower` makes a line of
every branch row with no tap ratio and no phase shift and a transformer of
every other, each in file order; contingency ids are line ids, then
`n_line +` transformer ids. Every element's buses are checked against its
row's from and to bus. Results come back in `my_defaults()` order, mapped
back to the case's outage order by id.
"""
import numpy as np

from adapters.batch_adapter import ContingencyAdapter, BatchSolution, base_case, outages
from adapters.lightsim2grid_adapter import Lightsim2gridAdapter
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.matpower import ANGLE, BUS_I, F_BUS, RATIO, T_BUS, parse_m
from cases.registry import CASES, mat_path


class Lightsim2gridN1(ContingencyAdapter):
    name = Lightsim2gridAdapter.name
    display_name = Lightsim2gridAdapter.display_name
    color = Lightsim2gridAdapter.color
    package = Lightsim2gridAdapter.package
    language = Lightsim2gridAdapter.language
    modules = Lightsim2gridAdapter.modules + ("lightsim2grid.contingencyAnalysis",)
    mode = "native"
    settings = Lightsim2gridAdapter.settings | {"mode": "native", "batch_api": "ContingencyAnalysisCPP.compute",
                                                "init_from_n_powerflow": True, "nb_thread": "the thread count"}

    def load(self, case):
        from lightsim2grid.algorithm import AlgorithmType
        from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
        from lightsim2grid.network import init_from_matpower
        base = base_case(case)
        mpc, out = parse_m(CASES[base]["file"]), outages(case)
        branch, number = mpc["branch"], mpc["bus"][:, BUS_I].astype(int)
        grid = init_from_matpower(str(mat_path(base)))
        lines, trafos = grid.get_lines(), grid.get_trafos()
        is_trafo = (branch[:, RATIO] != 0) | (branch[:, ANGLE] != 0)
        line_rows, trafo_rows = np.flatnonzero(~is_trafo), np.flatnonzero(is_trafo)
        assert len(lines) == len(line_rows) and len(trafos) == len(trafo_rows), "one element per branch row"
        for elements, rows in ((lines, line_rows), (trafos, trafo_rows)):
            for e, r in zip(elements, rows):
                assert {number[e.bus1_id], number[e.bus2_id]} == {int(branch[r, F_BUS]), int(branch[r, T_BUS])}
        cid = np.empty(len(branch), dtype=np.int64)
        cid[line_rows] = np.arange(len(line_rows))
        cid[trafo_rows] = len(line_rows) + np.arange(len(trafo_rows))
        ids = cid[out["branch_row"]]

        computer = ContingencyAnalysisCPP(grid)
        computer.change_algorithm(AlgorithmType.NR_KLU)
        computer.init_from_n_powerflow = True
        computer.add_multiple_n1([int(i) for i in ids])
        order = [d for d in computer.my_defaults()]
        assert sorted(len(d) for d in order) == [1] * len(ids), "one branch per contingency"
        row_of = {int(d[0]): j for j, d in enumerate(order)}
        v_init = np.full(len(grid.get_bus_vn_kv()), grid.get_init_vm_pu(), dtype=complex)
        return {"computer": computer, "v_init": v_init, "rows": np.array([row_of[int(i)] for i in ids]),
                "n": len(ids), "bus_ids": [str(b) for b in number], "v": None}

    def solve(self, model, threads=1):
        computer = model["computer"]
        computer.nb_thread = threads
        computer.compute(model["v_init"], MAX_ITERATIONS, TOLERANCE_PU)
        bad = model["n"] - int(np.asarray(computer.converged_mask(), bool).sum())
        if bad:
            raise DidNotConverge(f"{bad} of {model['n']} outages did not converge")
        model["v"] = computer.get_voltages()[model["rows"]]

    def solution(self, model, case):
        v = model["v"]
        return BatchSolution(model["bus_ids"], np.abs(v), np.angle(v, deg=True))
