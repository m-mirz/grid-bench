"""lightsim2grid: DC power flow (`dc_pf`) and DC N-1 (`ContingencyAnalysisCPP`
on its DC algorithm, `native`, one thread), driven without grid2op.

Input: the base case's `.mat` through `init_from_matpower`, as the power-flow
adapter (`adapters/lightsim2grid_adapter.py`). Branches: a line of every
branch row with no tap ratio and no phase shift, a transformer of every
other, each in file order, lines first (the N-1 adapter's join,
`contingency_ids`); each element has a side 1 and a side 2, and the row's
from-end flow is the side whose bus is the row's from bus (`sides`, asserted
for every row).

Settings:
- `change_algorithm(DC_KLU)`: the DC solver on KLU (lightsim2grid's default
  is Eigen's SparseLU), as the AC column uses NR_KLU.
- `dc_pf(v_init, MAX_ITERATIONS, TOLERANCE_PU)` with the power-flow
  adapter's flat `v_init` (a DC solve is one linear solve; the arguments are
  lightsim2grid's signature). An empty result means a failed solve.
- Results: angles from the returned voltages, flows from `get_line_res1` /
  `get_trafo_res1` and their side-2 counterparts (MW).
- DC N-1: `ContingencyAnalysisCPP` with `change_algorithm(DC_KLU)`, the
  outages given by `add_multiple_n1` (one branch each), `compute` and
  `compute_power_flows` (MW per contingency and branch, lines then
  transformers, side 1; a row whose from end is side 2 carries minus that,
  DC having no losses), angles from `get_voltages`; results mapped back to
  the case's outage order by `my_defaults`. Each outage is a new DC solve on
  its own modified Ybus (lightsim2grid's contingency analysis does not use
  LODF here).
- One thread (`nb_thread = 1`): with more, lightsim2grid 1.1.0's
  contingency analysis on a DC algorithm crashes the process (segmentation
  fault in `compute`, case14 already, at 2 threads and at 30, with DC_KLU
  and DC_SparseLU alike; its AC algorithms run threaded, see the N-1
  adapter). A crash would take every other case's record with it, so the
  thread axis is not run. Reported, not worked around.

Results (first run, AMD EPYC 7J13): the DC power flow is exact on every
case, phase shifters included (case9241pegase 0.98 ms). The DC N-1 is exact
on case14 but rejected on every PEGASE outage, on phase shifters alone
(known loss of 1.1.0's contingency analysis, reported rather than worked
around): every outage's angles match MATPOWER's model to 1e-11 degrees, but
`compute_power_flows` leaves a phase shifter's shift out of its flow
(Pfinj), so every shifter carries a wrong flow in every outage (case2869pegase:
all 12, off by up to 73 MW); and in the outage of a shifter itself the shift's
injection stays in the angles (case1354pegase 1 of 1 such outage,
case9241pegase 2 of 2). Its single `dc_pf` handles both.
"""
import numpy as np

from adapters.batch_adapter import base_case
from adapters.dc_adapter import DcAdapter, DcContingencyAdapter, DcSolution
from adapters.lightsim2grid_adapter import Lightsim2gridAdapter
from adapters.lightsim2grid_n1_adapter import contingency_ids
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.matpower import ANGLE, BUS_I, F_BUS, RATIO, T_BUS, parse_m
from cases.registry import CASES, mat_path


def sides(case: str, grid) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per branch row: its column among lightsim2grid's lines then
    transformers, whether its from end is side 1, and the case's bus numbers."""
    mpc = parse_m(CASES[case]["file"])
    branch, number = mpc["branch"], mpc["bus"][:, BUS_I].astype(int)
    lines, trafos = grid.get_lines(), grid.get_trafos()
    is_trafo = (branch[:, RATIO] != 0) | (branch[:, ANGLE] != 0)
    col = np.empty(len(branch), dtype=np.int64)
    col[~is_trafo] = np.arange((~is_trafo).sum())
    col[is_trafo] = len(lines) + np.arange(is_trafo.sum())
    elements = list(lines) + list(trafos)
    side1 = np.empty(len(branch), dtype=bool)
    for r in range(len(branch)):
        e = elements[col[r]]
        ends = (int(number[e.bus1_id]), int(number[e.bus2_id]))
        assert set(ends) == {int(branch[r, F_BUS]), int(branch[r, T_BUS])}, f"branch row {r}"
        side1[r] = ends[0] == int(branch[r, F_BUS])
    return col, side1, [str(b) for b in number]


def _from_end(p1: np.ndarray, p2: np.ndarray, col: np.ndarray, side1: np.ndarray) -> np.ndarray:
    return np.where(side1, p1[..., col], p2[..., col])


class Lightsim2gridDc(DcAdapter):
    name = Lightsim2gridAdapter.name
    display_name = Lightsim2gridAdapter.display_name
    color = Lightsim2gridAdapter.color
    package = Lightsim2gridAdapter.package
    language = Lightsim2gridAdapter.language
    modules = Lightsim2gridAdapter.modules
    settings = {"algorithm": "DC_KLU", "api": "dc_pf"}

    def load(self, case):
        from lightsim2grid.algorithm import AlgorithmType
        from lightsim2grid.network import init_from_matpower
        base = base_case(case)
        grid = init_from_matpower(str(mat_path(base)))
        grid.change_algorithm(AlgorithmType.DC_KLU)
        col, side1, ids = sides(base, grid)
        n = len(grid.get_bus_vn_kv())
        return {"grid": grid, "v_init": np.full(n, grid.get_init_vm_pu(), dtype=complex), "col": col,
                "side1": side1, "ids": ids, "v": None}

    def solve(self, model):
        v = model["grid"].dc_pf(model["v_init"], MAX_ITERATIONS, TOLERANCE_PU)
        if v.shape[0] == 0:
            raise DidNotConverge("dc_pf returned an empty voltage vector")
        model["v"] = v

    def solution(self, model, case):
        g = model["grid"]
        p1 = np.r_[g.get_line_res1()[0], g.get_trafo_res1()[0]]
        p2 = np.r_[g.get_line_res2()[0], g.get_trafo_res2()[0]]
        pf = _from_end(p1, p2, model["col"], model["side1"])
        return DcSolution(model["ids"], np.angle(model["v"], deg=True), list(range(len(pf))), pf)


class Lightsim2gridDcN1(DcContingencyAdapter):
    name = Lightsim2gridAdapter.name
    display_name = Lightsim2gridAdapter.display_name
    color = Lightsim2gridAdapter.color
    package = Lightsim2gridAdapter.package
    language = Lightsim2gridAdapter.language
    modules = Lightsim2gridAdapter.modules + ("lightsim2grid.contingencyAnalysis",)
    mode = "native"
    threaded = False
    settings = {"algorithm": "DC_KLU", "api": "ContingencyAnalysisCPP.compute, compute_power_flows",
                "method": "a new DC solve per outage", "nb_thread": 1}

    def load(self, case):
        from lightsim2grid.algorithm import AlgorithmType
        from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
        from lightsim2grid.network import init_from_matpower
        base = base_case(case)
        grid = init_from_matpower(str(mat_path(base)))
        ids, _ = contingency_ids(case, grid)
        col, side1, bus_ids = sides(base, grid)
        computer = ContingencyAnalysisCPP(grid)
        computer.change_algorithm(AlgorithmType.DC_KLU)
        computer.add_multiple_n1([int(i) for i in ids])
        order = [int(d[0]) for d in computer.my_defaults()]
        row_of = {d: j for j, d in enumerate(order)}
        n = len(grid.get_bus_vn_kv())
        return {"computer": computer, "v_init": np.full(n, grid.get_init_vm_pu(), dtype=complex),
                "rows": np.array([row_of[int(i)] for i in ids]), "col": col, "side1": side1, "bus_ids": bus_ids,
                "n": len(ids), "v": None, "p": None}

    def solve(self, model, threads=1):
        assert threads == 1, "lightsim2grid's DC contingency analysis crashes on more threads (see the docstring)"
        computer = model["computer"]
        computer.nb_thread = 1
        computer.compute(model["v_init"], MAX_ITERATIONS, TOLERANCE_PU)
        bad = model["n"] - int(np.asarray(computer.converged_mask(), bool).sum())
        if bad:
            raise DidNotConverge(f"{bad} of {model['n']} outages not solved")
        computer.compute_power_flows()
        model["v"] = computer.get_voltages()[model["rows"]]
        model["p"] = np.asarray(computer.get_power_flows())[model["rows"]]

    def solution(self, model, case):
        p = model["p"]
        # side-1 flows only; DC is lossless, so a row whose from end is side 2 carries minus that
        pf = np.where(model["side1"], p[:, model["col"]], -p[:, model["col"]])
        return DcSolution(model["bus_ids"], np.angle(model["v"], deg=True), list(range(pf.shape[1])), pf)
