"""gridoxide: DC power flow (`method="dc"`) and DC N-1 from line outage
distribution factors (`dc_contingencies`), in Rust, driven through its Python
bindings.

Input: the base case's `.m` through `PowerFlowModel.from_matpower`, as in the
power-flow adapter (`adapters/gridoxide_adapter.py`). For DC, gridoxide's
MATPOWER reader solves MATPOWER's DC model: b = 1 / (x * tap), resistance and
charging left out, phase shifts as injections, shunt conductance as load.
Joins: buses by MATPOWER bus number (`bus_numbers()`), branches by their row
in the `.m`'s branch matrix (`branch_rows()`, gridoxide's flat branch order:
lines, then transformers), both asserted against the case.

Settings:
- DC power flow: `solve()` on a model built with `method="dc"`: one
  factorization of the reduced susceptance matrix per island, built on the
  first solve (the warm-up round) and kept by the model, as its AC
  factorization is, then one right-hand side and two triangular solves.
  Angles from `voltage_ang()`, from-end flows from `branch_flow_p()` (per unit
  on the case's `baseMVA`, reported in MW).
- DC N-1 (`native`, angles and flows, threaded): the base case's DC solve,
  then `dc_contingencies(outages, threads)`, every outage in one call: per
  outage one triangular solve for the angle response to a unit transfer
  across the branch, and from it the angles and the flows after the outage
  (the base plus the transfer the removed branch would carry, which is the
  LODF column times its base flow for the flows), the outages in parallel on
  `threads` workers. The factorization behind them is built on first use in
  each timed call (`reset()` before it, so no round reuses another's) and
  shared by every outage. All of it is the timed call, the readout included:
  the two arrays come back as read-only float64 buffers that numpy views
  without a copy (case9241pegase#dcn1-full's are 2.9 GB). The oracle grades
  angles and flows.
- An outage `dc_contingencies` reports unsolved (a radial branch: no
  post-outage state) fails the case; none of these outages islands the grid
  (`cases.contingency`).

Results (first run, Docker, AMD Ryzen 7 250): exact on every DC and DC N-1
case, full ones included, to 1e-9 MW, phase shifters and shunt conductance
included. The DC power flow, with gridoxide#28 keeping its
factorization between solves (`DcSolver`), is 1.2 to 1.6x faster than
lightsim2grid's `dc_pf` (case9241pegase 0.34 against 0.44 ms); before it,
building the factorization on every solve made it 15 to 25x slower. The
DC N-1 adapter calls `reset()` before every round, so it still builds the
factorization in the timed call.
"""
import numpy as np

from adapters.batch_adapter import base_case, outages
from adapters.dc_adapter import DcAdapter, DcContingencyAdapter, DcSolution
from adapters.gridoxide_adapter import GridoxideAdapter
from adapters.solver_adapter import DidNotConverge
from cases.matpower import BUS_I, F_BUS, T_BUS, parse_m
from cases.registry import CASES


def _load(case: str):
    """The DC model and its joins, asserted against the case."""
    import gridoxide
    path = CASES[base_case(case)]["file"]
    mpc = parse_m(path)
    model = gridoxide.PowerFlowModel.from_matpower(str(path), method="dc")
    numbers = model.bus_numbers()
    assert numbers == [int(b) for b in mpc["bus"][:, BUS_I]], "one bus per row of mpc.bus, in order"
    rows = model.branch_rows()
    assert len(set(rows)) == len(rows) == model.n_branches, "one flat branch per in-service row"
    return model, [str(b) for b in numbers], rows, float(mpc["baseMVA"]), mpc


class GridoxideDc(DcAdapter):
    name = GridoxideAdapter.name
    display_name = GridoxideAdapter.display_name
    color = GridoxideAdapter.color
    package = GridoxideAdapter.package
    language = GridoxideAdapter.language
    modules = GridoxideAdapter.modules
    settings = {"method": "dc", "reader": "from_matpower (.m)", "shunt_conductance": "as load (MATPOWER)",
                "commit": GridoxideAdapter.settings["commit"]}

    def load(self, case):
        model, bus_ids, rows, base, _ = _load(case)
        return {"model": model, "bus_ids": bus_ids, "rows": rows, "base": base}

    def solve(self, model):
        try:
            model["model"].solve()
        except RuntimeError as e:
            raise DidNotConverge(str(e)) from e

    def solution(self, model, case):
        m = model["model"]
        va = np.rad2deg(np.asarray(m.voltage_ang()))
        pf = np.asarray(m.branch_flow_p()) * model["base"]
        return DcSolution(model["bus_ids"], va, model["rows"], pf)


class GridoxideDcN1(DcContingencyAdapter):
    name = GridoxideAdapter.name
    display_name = GridoxideAdapter.display_name
    color = GridoxideAdapter.color
    package = GridoxideAdapter.package
    language = GridoxideAdapter.language
    modules = GridoxideAdapter.modules
    mode = "native"
    threaded = True
    settings = {"method": "LODF, angles and flows", "api": "solve, dc_contingencies", "reader": "from_matpower (.m)",
                "commit": GridoxideAdapter.settings["commit"]}

    def load(self, case):
        model, bus_ids, rows, base, mpc = _load(case)
        flat = {row: k for k, row in enumerate(rows)}
        out = outages(case)
        branch = mpc["branch"]
        for row, f, t in zip(out["branch_row"], out["from_bus"], out["to_bus"]):
            assert (int(branch[row, F_BUS]), int(branch[row, T_BUS])) == (int(f), int(t)), f"row {row}"
        return {"model": model, "bus_ids": bus_ids, "rows": rows, "base": base,
                "outages": [flat[int(row)] for row in out["branch_row"]], "va": None, "pf": None}

    def solve(self, model, threads=1):
        m = model["model"]
        m.reset()
        try:
            m.solve()
        except RuntimeError as e:
            raise DidNotConverge(f"base case: {e}") from e
        angles, flows, solved = m.dc_contingencies(model["outages"], threads=threads)
        if not all(solved):
            k = solved.index(False)
            raise DidNotConverge(f"outage {k} (branch {model['outages'][k]}) is radial: no post-outage state")
        n = len(model["outages"])
        model["va"] = np.frombuffer(angles).reshape(n, len(model["bus_ids"]))
        model["pf"] = np.frombuffer(flows).reshape(n, len(model["rows"]))

    def solution(self, model, case):
        return DcSolution(model["bus_ids"], np.rad2deg(model["va"]), model["rows"], model["pf"] * model["base"])
