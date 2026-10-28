"""gridoxide: DC power flow (`method="dc"`) and DC N-1 from line outage
distribution factors (`outage_flows`), in Rust, driven through its Python
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
  factorization of the reduced susceptance matrix per island and one solve.
  Angles from `voltage_ang()`, from-end flows from `branch_flow_p()` (per unit
  on the case's `baseMVA`, reported in MW).
- DC N-1 (`native`, flows only): the base case's DC solve, then
  `outage_flows(branch)` for every outage, the base flows plus the outaged
  branch's LODF column times its base flow; the factorization behind the
  LODF columns is built on first use in each timed call (`reset()` before
  it, so no round reuses another's) and shared by every outage. All of it is
  the timed call. gridoxide computes no angles this way, so the oracle grades
  the flows only (`oracle.dc`'s flows-only verdict), as for
  `lightsim2grid_lodf` and ToOp. One thread: `outage_flows` takes one
  branch at a time and has no thread setting.
- Each outage's flows go straight into one numpy array (outages x branches),
  in the timed call: as Python lists, case9241pegase#dcn1-full's 14384
  outages x 16049 branches would take several gigabytes.
- An outage `outage_flows` declares radial (no LODF exists) fails the case;
  none of these outages islands the grid (`cases.contingency`).

Results (first run, Docker, AMD Ryzen 7 250): exact on every DC and DC N-1
case, full ones included, to 1e-9 MW, phase shifters and shunt conductance
included. The DC power flow is 15 to 25x lightsim2grid's time
(case9241pegase 10.3 against 0.43 ms): gridoxide partitions the network and
factors the susceptance matrix on every solve. DC N-1 on 200 outages is
within 25% of lightsim2grid's solve per outage (which fails the oracle on
every PEGASE case, see its adapter) and faster on case9241pegase; on the
full cases lightsim2grid's whole-matrix LODF is 2.3 to 3x faster.
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
    threaded = False
    settings = {"method": "LODF (flows only)", "api": "solve, outage_flows", "reader": "from_matpower (.m)",
                "threads": "1 (outage_flows has no thread setting)", "commit": GridoxideAdapter.settings["commit"]}

    def load(self, case):
        model, _, rows, base, mpc = _load(case)
        flat = {row: k for k, row in enumerate(rows)}
        out = outages(case)
        branch = mpc["branch"]
        for row, f, t in zip(out["branch_row"], out["from_bus"], out["to_bus"]):
            assert (int(branch[row, F_BUS]), int(branch[row, T_BUS])) == (int(f), int(t)), f"row {row}"
        return {"model": model, "rows": rows, "base": base,
                "outages": [flat[int(row)] for row in out["branch_row"]], "pf": None}

    def solve(self, model, threads=1):
        assert threads == 1, "outage_flows takes one branch at a time"
        m = model["model"]
        m.reset()
        try:
            m.solve()
        except RuntimeError as e:
            raise DidNotConverge(f"base case: {e}") from e
        pf = np.empty((len(model["outages"]), len(model["rows"])))
        for k, b in enumerate(model["outages"]):
            flows = m.outage_flows(b)
            if flows is None:
                raise DidNotConverge(f"outage {k} (branch {b}) is radial: no LODF")
            pf[k] = flows
        model["pf"] = pf

    def solution(self, model, case):
        pf = model["pf"] * model["base"]
        return DcSolution(None, None, model["rows"], pf)
