"""pandapower: batch power flow as a loop of `runpp` (`loop`).

pandapower's own sweep tool, `run_timeseries`, is this loop with controllers
around it: a `ConstControl` per profile writes the time step's values into
the element tables, then `runpp` runs, then output writers copy results. Here
the scenario is written into the tables directly and `res_bus` copied out,
the same work without the controller and output-writer machinery, which
would time pandapower's bookkeeping rather than its solver. It runs on one
thread; pandapower has no parallel batch power flow.

Every `runpp` is the power-flow adapter's (`adapters/pandapower_adapter.py`),
with its settings and its `from_mpc` losses: the base case (the tables'
own values) with its `init="flat"`, then every scenario started from the
base solution, `init="auto"` with `init_vm_pu` and `init_va_degree` set to
the base case's `res_bus`, as the N-1 adapter does (where `run_timeseries`'
default would start each step from the last).

The scenario values of every element are built in `load` (`scenario_tables`,
also used by p3s, which reads the same nets): the per-scenario update is
then one column assignment per table.
"""
import numpy as np

from adapters.batch_adapter import LoopBatchAdapter, base_case, columns, scenarios
from adapters.pandapower_adapter import PandapowerAdapter
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge

TABLES = (("load", "p_mw"), ("load", "q_mvar"), ("sgen", "p_mw"), ("sgen", "q_mvar"), ("gen", "p_mw"))


def scenario_tables(net, sweep: dict) -> dict[tuple[str, str], np.ndarray]:
    """`{(element, column): scenarios x rows of net[element]}`, every row of
    the table, in its order: the base value where the sweep changes nothing.

    Joins, from what `from_mpc` records:
    - loads: `from_mpc` makes a `load` for each bus with `Pd > 0` (or
      `Pd == 0` and `Qd != 0`) and an `sgen` with `p_mw = -Pd`,
      `q_mvar = -Qd` for each bus with `Pd < 0`. Joined by MATPOWER bus
      number: pandapower bus index + 1, or `old_index` + 1 where the buses
      were renumbered (p3s). A load on an out-of-service (isolated) bus has
      no scenario value and keeps its own.
    - generators: `net._from_ppc_lookups["gen"]`, the converter's own table
      from gen row to (element type, index). A row that became a `gen` or an
      `sgen` gets the scenario's `Pg`; the slack's `ext_grid` has no P
      setpoint.
    """
    n = len(sweep["scale"])
    number = (net.bus.old_index if "old_index" in net.bus else net.bus.index.to_series(index=net.bus.index)) + 1
    live = net.bus.in_service
    out = {(el, col): np.repeat(net[el][col].to_numpy(float)[None, :], n, axis=0) for el, col in TABLES}

    lookup = net._from_ppc_lookups["gen"]   # noqa: SLF001, the converter's own join table
    assert (lookup.index == np.arange(len(lookup))).all()
    gen_sgens = set(lookup.element[lookup.element_type == "sgen"].astype(int))

    on = live.loc[net.load.bus].to_numpy(bool)
    col = columns(number.loc[net.load.bus[on]], sweep["load_bus"])
    out["load", "p_mw"][:, on] = sweep["pd"][:, col]
    out["load", "q_mvar"][:, on] = sweep["qd"][:, col]

    neg = np.array([i not in gen_sgens for i in net.sgen.index], dtype=bool) & live.loc[net.sgen.bus].to_numpy(bool)
    col = columns(number.loc[net.sgen.bus[neg]], sweep["load_bus"])
    out["sgen", "p_mw"][:, neg] = -sweep["pd"][:, col]
    out["sgen", "q_mvar"][:, neg] = -sweep["qd"][:, col]

    types = lookup.element_type.loc[sweep["gen_row"]].to_numpy()
    elements = lookup.element.loc[sweep["gen_row"]].to_numpy().astype(int)
    for el in ("gen", "sgen"):
        rows = net[el].index.get_indexer(elements[types == el])
        assert (rows >= 0).all()
        out[el, "p_mw"][:, rows] = sweep["pg"][:, types == el]
    return out


class PandapowerBatch(LoopBatchAdapter):
    name = PandapowerAdapter.name
    display_name = PandapowerAdapter.display_name
    color = PandapowerAdapter.color
    package = PandapowerAdapter.package
    language = PandapowerAdapter.language
    modules = PandapowerAdapter.modules
    single = PandapowerAdapter()
    settings = PandapowerAdapter.settings | {"mode": "loop", "update": "element tables, then runpp",
                                             "start": "base-case solution (init_vm_pu, init_va_degree)"}

    def load(self, case):
        net = self.single.load(base_case(case))
        sweep = scenarios(case)
        return {"single": net, "sweep": sweep, "bus_ids": [str(int(i) + 1) for i in net.bus.index],
                "tables": scenario_tables(net, sweep),
                "base": {(el, col): net[el][col].to_numpy().copy() for el, col in TABLES}}

    def solve_base(self, model):
        net = model["single"]
        for (el, col), values in model["base"].items():
            net[el][col] = values
        self.single.solve(net)
        model["vm0"], model["va0"] = net.res_bus.vm_pu.to_numpy().copy(), net.res_bus.va_degree.to_numpy().copy()

    def solve_scenario(self, model, k):
        import pandapower as pp
        net = model["single"]
        for (el, col), values in model["tables"].items():
            net[el][col] = values[k]
        try:
            pp.runpp(net, algorithm="nr", init="auto", init_vm_pu=model["vm0"], init_va_degree=model["va0"],
                     calculate_voltage_angles=True, enforce_q_lims=False, distributed_slack=False,
                     tolerance_mva=TOLERANCE_PU, max_iteration=MAX_ITERATIONS, numba=True, lightsim2grid=False)
        except pp.LoadflowNotConverged as e:
            raise DidNotConverge(f"scenario {k}: {e}") from e

    def voltages(self, model):
        res = model["single"].res_bus
        return res.vm_pu.values, res.va_degree.values

    def solution(self, model, case):
        net = model["single"]
        assert (net.res_bus.index == net.bus.index).all()
        return super().solution(model, case)
