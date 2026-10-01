"""pandapower: N-1 contingency analysis as a loop of `runpp` (`loop`).

pandapower's own contingency module (`pandapower.contingency.run_contingency`)
is this loop with result bookkeeping (minimum and maximum loadings and
voltages per element, not each outage's voltages), and it can hand the solves
to lightsim2grid, which has its own column. So the loop is written here: the
power-flow adapter's `runpp` (`adapters/pandapower_adapter.py`, its settings
and its `from_mpc` losses) on the base case from flat start, then for every
outage its element taken out of service, `runpp` started from the base
solution, the voltages copied out, the element put back. One thread.

- Start: `init="auto"` with `init_vm_pu` and `init_va_degree` set to the
  base case's `res_bus` (pandapower's documented way to start from given
  voltages; they only apply with `init="auto"`).
- Join: `net._from_ppc_lookups["branch"]`, the converter's own table from
  branch row to (element type, index): a `line`, `trafo` or `impedance`.
"""
import numpy as np

from adapters.batch_adapter import LoopContingencyAdapter, base_case, outages
from adapters.pandapower_adapter import PandapowerAdapter
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge


class PandapowerN1(LoopContingencyAdapter):
    name = PandapowerAdapter.name
    display_name = PandapowerAdapter.display_name
    color = PandapowerAdapter.color
    package = PandapowerAdapter.package
    language = PandapowerAdapter.language
    modules = PandapowerAdapter.modules
    settings = PandapowerAdapter.settings | {"mode": "loop", "start": "base-case solution (init_vm_pu, init_va_degree)"}
    single = PandapowerAdapter()

    def load(self, case):
        net = self.single.load(base_case(case))
        out = outages(case)
        lookup = net._from_ppc_lookups["branch"]   # noqa: SLF001, the converter's own join table
        assert (lookup.index == np.arange(len(lookup))).all()
        elements = [(lookup.element_type.iat[r], int(lookup.element.iat[r])) for r in out["branch_row"]]
        assert all(t in ("line", "trafo", "impedance") for t, _ in elements)
        return {"net": net, "outages": out, "elements": elements,
                "bus_ids": [str(int(i) + 1) for i in net.bus.index]}

    def solve_base(self, model):
        net = model["net"]
        self.single.solve(net)
        model["vm0"], model["va0"] = net.res_bus.vm_pu.to_numpy().copy(), net.res_bus.va_degree.to_numpy().copy()

    def solve_outage(self, model, k):
        import pandapower as pp
        net = model["net"]
        table, element = model["elements"][k]
        net[table].at[element, "in_service"] = False
        try:
            pp.runpp(net, algorithm="nr", init="auto", init_vm_pu=model["vm0"], init_va_degree=model["va0"],
                     calculate_voltage_angles=True, enforce_q_lims=False, distributed_slack=False,
                     tolerance_mva=TOLERANCE_PU, max_iteration=MAX_ITERATIONS, numba=True, lightsim2grid=False)
        except pp.LoadflowNotConverged as e:
            raise DidNotConverge(f"outage {k}: {e}") from e
        net[table].at[element, "in_service"] = True

    def voltages(self, model):
        res = model["net"].res_bus
        return res.vm_pu.to_numpy(), res.va_degree.to_numpy()
