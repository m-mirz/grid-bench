"""pypowsybl: DC power flow (OpenLoadFlow's `run_dc`) and DC N-1 (its
security analysis in DC, `run_dc`, `native`, threaded).

Input: PowSyBl's MATPOWER importer on the `.mat`, as the power-flow adapter
(`adapters/pypowsybl_adapter.py`). Branch rows are joined to IIDM ids by the
N-1 adapter's `branch_ids` (LINE-<f>-<t> and TWT-<f>-<t>, parallel ones by
file order, asserted); side 1 of each is the row's from bus, as the
importer names it. Buses: `BUS-<n>` (bus-breaker view) in the security
analysis; in the DC load flow every bus number through the ends of the
lines and transformers, as the batch adapter reads voltages.

Settings: the power-flow adapter's load-flow parameters, which matter in DC
as `distributed_slack=False` (one slack, the case's) and
`dc_use_transformer_ratio=True` (its default; MATPOWER's DC model divides a
branch's admittance by its tap ratio). Phase shifts are OpenLoadFlow's.
- DC: `run_dc`; angles from `get_buses`, flows `p1` of every line and
  two-winding transformer (MW).
- DC N-1: one single-element contingency per outage, every voltage level and
  branch monitored, `run_dc` with `threadCount` the thread count; flows `p1`
  and angles per contingency from `branch_results` and `bus_results` (the
  outaged branch is not reported, which the oracle allows). OpenLoadFlow
  computes DC contingencies from the base case's factorization and
  sensitivities rather than solving each anew; both are exact in DC.

Finding (first run, pypowsybl 1.16.1): OpenLoadFlow's DC model leaves a
bus's shunt conductance (Gs) out, where MATPOWER counts it as load. Every
case with Gs is rejected for it alone: graded against the same case with
Gs set to 0, pypowsybl matches to 1e-9 MW (case300 17 buses with Gs, off
by 1.3 MW; case2869pegase 46, 2.5 MW; case9241pegase 292, 13 MW, and the
same in every outage of their N-1); the cases without Gs (case14, case118,
case1354pegase, case3120sp, case2848rte with its phase shifters) are exact.
A modelling choice of the tool, reported and left as it is.
"""
import numpy as np

from adapters.batch_adapter import base_case, outages
from adapters.dc_adapter import DcAdapter, DcContingencyAdapter, DcSolution
from adapters.pypowsybl_adapter import PypowsyblAdapter
from adapters.pypowsybl_n1_adapter import branch_ids
from adapters.solver_adapter import DidNotConverge
from cases.matpower import BUS_I, parse_m
from cases.registry import CASES


class PypowsyblDc(DcAdapter):
    name = PypowsyblAdapter.name
    display_name = PypowsyblAdapter.display_name
    color = PypowsyblAdapter.color
    package = PypowsyblAdapter.package
    language = PypowsyblAdapter.language
    modules = PypowsyblAdapter.modules
    settings = {"api": "run_dc (OpenLoadFlow)", "distributed_slack": False, "dc_use_transformer_ratio": True}

    def __init__(self):
        self.single = PypowsyblAdapter()

    def load(self, case):
        base = base_case(case)
        model = self.single.load(base)
        net, mpc = model["network"], parse_m(CASES[base]["file"])
        by_number = {}
        for df in (net.get_lines(), net.get_2_windings_transformers()):
            for elem_id, b1, b2 in zip(df.index, df["bus1_id"], df["bus2_id"]):
                _, f, t = elem_id.split("-")
                by_number.setdefault(int(f), b1)
                by_number.setdefault(int(t.split("#")[0]), b2)
        numbers = mpc["bus"][:, BUS_I].astype(int)
        return {"network": net, "params": model["params"], "ids": branch_ids(net, mpc["branch"]),
                "bus_ids": [str(b) for b in numbers], "bus_index": [by_number.get(int(b), "") for b in numbers]}

    def solve(self, model):
        import pypowsybl.loadflow as lf
        result = lf.run_dc(model["network"], parameters=model["params"])[0]
        if result.status != lf.ComponentStatus.CONVERGED:
            raise DidNotConverge(f"{result.status.name}: {result.status_text}")

    def solution(self, model, case):
        net = model["network"]
        lines, twts = net.get_lines(attributes=["p1"])["p1"], net.get_2_windings_transformers(attributes=["p1"])["p1"]
        p1 = np.array([(twts if i.startswith("TWT") else lines)[i] for i in model["ids"]], float)
        va = net.get_buses(attributes=["v_angle"]).reindex(model["bus_index"])["v_angle"].to_numpy()
        return DcSolution(model["bus_ids"], va, list(range(len(p1))), p1)


class PypowsyblDcN1(DcContingencyAdapter):
    name = PypowsyblAdapter.name
    display_name = PypowsyblAdapter.display_name
    color = PypowsyblAdapter.color
    package = PypowsyblAdapter.package
    language = PypowsyblAdapter.language
    modules = PypowsyblAdapter.modules + ("pypowsybl.security",)
    mode = "native"
    settings = {"api": "security analysis run_dc (OpenLoadFlow)", "distributed_slack": False,
                "dc_use_transformer_ratio": True, "method": "OpenLoadFlow's DC contingency computation",
                "threadCount": "the thread count"}

    def __init__(self):
        self.single = PypowsyblAdapter()

    def load(self, case):
        import pypowsybl.security as sa
        base = base_case(case)
        model = self.single.load(base)
        net, mpc = model["network"], parse_m(CASES[base]["file"])
        ids = branch_ids(net, mpc["branch"])
        rows = outages(case)["branch_row"]
        names = [f"outage{k}" for k in range(len(rows))]
        analysis = sa.create_analysis()
        for name, r in zip(names, rows):
            analysis.add_single_element_contingency(ids[r], name)
        analysis.add_monitored_elements(voltage_level_ids=list(net.get_voltage_levels().index), branch_ids=list(ids))
        return {"network": net, "params": model["params"], "analysis": analysis, "names": names, "ids": ids,
                "bus_ids": [str(int(b)) for b in mpc["bus"][:, BUS_I]], "result": None}

    def solve(self, model, threads=1):
        import pypowsybl.loadflow as lf
        import pypowsybl.security as sa
        params = sa.Parameters(load_flow_parameters=model["params"], provider_parameters={"threadCount": str(threads)})
        result = model["analysis"].run_dc(model["network"], parameters=params)
        if result.pre_contingency_result.status != lf.ComponentStatus.CONVERGED:
            raise DidNotConverge(f"base case: {result.pre_contingency_result.status.name}")
        post = result.post_contingency_results
        bad = [n for n in model["names"] if post[n].status.name != "CONVERGED"]
        if bad:
            raise DidNotConverge(f"{len(bad)} of {len(model['names'])} outages not solved ({post[bad[0]].status.name})")
        model["result"] = result

    def solution(self, model, case):
        names, result = model["names"], model["result"]
        br = result.branch_results.reset_index()
        br = br[br["contingency_id"].isin(names)]
        pf = br.pivot(index="contingency_id", columns="branch_id", values="p1").reindex(index=names, columns=model["ids"])
        bus = result.bus_results.reset_index()
        bus = bus[bus["contingency_id"].isin(names)].assign(number=lambda d: d["bus_id"].str.removeprefix("BUS-"))
        va = bus.pivot(index="contingency_id", columns="number", values="v_angle").reindex(index=names, columns=model["bus_ids"])
        return DcSolution(model["bus_ids"], va.to_numpy(), list(range(len(model["ids"]))), pf.to_numpy())
