"""pypowsybl: N-1 contingency analysis through OpenLoadFlow's security
analysis (`native`).

PowSyBl's security analysis is its contingency analysis: one AC load flow of
the base case, then every contingency started from that state, here one
single-element contingency per outage. OpenLoadFlow runs the contingencies
over `threadCount` threads. Voltages come back for every bus of every
monitored voltage level (all of them) per contingency.

Settings: the power-flow adapter's load-flow parameters
(`adapters/pypowsybl_adapter.py`: flat start of the base case, no controls,
`newtonRaphsonConvEpsPerEq=TOLERANCE_PU`, `MAX_ITERATIONS`) as the security
analysis' `load_flow_parameters`, and `threadCount` = the thread count. Its
importer's loss (transformer line charging on one end) stands.

Joins: PowSyBl's MATPOWER importer names a branch `LINE-<f>-<t>` or
`TWT-<f>-<t>`, and a further one of the same kind between the same buses
`#0`, `#1`, ... in file order (a line and a transformer in parallel are
told apart by the transformer's tap ratio or phase shift); built in
`load` and asserted (every id exists; within a parallel group, the
elements' reactance in ohm over the rows' in p.u. is one constant, so the
order is the file's). Bus results are keyed by the
bus-breaker bus `BUS-<n>`, the MATPOWER number, in kV on the voltage
level's nominal voltage. A contingency that does not converge, or a base
case that does not, fails the batch.
"""
import collections

import numpy as np

from adapters.batch_adapter import BatchSolution, ContingencyAdapter, base_case, outages
from adapters.pypowsybl_adapter import PypowsyblAdapter
from adapters.solver_adapter import DidNotConverge
from cases.matpower import ANGLE, BR_X, BUS_I, F_BUS, RATIO, T_BUS, parse_m
from cases.registry import CASES


def branch_ids(net, branch) -> list[str]:
    """The IIDM id of every branch row, as PowSyBl's MATPOWER importer names it."""
    lines, twts = net.get_lines(), net.get_2_windings_transformers()
    seen, ids = collections.Counter(), []
    for f, t, ratio, angle in zip(branch[:, F_BUS].astype(int), branch[:, T_BUS].astype(int),
                                  branch[:, RATIO], branch[:, ANGLE]):
        def candidate(kind):
            c = seen[(kind, f, t)]
            return f"{kind}-{f}-{t}" + ("" if c == 0 else f"#{c - 1}")
        line, twt = candidate("LINE"), candidate("TWT")
        has_line, has_twt = line in lines.index, twt in twts.index
        assert has_line or has_twt, f"no element for branch {f}-{t}"
        kind = ("TWT" if ratio != 0 or angle != 0 else "LINE") if has_line and has_twt else \
            ("LINE" if has_line else "TWT")
        seen[(kind, f, t)] += 1
        ids.append(line if kind == "LINE" else twt)
    x = np.array([lines.at[i, "x"] if i.startswith("LINE") else twts.at[i, "x"] for i in ids])
    groups = collections.defaultdict(list)
    for r, i in enumerate(ids):
        groups[i.split("#")[0]].append(r)
    for rows in groups.values():
        if len(rows) > 1 and (branch[rows, BR_X] != 0).all():
            ratio = x[rows] / branch[rows, BR_X]
            assert np.allclose(ratio, ratio[0], rtol=1e-6), f"parallel branches out of file order: rows {rows}"
    return ids


class PypowsyblN1(ContingencyAdapter):
    name = PypowsyblAdapter.name
    display_name = PypowsyblAdapter.display_name
    color = PypowsyblAdapter.color
    package = PypowsyblAdapter.package
    language = PypowsyblAdapter.language
    modules = PypowsyblAdapter.modules + ("pypowsybl.security",)
    mode = "native"
    settings = PypowsyblAdapter.settings | {"mode": "native", "batch_api": "security analysis (OpenLoadFlow)",
                                            "start": "base-case state", "threadCount": "the thread count"}

    def __init__(self):
        self.single = PypowsyblAdapter()

    def load(self, case):
        import pypowsybl.security as sa
        base = base_case(case)
        net = self.single.load(base)["network"]
        mpc = parse_m(CASES[base]["file"])
        ids = branch_ids(net, mpc["branch"])
        rows = outages(case)["branch_row"]
        names = [f"outage{k}" for k in range(len(rows))]
        analysis = sa.create_analysis()
        for name, r in zip(names, rows):
            analysis.add_single_element_contingency(ids[r], name)
        vls = net.get_voltage_levels()
        analysis.add_monitored_elements(voltage_level_ids=list(vls.index))
        return {"network": net, "analysis": analysis, "names": names, "nominal": vls["nominal_v"],
                "bus_ids": [str(int(b)) for b in mpc["bus"][:, BUS_I]], "result": None}

    def solve(self, model, threads=1):
        import pypowsybl.loadflow as lf
        import pypowsybl.security as sa
        params = sa.Parameters(load_flow_parameters=self.single.params, provider_parameters={"threadCount": str(threads)})
        result = model["analysis"].run_ac(model["network"], parameters=params)
        if result.pre_contingency_result.status != lf.ComponentStatus.CONVERGED:
            raise DidNotConverge(f"base case: {result.pre_contingency_result.status.name}")
        post = result.post_contingency_results
        bad = [n for n in model["names"] if post[n].status.name != "CONVERGED"]
        if bad:
            raise DidNotConverge(f"{len(bad)} of {len(model['names'])} outages did not converge "
                                 f"({post[bad[0]].status.name})")
        model["result"] = result

    def solution(self, model, case):
        res = model["result"].bus_results.reset_index()
        res = res[res["contingency_id"].isin(model["names"])]
        number = res["bus_id"].str.removeprefix("BUS-")
        vm = res["v_mag"].to_numpy() / model["nominal"].loc[res["voltage_level_id"]].to_numpy()
        table = res.assign(number=number, vm=vm)
        vm_t = table.pivot(index="contingency_id", columns="number", values="vm").reindex(
            index=model["names"], columns=model["bus_ids"])
        va_t = table.pivot(index="contingency_id", columns="number", values="v_angle").reindex(
            index=model["names"], columns=model["bus_ids"])
        return BatchSolution(model["bus_ids"], vm_t.to_numpy(), va_t.to_numpy())
