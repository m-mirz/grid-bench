"""VeraGrid: N-1 contingency analysis through `ContingencyAnalysisDriver`
(`native`, one thread).

With `contingency_method=PowerFlow`, VeraGrid's driver solves the base case,
then every contingency group with the base voltages as its start
(`nonlinear_contingency_analysis`, `V_guess=base_res.voltage`): the
problem's base-case start, base solve included. It runs the groups one
after another in this process; its C++ engine (GSLV) is not installed, so
there is no thread setting: one thread.

Input and settings: the power-flow adapter's (`adapters/veragrid_adapter.py`),
its `PowerFlowOptions` as the contingency options' `pf_options`, and its
losses (a `baseMVA` other than 100; offline generators and generators on
PQ-typed buses regulating). One contingency group per outage, holding one
`Contingency` that sets its branch's `active` to 0.

Joins: `parse_matpower_file` makes one branch per branch row, lines first
and then transformers; each row is joined to the element between its from
and to bus, parallel branches in file order, and every element's reactance
is asserted to be its row's. The
driver records no convergence per contingency; a group that did not
converge shows in the oracle.
"""
import collections

import numpy as np

from adapters.batch_adapter import BatchSolution, ContingencyAdapter, base_case, outages
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU
from adapters.veragrid_adapter import VeragridAdapter
from cases.matpower import BR_X, F_BUS, T_BUS, parse_m
from cases.registry import CASES


class VeragridN1(ContingencyAdapter):
    name = VeragridAdapter.name
    display_name = VeragridAdapter.display_name
    color = VeragridAdapter.color
    package = VeragridAdapter.package
    language = VeragridAdapter.language
    modules = VeragridAdapter.modules
    mode = "native"
    threaded = False
    settings = VeragridAdapter.settings | {"mode": "native", "batch_api": "ContingencyAnalysisDriver (PowerFlow)",
                                           "start": "base-case solution", "engine": "VeraGrid (single thread)"}

    def load(self, case):
        import VeraGridEngine as vg
        base = base_case(case)
        grid, _ = vg.parse_matpower_file(str(CASES[base]["file"]))
        branch = parse_m(CASES[base]["file"])["branch"]
        by_pair = collections.defaultdict(list)
        for b in grid.get_branches():
            by_pair[(int(b.bus_from.code), int(b.bus_to.code))].append(b)
        taken = collections.Counter()
        branches = []
        for f, t in branch[:, [F_BUS, T_BUS]].astype(int):
            branches.append(by_pair[(f, t)][taken[(f, t)]])
            taken[(f, t)] += 1
        assert taken == collections.Counter({p: len(b) for p, b in by_pair.items()}), "one branch per branch row"
        assert np.allclose([b.X for b in branches], branch[:, BR_X]), "branches joined to the wrong rows"
        groups = []
        for k, r in enumerate(outages(case)["branch_row"]):
            group = vg.ContingencyGroup(name=f"outage{k}")
            grid.add_contingency_group(group)
            grid.add_contingency(vg.Contingency(device=branches[r], name=f"outage{k}",
                                                prop=vg.ContingencyOperationTypes.Active, value=0.0, group=group))
            groups.append(group)
        pf = vg.PowerFlowOptions(
            solver_type=vg.SolverType.NR, retry_with_other_methods=False, initialize_with_existing_solution=False,
            distributed_slack=False, control_q=False, control_taps_modules=False, control_taps_phase=False,
            control_remote_voltage=True, tolerance=TOLERANCE_PU, max_iter=MAX_ITERATIONS, verbose=0)
        options = vg.ContingencyAnalysisOptions(pf_options=pf, contingency_method=vg.ContingencyMethod.PowerFlow,
                                                contingency_groups=groups)
        return {"driver": vg.ContingencyAnalysisDriver(grid, options), "n": len(groups),
                "bus_ids": [str(b.code) for b in grid.get_buses()]}

    def solve(self, model, threads=1):
        assert threads == 1, "VeraGrid's own contingency engine runs on one thread"
        model["driver"].run()

    def solution(self, model, case):
        v = np.asarray(model["driver"].results.voltage, dtype=np.complex128)
        assert v.shape == (model["n"], len(model["bus_ids"]))
        return BatchSolution(model["bus_ids"], np.abs(v), np.angle(v, deg=True))
