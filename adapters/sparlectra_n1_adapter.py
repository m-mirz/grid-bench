"""Sparlectra: N-1 contingency analysis as a loop of `runpf_rectangular!`
(`loop`).

Sparlectra's scenario engine runs N-1 outages, but each result is a set of
limit metrics, not the voltages of every bus, so it cannot be graded; the
loop does what its outage does (`GridBenchSparlectra.solve_outage!`): the
branch's status and both end statuses set to 0, the solve, the branch back
in. The base case is the power-flow adapter's solve from the common flat
start (`adapters/sparlectra_adapter.py`, its settings); every outage starts
from the base solution, written into the nodes (`opt_flatstart=false`). One
call into Julia per outage; Julia runs single-threaded.

Outages are solved with `power_mode=false` (the base case keeps the power
flow's `power_mode=true`): in 0.30.1, power mode replays the previous
solve's Jacobian assembly whenever the bus types and the entry count are
unchanged, and two outages of one branch each have the same count but a
different sparsity pattern, so the second outage's Jacobian is assembled into
the first one's slots. On case14 13 of 20 outages then diverge, with KLU and
UMFPACK alike, while each converges from a fresh `Net` or without power
mode. The Ybus itself is rebuilt correctly (its fingerprint covers branch
status); the assembly is not. Without power mode every outage builds its
own, as in 0.17.3.

Join: Sparlectra's importer makes `branchVec[row + 1]` of every branch row,
asserted by the MATPOWER numbers of its buses (`busOrigIdxDict`).
"""
import numpy as np

from adapters.batch_adapter import LoopContingencyAdapter, base_case, outages
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from adapters.sparlectra_adapter import SparlectraAdapter, _gb


class SparlectraN1(LoopContingencyAdapter):
    name = SparlectraAdapter.name
    display_name = SparlectraAdapter.display_name
    color = SparlectraAdapter.color
    package = SparlectraAdapter.package
    language = SparlectraAdapter.language
    modules = SparlectraAdapter.modules
    settings = SparlectraAdapter.settings | {"mode": "loop", "start": "base-case solution (opt_flatstart=false)",
                                            "power_mode": "base case only"}
    single = SparlectraAdapter()
    version = SparlectraAdapter.version
    dependencies = SparlectraAdapter.dependencies

    def load(self, case):
        model = self.single.load(base_case(case))
        out = outages(case)
        jl_out = _gb().load_outages(model, out["branch_row"].tolist(), out["from_bus"].tolist(), out["to_bus"].tolist())
        ids = [str(i) for i in _gb().solution(model)[0]]
        return {"single": model, "outages": out, "jl": jl_out, "bus_ids": ids}

    def solve_base(self, model):
        if _gb().solve_base_b(model["single"], model["jl"], TOLERANCE_PU, MAX_ITERATIONS) < 0:   # solve_base!
            raise DidNotConverge(f"base case: NR did not converge in {MAX_ITERATIONS} iterations")

    def solve_outage(self, model, k):
        if _gb().solve_outage_b(model["single"], model["jl"], k + 1, TOLERANCE_PU, MAX_ITERATIONS) < 0:
            raise DidNotConverge(f"outage {k}: NR did not converge in {MAX_ITERATIONS} iterations")

    def voltages(self, model):
        _, vm, va = _gb().solution(model["single"])
        return np.asarray(vm), np.asarray(va)
