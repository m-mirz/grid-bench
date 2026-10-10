"""Sparlectra: batch power flow as a loop of `runpf_rectangular!` (`loop`).

Sparlectra's batch machinery, the scenario engine (`src/scenario/`), runs
patch scenarios and N-1 outages for contingency analysis, each case
started from the solved base as here, but a result reports limit metrics,
not the voltages of every bus. So a sweep is a loop on the power-flow
adapter's model (`adapters/sparlectra_adapter.py`, its settings): the base
case from its flat start (`solve_sweep_base!`), then every scenario started
from the base solution written into the nodes (`solve_scenario!`,
`opt_flatstart=false`), as the N-1 adapter's outages; one thread (Julia
runs single-threaded).

Scenarios (`GridBenchSparlectra.load_sweep`, `apply!`, which with k = 0
restores the case's own values for the base solve): Sparlectra's solver
builds its injections from the prosumers on every solve, so scenario k sets
the load prosumer of each loaded bus and one generator prosumer of each
generator bus to its imported value plus the scenario's change (`Pd`,
`Qd`, and the change of the bus's online generators' `Pg` summed: at a PV
bus only the total enters the equations), in MW and MVAr, joined by MATPOWER
bus number through the importer's own `busOrigIdxDict` (one load prosumer
per loaded bus asserted). One Julia call per scenario; the changes are
passed once, in `load`.
"""
import numpy as np

from adapters.batch_adapter import LoopBatchAdapter, base_case, scenarios
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from adapters.sparlectra_adapter import SparlectraAdapter, _gb
from cases.matpower import BUS_I, PD, PG, QD, parse_m
from cases.registry import CASES


class SparlectraBatch(LoopBatchAdapter):
    name = SparlectraAdapter.name
    display_name = SparlectraAdapter.display_name
    color = SparlectraAdapter.color
    package = SparlectraAdapter.package
    language = SparlectraAdapter.language
    modules = SparlectraAdapter.modules
    settings = SparlectraAdapter.settings | {"mode": "loop", "update": "node load and generation totals",
                                             "start": "base-case solution (opt_flatstart=false)"}
    single = SparlectraAdapter()
    version = SparlectraAdapter.version
    dependencies = SparlectraAdapter.dependencies

    def load(self, case):
        base = base_case(case)
        mpc, sweep = parse_m(CASES[base]["file"]), scenarios(case)
        numbers = np.union1d(sweep["load_bus"], sweep["gen_bus"])
        pos = {int(b): i for i, b in enumerate(numbers)}
        row = {int(b): i for i, b in enumerate(mpc["bus"][:, BUS_I])}
        n = len(sweep["scale"])
        dp_load, dq_load, dp_gen = (np.zeros((len(numbers), n)) for _ in range(3))
        for b, pd, qd in zip(sweep["load_bus"], sweep["pd"].T, sweep["qd"].T):
            dp_load[pos[int(b)]] = pd - mpc["bus"][row[int(b)], PD]
            dq_load[pos[int(b)]] = qd - mpc["bus"][row[int(b)], QD]
        for b, r, pg in zip(sweep["gen_bus"], sweep["gen_row"], sweep["pg"].T):
            dp_gen[pos[int(b)]] += pg - mpc["gen"][r, PG]
        model = self.single.load(base)
        ids = [str(i) for i in _gb().solution(model)[0]]
        return {"single": model, "sweep": sweep, "bus_ids": ids,
                "jl_sweep": _gb().load_sweep(model, numbers.tolist(), dp_load, dq_load, dp_gen)}

    def solve_base(self, model):
        if _gb().solve_sweep_base_b(model["single"], model["jl_sweep"], TOLERANCE_PU, MAX_ITERATIONS) < 0:
            raise DidNotConverge(f"base case: NR did not converge in {MAX_ITERATIONS} iterations")

    def solve_scenario(self, model, k):
        # solve_scenario!, Julia counts from 1
        if _gb().solve_scenario_b(model["single"], model["jl_sweep"], k + 1, TOLERANCE_PU, MAX_ITERATIONS) < 0:
            raise DidNotConverge(f"scenario {k}: NR did not converge in {MAX_ITERATIONS} iterations")

    def voltages(self, model):
        _, vm, va = _gb().solution(model["single"])
        return np.asarray(vm), np.asarray(va)
