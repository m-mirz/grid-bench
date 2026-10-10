"""Sienna: batch power flow as PowerFlows.jl's multi-period solve (`native`,
one thread).

PowerFlows solves several operating points of one system as the time steps
of one `PowerFlowData` (`ACPowerFlow(...; time_steps=n)`): one Ybus and one
bus mapping, every time step a Newton-Raphson of its own, one after another
(`solve_power_flow!` loops over them; Julia runs single-threaded here, see
the power-flow adapter). Time-varying callers are meant to overwrite the
columns PowerFlows seeds from the system, which is what `load_batch` does.

Input and settings: the power-flow adapter's (`adapters/sienna_adapter.py`):
PowerSystems' MATPOWER parser on the `.m`, `NewtonRaphsonACPowerFlow`,
`correct_bustypes=true`, `enhanced_flat_start=false`, `TOLERANCE_PU`,
`MAX_ITERATIONS`. Start: the timed call solves the base case with the
power-flow adapter's model from its flat start, then writes that solution
into every time step and solves them from there (the same bus lookup,
asserted), as the N-1 adapter does. Its losses stand (single-precision Ybus, transformer line charging on
one end, phase shifters that do not parse).

Scenarios: PowerFlows keeps injections and withdrawals per bus and time
step, so each scenario's change against the case is added per bus, in p.u.
on the case's base power: the change of every online generator's `Pg`
summed per bus to the injections, the change of `Pd` and `Qd` to the
withdrawals. Rows are joined by MATPOWER bus number through PowerFlows' own
bus lookup (`GridBenchSienna.load_batch`). A time step that does not
converge fails the batch, with the count.

Result: the single-precision Ybus that leaves the single solve of mvlv1004
at 8.8e-4 MVA, just inside the oracle's 1e-3, takes most of that sweep's
scenarios just over it (1.3e-3 MVA); on case14 and case33bw the residual is
the single solve's.
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case, scenarios
from adapters.sienna_adapter import SiennaAdapter, _gb
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.matpower import BUS_I, PD, PG, QD, parse_m
from cases.registry import CASES


class SiennaBatch(BatchAdapter):
    name = SiennaAdapter.name
    display_name = SiennaAdapter.display_name
    color = SiennaAdapter.color
    package = SiennaAdapter.package
    language = SiennaAdapter.language
    modules = SiennaAdapter.modules
    mode = "native"
    threaded = False
    settings = SiennaAdapter.settings | {"mode": "native", "batch_api": "PowerFlowData(time_steps=n)",
                                         "start": "base-case solution"}
    version = SiennaAdapter.version
    dependencies = SiennaAdapter.dependencies

    def load(self, case):
        base = base_case(case)
        mpc, sweep = parse_m(CASES[base]["file"]), scenarios(case)
        base_mva = float(mpc["baseMVA"])
        numbers = np.union1d(sweep["load_bus"], sweep["gen_bus"])
        pos = {int(b): i for i, b in enumerate(numbers)}
        row = {int(b): i for i, b in enumerate(mpc["bus"][:, BUS_I])}
        n = len(sweep["scale"])

        dp_inj = np.zeros((len(numbers), n))
        for b, r, pg in zip(sweep["gen_bus"], sweep["gen_row"], sweep["pg"].T):
            dp_inj[pos[int(b)]] += (pg - mpc["gen"][r, PG]) / base_mva
        dp_wd, dq_wd = np.zeros((len(numbers), n)), np.zeros((len(numbers), n))
        for b, pd, qd in zip(sweep["load_bus"], sweep["pd"].T, sweep["qd"].T):
            dp_wd[pos[int(b)]] = (pd - mpc["bus"][row[int(b)], PD]) / base_mva
            dq_wd[pos[int(b)]] = (qd - mpc["bus"][row[int(b)], QD]) / base_mva
        return {"batch": _gb().load_batch(str(CASES[base]["file"]), numbers.tolist(), dp_inj, dp_wd, dq_wd), "n": n}

    def solve(self, model, threads=1):
        assert threads == 1, "PowerFlows solves time steps one after another; Julia runs single-threaded"
        bad = int(_gb().solve_batch_b(model["batch"], TOLERANCE_PU, MAX_ITERATIONS))   # solve_batch!
        if bad < 0:
            raise DidNotConverge(f"base case: NR did not converge in {MAX_ITERATIONS} iterations")
        if bad:
            raise DidNotConverge(f"{bad} of {model['n']} time steps did not converge")

    def solution(self, model, case):
        numbers, vm, va = _gb().batch_solution(model["batch"])
        return BatchSolution([str(int(b)) for b in numbers], np.asarray(vm).T, np.asarray(va).T)
