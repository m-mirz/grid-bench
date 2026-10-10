"""VeraGrid: batch power flow through its time-series driver (`native`, one
thread).

`PowerFlowTimeSeriesDriver` solves every time step of the grid's profiles
with the snapshot power flow's options. With its own engine it runs the
steps one after another in this process (`run_single_thread`); its C++
engine (GSLV) is a separate, optional package and not installed, so there
is no thread setting to time: one thread.

Input: the base case's `.m` through `parse_matpower_file`, as in the
power-flow adapter (`adapters/veragrid_adapter.py`), with its documented
losses (a `baseMVA` other than 100 is applied inconsistently, so every
distribution sweep is solved wrongly; offline generators and generators on
PQ-typed buses regulate voltage).

Scenarios: the grid's profiles, set in `load` (`create_profiles` with one
step per scenario, then each element's `P_prof`, `Q_prof`): input data, as
the profiles of a real time series are. `parse_matpower_file` makes one
load per bus with a nonzero `Pd` or `Qd`, its `bus.code` the MATPOWER
number, and one generator per gen row, in order (asserted against the
case's gen buses). Offline generators keep their own P.

Settings: the power-flow adapter's `PowerFlowOptions` (Newton-Raphson, no
fallback, every outer-loop control off, `TOLERANCE_PU`, `MAX_ITERATIONS`).
Start: the timed call solves the base case (the grid's snapshot, the case's
own values) with `PowerFlowDriver` from the flat start, writes its voltages
into every bus's stored guess (`Vm0`, `Va0` and their profiles, angles in
radians) and runs the time series with `use_stored_guess=True`, from which
every step's initial voltages are compiled (`Bus.get_voltage_guess_at`):
each starts from the base solution, never from the previous step (the
Python engine passes no other guess between steps). A step that does not
converge (`results.converged_values`) fails the batch, with the count.

Known loss of that start, reported rather than worked around: on
mvlv1004#sweep and mvlv10616#sweep 51 and 33 of the 100 scenarios do not
converge in `MAX_ITERATIONS`, the lightest ones (60 % of the base load,
|V| no lower than 0.89 p.u.) started from a base case loaded down to 0.67
p.u. From the flat start every one converges, and from this start every
one does with 200 iterations: VeraGrid's Newton-Raphson crosses that
distance slowly, where the other tools converge within the limit from the
same start. Every step's initial voltages were checked to be the base
solution (to 2e-16).

Result: every step is solved exactly (checked: the worst scenario of
case1354pegase#sweep, solved by the driver's own `multi_island_pf` and kept
in double precision, has a residual of 6e-10 MVA), but `set_at` copies each
step's voltage into a `complex64` matrix (`PowerFlowTimeSeriesResults.
voltage`), so what the driver returns is rounded to single precision, about
6e-8 p.u. On a meshed grid that rounding alone is a residual of 0.1 MVA
(case1354pegase), over the oracle's 1e-3: the driver's result, as returned,
does not solve the case, where the same steps in the power-flow driver do.
That is reported, not worked around (the adapter could re-solve each step
itself, but then it would not be timing VeraGrid's time-series driver).
The voltages are widened to complex128 before `abs` and `angle`, which
numpy would otherwise compute in float32 and so round a second time.
"""
import datetime

import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case, columns, scenarios
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from adapters.veragrid_adapter import VeragridAdapter
from cases.matpower import GEN_BUS, parse_m
from cases.registry import CASES


class VeragridBatch(BatchAdapter):
    name = VeragridAdapter.name
    display_name = VeragridAdapter.display_name
    color = VeragridAdapter.color
    package = VeragridAdapter.package
    language = VeragridAdapter.language
    modules = VeragridAdapter.modules
    mode = "native"
    threaded = False
    settings = VeragridAdapter.settings | {"mode": "native", "batch_api": "PowerFlowTimeSeriesDriver",
                                           "engine": "VeraGrid (single thread)",
                                           "start": "base-case solution (use_stored_guess)"}

    def load(self, case):
        import VeraGridEngine as vg
        base = base_case(case)
        sweep = scenarios(case)
        n = len(sweep["scale"])
        grid, _ = vg.parse_matpower_file(str(CASES[base]["file"]))
        grid.create_profiles(steps=n, step_length=1, step_unit="h", time_base=datetime.datetime(2026, 1, 1))

        load_bus = [int(ld.bus.code) for ld in grid.loads]
        assert len(set(load_bus)) == len(load_bus) == len(sweep["load_bus"]), "one load per loaded bus"
        col = columns(load_bus, sweep["load_bus"])
        for ld, c in zip(grid.loads, col):
            ld.P_prof.set(sweep["pd"][:, c])
            ld.Q_prof.set(sweep["qd"][:, c])

        gen = parse_m(CASES[base]["file"])["gen"]
        assert [int(g.bus.code) for g in grid.generators] == gen[:, GEN_BUS].astype(int).tolist(), \
            "one generator per gen row"
        for g, pg in zip([grid.generators[r] for r in sweep["gen_row"]], sweep["pg"].T):
            g.P_prof.set(pg)

        settings = dict(
            solver_type=vg.SolverType.NR, retry_with_other_methods=False, initialize_with_existing_solution=False,
            distributed_slack=False, control_q=False, control_taps_modules=False, control_taps_phase=False,
            control_remote_voltage=True, tolerance=TOLERANCE_PU, max_iter=MAX_ITERATIONS, verbose=0)
        return {"base": vg.PowerFlowDriver(grid, vg.PowerFlowOptions(**settings)), "grid": grid,
                "driver": vg.PowerFlowTimeSeriesDriver(grid, vg.PowerFlowOptions(**settings, use_stored_guess=True)),
                "n": n, "bus_ids": [str(b.code) for b in grid.get_buses()], "v": None}

    def solve(self, model, threads=1):
        assert threads == 1, "VeraGrid's own time-series engine runs on one thread"
        base, n = model["base"], model["n"]
        base.run()
        if not base.results.converged:
            raise DidNotConverge("base case: NR did not converge")
        for bus, v in zip(model["grid"].get_buses(), base.results.voltage):
            bus.Vm0, bus.Va0 = float(abs(v)), float(np.angle(v))
            bus.Vm0_prof.set(np.full(n, bus.Vm0))
            bus.Va0_prof.set(np.full(n, bus.Va0))
        driver = model["driver"]
        driver.run()
        bad = int((~np.asarray(driver.results.converged_values, bool)).sum())
        if bad:
            raise DidNotConverge(f"{bad} of {model['n']} scenarios did not converge")
        model["v"] = driver.results.voltage

    def solution(self, model, case):
        v = np.asarray(model["v"], dtype=np.complex128)   # widened before abs/angle, which numpy would do in float32
        assert v.shape == (model["n"], len(model["bus_ids"]))
        return BatchSolution(model["bus_ids"], np.abs(v), np.angle(v, deg=True))
