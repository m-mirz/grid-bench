"""lightsim2grid: batch power flow through `InjectionSweepCPP` (`native`).

lightsim2grid has two batch computers on one fixed topology: `TimeSeriesCPP`
chains the steps (each starts from the previous result, so it cannot be
split over threads), and `InjectionSweepCPP`, whose every step starts from
the same voltage and which splits the steps over `nb_thread` OS threads,
each with its own solver. The second is this problem (every scenario from
the base case's solution); it is driven here without grid2op, as the
power-flow adapter drives `ac_pf`.

Input: the base case's `.mat` through `init_from_matpower`, as in the
power-flow adapter (`adapters/lightsim2grid_adapter.py`), with its documented
loss (online generators on PQ-typed buses regulate voltage).

Joins, built in `load` and asserted against the case: `init_from_matpower`
makes one load per bus with a nonzero `Pd` or `Qd`, on the bus whose row in
the `.m` is its `bus_id`, and one generator per gen row, in order (online or
not). Scenario values go to the loads by MATPOWER bus number and to the
generators by gen row; offline generators keep their own P.

Settings:
- `change_algorithm(NR_KLU)`, as in the power-flow adapter.
- `modify_gen_p`, `modify_load_p`, `modify_load_q` (scenarios x elements;
  no static generators), then `compute(v_init, MAX_ITERATIONS,
  TOLERANCE_PU)` with `v_init` the power-flow adapter's flat start. The
  `compute_Vs` call that does both at once is deprecated in 1.0.
- `init_from_n_powerflow=True`: `compute` first solves the base case from
  the `v_init` it is given (the power-flow adapter's flat start) and starts
  every step from that solution: the problem's base-case start, base solve
  included in the timed call, as the N-1 adapter does.
- `nb_thread`: the thread count. The steps are split into contiguous
  ranges, one per thread; lightsim2grid documents that the results do not
  depend on it, which the oracle checks per thread count.
- A batch with a step that did not converge (`converged_mask`, which is
  False for a step that was not invertible or did not converge) fails,
  with the count.
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case, columns, scenarios
from adapters.lightsim2grid_adapter import Lightsim2gridAdapter
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.matpower import BUS_I, GEN_BUS, PD, parse_m
from cases.registry import CASES, mat_path


def injections(case: str, grid) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The sweep's generator P, load P and load Q, scenarios x elements, in
    `grid`'s element order (the joins in the module docstring, asserted),
    and the case's MATPOWER bus numbers. Shared with gpusim2grid, which is
    seeded from the same grid."""
    mpc = parse_m(CASES[base_case(case)]["file"])
    number = mpc["bus"][:, BUS_I].astype(int)
    sweep = scenarios(case)
    n = len(sweep["scale"])
    loads = grid.get_loads()
    load_bus = number[[ld.bus_id for ld in loads]]
    col = columns(load_bus, sweep["load_bus"])
    assert len(set(load_bus)) == len(loads) == len(sweep["load_bus"]), "one load per loaded bus"
    row = {b: i for i, b in enumerate(number)}
    assert np.allclose([ld.target_p_mw for ld in loads], mpc["bus"][[row[b] for b in load_bus], PD])
    gens = grid.get_generators()
    assert len(gens) == len(mpc["gen"]) and \
        (number[[g.bus_id for g in gens]] == mpc["gen"][:, GEN_BUS].astype(int)).all(), "one generator per gen row"
    gen_p = np.repeat(np.array([g.target_p_mw for g in gens])[None, :], n, axis=0)
    gen_p[:, sweep["gen_row"]] = sweep["pg"]
    return (np.ascontiguousarray(gen_p), np.ascontiguousarray(sweep["pd"][:, col]),
            np.ascontiguousarray(sweep["qd"][:, col]), number)


class Lightsim2gridBatch(BatchAdapter):
    name = Lightsim2gridAdapter.name
    display_name = Lightsim2gridAdapter.display_name
    color = Lightsim2gridAdapter.color
    package = Lightsim2gridAdapter.package
    language = Lightsim2gridAdapter.language
    modules = Lightsim2gridAdapter.modules + ("lightsim2grid.injectionSweep",)
    mode = "native"
    settings = Lightsim2gridAdapter.settings | {"mode": "native", "batch_api": "InjectionSweepCPP.compute",
                                                "init_from_n_powerflow": True, "start": "base-case solution",
                                                "nb_thread": "the thread count"}

    def load(self, case):
        from lightsim2grid.algorithm import AlgorithmType
        from lightsim2grid.injectionSweep import InjectionSweepCPP
        from lightsim2grid.network import init_from_matpower
        grid = init_from_matpower(str(mat_path(base_case(case))))
        gen_p, load_p, load_q, number = injections(case, grid)
        n = len(gen_p)
        computer = InjectionSweepCPP(grid)
        computer.change_algorithm(AlgorithmType.NR_KLU)
        computer.init_from_n_powerflow = True
        computer.modify_gen_p(gen_p)
        computer.modify_sgen_p(np.zeros((n, 0)))
        computer.modify_load_p(load_p)
        computer.modify_load_q(load_q)
        v_init = np.full(len(grid.get_bus_vn_kv()), grid.get_init_vm_pu(), dtype=complex)
        return {"computer": computer, "grid": grid, "v_init": v_init, "n": n,
                "bus_ids": [str(b) for b in number], "v": None}

    def solve(self, model, threads=1):
        computer = model["computer"]
        computer.nb_thread = threads
        computer.compute(model["v_init"], MAX_ITERATIONS, TOLERANCE_PU)
        bad = model["n"] - int(np.asarray(computer.converged_mask(), bool).sum())
        if bad:
            raise DidNotConverge(f"{bad} of {model['n']} scenarios did not converge")
        model["v"] = computer.get_voltages()

    def solution(self, model, case):
        v = model["v"]
        assert v.shape == (model["n"], len(model["bus_ids"]))
        return BatchSolution(model["bus_ids"], np.abs(v), np.angle(v, deg=True))
