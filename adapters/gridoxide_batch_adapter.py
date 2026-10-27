"""gridoxide: batch power flow through `PowerFlowModel.solve_batch`
(`native`), Newton-Raphson on rayon threads, in Rust, driven through its
Python bindings.

Input: the base case's `.m` through `PowerFlowModel.from_matpower`, as in the
power-flow adapter (`adapters/gridoxide_adapter.py`), with its settings
(backend, `tol=TOLERANCE_PU`, `max_iter=MAX_ITERATIONS`).

Scenarios: gridoxide's model holds one net injection per bus, generation
less load, per unit on the case's `baseMVA`; a scenario overrides it. So
each scenario sets, at every energized bus with a load or an online
generator, P to the sum of its online generators' scenario `Pg` less its
scenario `Pd`, and Q to the sum of their `Qg` (the case's; the sweep keeps
it) less its scenario `Qd`. Exactly the injection the importer computes
from the case (`src/matpower.rs`), so the zero scenario is the case.
Joined by MATPOWER bus number (`bus_numbers()`, the bus index's number) and
gen row, never by position. Built in `load` and handed to the model with
`set_scenarios` (input data, like the network), so the timed call converts
nothing.

Settings:
- The timed `solve` is the base case from flat (`solve(init="flat")`, as
  the power-flow adapter) and then `solve_batch(threads=..., start="last")`:
  every scenario starts from that solution, the problem's base-case start.
- `threads`: the thread count. gridoxide hands scenarios to its workers
  one at a time from a shared counter; each worker keeps one solver and its
  symbolic factorization across its scenarios. Results do not depend on the
  thread count, which the oracle checks per thread count.
- A batch with a scenario that did not converge fails, with the count.
- Reading the voltages is in the timed call: `solve_batch` returns every
  scenario's |V| and angle as Python lists.
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case, scenarios
from adapters.gridoxide_adapter import GridoxideAdapter
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.matpower import BUS_I, BUS_TYPE, GEN_BUS, GEN_STATUS, ISOLATED, QG, parse_m
from cases.registry import CASES


def overrides(case: str, numbers: list[int]) -> list[list[tuple[int, float, float]]]:
    """Every scenario as `(bus index, p, q)` overrides, per unit; see the
    module docstring."""
    path = CASES[base_case(case)]["file"]
    mpc = parse_m(path)
    base = float(mpc["baseMVA"])
    index = {int(b): i for i, b in enumerate(numbers)}
    energized = {int(b) for b, t in zip(mpc["bus"][:, BUS_I], mpc["bus"][:, BUS_TYPE]) if t != ISOLATED}
    sweep = scenarios(case)
    n, nb = len(sweep["scale"]), len(numbers)
    p, q = np.zeros((n, nb)), np.zeros((n, nb))
    touched = np.zeros(nb, bool)

    gen = mpc["gen"]
    online = gen[:, GEN_STATUS] > 0
    assert set(np.flatnonzero(online)) == set(sweep["gen_row"].tolist()), "the sweep's gen rows are the online ones"
    for k, row in enumerate(sweep["gen_row"]):
        bus = int(gen[row, GEN_BUS])
        if bus in energized:
            i = index[bus]
            p[:, i] += sweep["pg"][:, k]
            q[:, i] += gen[row, QG]
            touched[i] = True
    for k, bus in enumerate(sweep["load_bus"]):
        i = index[int(bus)]
        p[:, i] -= sweep["pd"][:, k]
        q[:, i] -= sweep["qd"][:, k]
        touched[i] = True
    cols = np.flatnonzero(touched)
    p, q = p[:, cols] / base, q[:, cols] / base
    return [list(zip(cols.tolist(), p[s].tolist(), q[s].tolist())) for s in range(n)]


class GridoxideBatch(BatchAdapter):
    name = GridoxideAdapter.name
    display_name = GridoxideAdapter.display_name
    color = GridoxideAdapter.color
    package = GridoxideAdapter.package
    language = GridoxideAdapter.language
    modules = GridoxideAdapter.modules
    backend = GridoxideAdapter.backend
    mode = "native"
    settings = GridoxideAdapter.settings | {"mode": "native", "batch_api": "PowerFlowModel.solve_batch",
                                            "start": "base-case solution (start=\"last\")",
                                            "threads": "the thread count"}

    def load(self, case):
        import gridoxide
        model = gridoxide.PowerFlowModel.from_matpower(
            str(CASES[base_case(case)]["file"]), backend=self.backend, tol=TOLERANCE_PU, max_iter=MAX_ITERATIONS)
        numbers = model.bus_numbers()
        scen = overrides(case, numbers)
        model.set_scenarios(scen)
        return {"model": model, "n": len(scen), "bus_ids": [str(b) for b in numbers], "results": None}

    def solve(self, model, threads=1):
        m = model["model"]
        try:
            m.solve(init="flat")
        except RuntimeError as e:
            raise DidNotConverge(f"base case: {e}") from e
        results = m.solve_batch(threads=threads, start="last")
        bad = sum(not r.converged for r in results)
        if bad:
            raise DidNotConverge(f"{bad} of {model['n']} scenarios did not converge")
        model["results"] = results

    def solution(self, model, case):
        results = model["results"]
        vm = np.array([r.voltage_mag for r in results])
        va = np.rad2deg(np.array([r.voltage_ang for r in results]))
        assert vm.shape == (model["n"], len(model["bus_ids"]))
        return BatchSolution(model["bus_ids"], vm, va)
