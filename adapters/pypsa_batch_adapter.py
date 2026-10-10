"""PyPSA: batch power flow over snapshots, `Network.pf()` (`native`, one
thread).

A PyPSA network carries time series natively: one snapshot per scenario,
`loads_t.p_set`, `loads_t.q_set` and `generators_t.p_set` per snapshot, and
one `pf()` call solves every snapshot (a Newton-Raphson per snapshot, one
after another in this process; no thread setting).

Input and settings: the power-flow adapter's (`adapters/pypsa_adapter.py`):
`import_from_pypower_ppc`, transformers as pi-models, `x_tol=TOLERANCE_PU`.
Start: the timed call solves the base case on the power-flow adapter's own
network (one snapshot, `use_seed=False`, flat), writes its voltages into
`buses_t.v_mag_pu` and `buses_t.v_ang` of every snapshot, and calls
`pf(use_seed=True)`, so each snapshot starts from the base solution (not
from the previous snapshot's result), as the N-1 adapter seeds its
outages. Its import losses stand
(out-of-service branches imported in service, offline generators
regulating).

Joins, built in `load`: the importer makes one load per bus with a nonzero
`Pd` or `Qd`, its `bus` the MATPOWER number, and one generator per gen row,
in order (asserted against the case's gen buses). Offline generators keep
their own P. A snapshot that does not converge fails the batch, with the
count.
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case, columns, scenarios
from adapters.pypsa_adapter import PypsaAdapter
from adapters.solver_adapter import TOLERANCE_PU, DidNotConverge
from cases.matpower import GEN_BUS, parse_m
from cases.registry import CASES


class PypsaBatch(BatchAdapter):
    name = PypsaAdapter.name
    display_name = PypsaAdapter.display_name
    color = PypsaAdapter.color
    package = PypsaAdapter.package
    language = PypsaAdapter.language
    modules = PypsaAdapter.modules
    mode = "native"
    threaded = False
    settings = PypsaAdapter.settings | {"mode": "native", "batch_api": "Network.pf over snapshots",
                                        "start": "base-case solution (use_seed)"}
    single = PypsaAdapter()

    def load(self, case):
        import pandas as pd
        base = base_case(case)
        sweep = scenarios(case)
        n = self.single.load(base)["network"]
        snapshots = pd.RangeIndex(len(sweep["scale"]))
        n.set_snapshots(snapshots)

        col = columns(n.loads.bus.astype(int), sweep["load_bus"])
        assert len(n.loads) == len(sweep["load_bus"]), "one load per loaded bus"
        n.loads_t.p_set = pd.DataFrame(sweep["pd"][:, col], index=snapshots, columns=n.loads.index)
        n.loads_t.q_set = pd.DataFrame(sweep["qd"][:, col], index=snapshots, columns=n.loads.index)

        gen = parse_m(CASES[base]["file"])["gen"]
        assert (n.generators.bus.astype(int).to_numpy() == gen[:, GEN_BUS].astype(int)).all(), \
            "one generator per gen row"
        p = np.repeat(n.generators.p_set.to_numpy(float)[None, :], len(snapshots), axis=0)
        p[:, sweep["gen_row"]] = sweep["pg"]
        n.generators_t.p_set = pd.DataFrame(p, index=snapshots, columns=n.generators.index)
        base_model = self.single.load(base)
        assert (base_model["network"].buses.index == n.buses.index).all()
        return {"network": n, "base": base_model, "bus_ids": [str(b) for b in n.buses.index]}

    def solve(self, model, threads=1):
        import pandas as pd
        assert threads == 1, "PyPSA solves snapshots one after another"
        n, base = model["network"], model["base"]
        self.single.solve(base)
        b = base["network"].buses_t
        seed = lambda x: pd.DataFrame(np.repeat(x.iloc[[0]][n.buses.index].to_numpy(), len(n.snapshots), axis=0),
                                      index=n.snapshots, columns=n.buses.index)
        n.buses_t.v_mag_pu, n.buses_t.v_ang = seed(b.v_mag_pu), seed(b.v_ang)
        res = n.pf(x_tol=TOLERANCE_PU, use_seed=True)
        converged = np.asarray(res.converged.values, bool)
        if not converged.all():
            raise DidNotConverge(f"{int((~converged).sum())} of {converged.size} snapshots did not converge")

    def solution(self, model, case):
        n = model["network"]
        vm, va = n.buses_t.v_mag_pu[n.buses.index], n.buses_t.v_ang[n.buses.index]
        return BatchSolution(model["bus_ids"], vm.to_numpy(), np.rad2deg(va.to_numpy()))
