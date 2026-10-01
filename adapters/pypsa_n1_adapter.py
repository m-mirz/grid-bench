"""PyPSA: N-1 contingency analysis as a loop of `Network.pf()` (`loop`).

PyPSA's contingency tool, `lpf_contingency`, is linear (a DC power flow per
outage); there is no AC one. So the loop is written here: the power-flow
adapter's `pf` (`adapters/pypsa_adapter.py`, its settings and import losses)
on the base case from flat start, then for every outage its line or
transformer set inactive (`active = False`, PyPSA's switch for taking a
branch out), `pf` started from the base solution, the voltages copied out,
the branch set active again. One thread.

- Start: `use_seed=True`, with the bus voltages (`buses_t.v_mag_pu`,
  `buses_t.v_ang`) reset to the base solution before every outage, since a
  solve writes its result there.
- Join: `import_from_pypower_ppc` makes a transformer of every branch row
  whose ends have different base voltages, whose tap ratio is neither 0 nor
  1, or which has a phase shift, and a line of every other, numbered `T<i>`
  and `L<i>` in file order; the same rule on the prepared case (the input
  PyPSA reads), and every element's buses asserted against its row.
"""
import numpy as np

from adapters.batch_adapter import LoopContingencyAdapter, base_case, outages
from adapters.pypsa_adapter import PypsaAdapter
from adapters.solver_adapter import TOLERANCE_PU, DidNotConverge
from cases.matpower import ANGLE, BASE_KV, BUS_I, F_BUS, RATIO, T_BUS, normalize_for_tools, parse_m
from cases.registry import CASES


class PypsaN1(LoopContingencyAdapter):
    name = PypsaAdapter.name
    display_name = PypsaAdapter.display_name
    color = PypsaAdapter.color
    package = PypsaAdapter.package
    language = PypsaAdapter.language
    modules = PypsaAdapter.modules
    settings = PypsaAdapter.settings | {"mode": "loop", "start": "base-case solution (use_seed)"}
    single = PypsaAdapter()

    def load(self, case):
        base = base_case(case)
        n = self.single.load(base)["network"]
        mpc = normalize_for_tools(parse_m(CASES[base]["file"]))
        branch = mpc["branch"]
        kv = dict(zip(mpc["bus"][:, BUS_I].astype(int), mpc["bus"][:, BASE_KV]))
        f, t = branch[:, F_BUS].astype(int), branch[:, T_BUS].astype(int)
        is_trafo = (np.array([kv[a] != kv[b] for a, b in zip(f, t)]) | ~np.isin(branch[:, RATIO], (0.0, 1.0))
                    | (branch[:, ANGLE] != 0))
        names = np.empty(len(branch), dtype=object)
        names[~is_trafo] = [f"L{i}" for i in range(int((~is_trafo).sum()))]
        names[is_trafo] = [f"T{i}" for i in range(int(is_trafo.sum()))]
        for table, rows in ((n.lines, np.flatnonzero(~is_trafo)), (n.transformers, np.flatnonzero(is_trafo))):
            assert len(table) == len(rows) and (table.bus0.astype(int).to_numpy() == f[rows]).all() and \
                (table.bus1.astype(int).to_numpy() == t[rows]).all(), "one element per branch row, in file order"
        elements = [("lines" if names[r].startswith("L") else "transformers", names[r]) for r in outages(case)["branch_row"]]
        return {"network": n, "outages": outages(case), "elements": elements,
                "bus_ids": [str(b) for b in n.buses.index]}

    def _pf(self, model, use_seed):
        res = model["network"].pf(x_tol=TOLERANCE_PU, use_seed=use_seed)
        if not bool(np.all(res.converged.values)):
            raise DidNotConverge(f"pf did not converge after {int(np.max(res.n_iter.values))} iterations")

    def solve_base(self, model):
        n = model["network"]
        self._pf(model, use_seed=False)
        model["seed"] = n.buses_t.v_mag_pu.copy(), n.buses_t.v_ang.copy()

    def solve_outage(self, model, k):
        n = model["network"]
        table, name = model["elements"][k]
        n.buses_t.v_mag_pu, n.buses_t.v_ang = (x.copy() for x in model["seed"])
        getattr(n, table).at[name, "active"] = False
        self._pf(model, use_seed=True)
        getattr(n, table).at[name, "active"] = True

    def voltages(self, model):
        n = model["network"]
        return n.buses_t.v_mag_pu.iloc[0].to_numpy(), np.rad2deg(n.buses_t.v_ang.iloc[0].to_numpy())
