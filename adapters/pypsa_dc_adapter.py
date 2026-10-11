"""PyPSA: DC power flow (`Network.lpf()`). No DC N-1, see
`adapters/pypsa_adapter.py`.

Input and import as the power-flow adapter (`adapters/pypsa_adapter.py`,
`import_from_pypower_ppc`, transformers as pi-models; its import losses,
out-of-service branches imported in service among them, apply here too).
Branch rows are joined to PyPSA's `L<i>`/`T<i>` by `branch_names`
(asserted against every row's buses); `bus0` is the row's from bus, so `p0`
is its from-end flow (MW).

- DC: `lpf()`, PyPSA's linear power flow; angles from `buses_t.v_ang`,
  flows from `lines_t.p0` and `transformers_t.p0`.

Time: `lpf` rebuilds the network's topology and matrices on every call
(case9241pegase 9.4 s, case1354pegase 0.7 s, against 1 ms for
lightsim2grid's `dc_pf`).
"""
import numpy as np

from adapters.batch_adapter import base_case
from adapters.dc_adapter import DcAdapter, DcSolution
from adapters.pypsa_adapter import PypsaAdapter
from cases.matpower import ANGLE, BASE_KV, BUS_I, F_BUS, RATIO, T_BUS, normalize_for_tools, parse_m
from cases.registry import CASES


def branch_names(case: str, n) -> np.ndarray:
    """The PyPSA name (`L<i>` or `T<i>`) of every branch row of `case`."""
    mpc = normalize_for_tools(parse_m(CASES[case]["file"]))
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
    return names


def _results(n, names) -> tuple[np.ndarray, np.ndarray]:
    lines, trafos = n.lines_t.p0.iloc[0], n.transformers_t.p0.iloc[0]
    pf = np.array([(trafos if name.startswith("T") else lines).get(name, np.nan) for name in names], float)
    pf[[not getattr(n, "transformers" if name.startswith("T") else "lines").at[name, "active"] for name in names]] = np.nan
    return np.rad2deg(n.buses_t.v_ang.iloc[0].to_numpy()), pf


class PypsaDc(DcAdapter):
    name = PypsaAdapter.name
    display_name = PypsaAdapter.display_name
    color = PypsaAdapter.color
    package = PypsaAdapter.package
    language = PypsaAdapter.language
    modules = PypsaAdapter.modules
    single = PypsaAdapter()
    settings = {"api": "lpf", "transformer_model": "pi"}

    def load(self, case):
        base = base_case(case)
        n = self.single.load(base)["network"]
        return {"network": n, "names": branch_names(base, n), "bus_ids": [str(b) for b in n.buses.index]}

    def solve(self, model):
        model["network"].lpf()

    def solution(self, model, case):
        va, pf = _results(model["network"], model["names"])
        return DcSolution(model["bus_ids"], va, list(range(len(pf))), pf)
