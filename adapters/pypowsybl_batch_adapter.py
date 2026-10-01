"""pypowsybl: batch power flow as a loop of OpenLoadFlow's `run_ac` (`loop`).

pypowsybl has no batch AC power flow: its security analysis runs
contingencies and its sensitivity analysis is linear, neither of which is a
sweep of operating points. A sweep with pypowsybl is this loop: update the
injections, `run_ac`, read the bus voltages. One thread (OpenLoadFlow solves
one network per call).

Every `run_ac` is the power-flow adapter's (`adapters/pypowsybl_adapter.py`),
with its parameters (in particular `voltage_init_mode=UNIFORM_VALUES`: every
scenario starts flat, not from the previous one) and its importer's loss
(transformer line charging on one end).

Joins, built in `load`: PowSyBl's MATPOWER importer names each load
`LOAD-<bus>` and each generator `GEN-<bus>` after its MATPOWER bus number
(one generator per bus asserted, so a bus with two would fail here rather
than be joined by position). Bus voltages are read by the bus ids at the
ends of `LINE-<f>-<t>` and `TWT-<f>-<t>`, which carry the true MATPOWER
numbers, as in the power-flow adapter; that map is built once, and a
scenario reads every bus in one `get_buses` call. The per-scenario update is
one `update_loads` and one `update_generators` call with the scenario's
values; offline generators keep their own.
"""
import numpy as np

from adapters.batch_adapter import LoopBatchAdapter, base_case, columns, scenarios
from adapters.pypowsybl_adapter import PypowsyblAdapter
from cases.matpower import BUS_I, parse_m
from cases.registry import CASES


class PypowsyblBatch(LoopBatchAdapter):
    name = PypowsyblAdapter.name
    display_name = PypowsyblAdapter.display_name
    color = PypowsyblAdapter.color
    package = PypowsyblAdapter.package
    language = PypowsyblAdapter.language
    modules = PypowsyblAdapter.modules
    settings = PypowsyblAdapter.settings | {"mode": "loop", "update": "update_loads, update_generators, then run_ac"}

    def __init__(self):
        self.single = PypowsyblAdapter()

    def load(self, case):
        base = base_case(case)
        model = self.single.load(base)
        net, sweep = model["network"], scenarios(case)

        loads = net.get_loads()
        load_bus = [int(i.removeprefix("LOAD-")) for i in loads.index]
        assert len(load_bus) == len(sweep["load_bus"]), "one load per loaded bus"
        col = columns(load_bus, sweep["load_bus"])

        gens = net.get_generators()
        gen_bus = [int(i.removeprefix("GEN-")) for i in gens.index]
        assert len(set(gen_bus)) == len(gen_bus), "one generator per bus"
        gen_ids = gens.index[columns(sweep["gen_bus"], gen_bus)]   # the generator of each online gen row

        buses, nominal = net.get_buses(), net.get_voltage_levels()["nominal_v"]
        by_number = {}
        for df in (net.get_lines(), net.get_2_windings_transformers()):
            for elem_id, b1, b2 in zip(df.index, df["bus1_id"], df["bus2_id"]):
                _, f, t = elem_id.split("-")
                by_number.setdefault(int(f), b1)
                by_number.setdefault(int(t.split("#")[0]), b2)
        numbers = parse_m(CASES[base]["file"])["bus"][:, BUS_I].astype(int)
        bus_index = [by_number.get(int(b), "") for b in numbers]
        vbase = np.array([nominal[buses.at[b, "voltage_level_id"]] if b in buses.index else np.nan for b in bus_index])
        return {"single": model, "sweep": sweep, "bus_ids": [str(b) for b in numbers], "bus_index": bus_index,
                "vbase": vbase, "load_ids": list(loads.index), "load_col": col, "gen_ids": list(gen_ids)}

    def apply(self, model, sweep, k):
        net = model["single"]["network"]
        net.update_loads(id=model["load_ids"], p0=sweep["pd"][k, model["load_col"]],
                         q0=sweep["qd"][k, model["load_col"]])
        net.update_generators(id=model["gen_ids"], target_p=sweep["pg"][k])

    def voltages(self, model):
        buses = model["single"]["network"].get_buses(attributes=["v_mag", "v_angle"]).reindex(model["bus_index"])
        return buses["v_mag"].to_numpy() / model["vbase"], buses["v_angle"].to_numpy()
