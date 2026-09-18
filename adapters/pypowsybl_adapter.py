"""pypowsybl: PowSyBl OpenLoadFlow (Java, compiled with GraalVM native-image).

Inputs:
- matpower: `network.load` on the prepared `.mat` (PowSyBl's own MATPOWER
  importer). It maps a transformer's line charging `b` onto the IIDM
  two-winding transformer's single magnetizing susceptance, while MATPOWER
  splits it half to each end. The oracle shows exactly b/2 as Q residual at
  the affected buses (8.26 MVAr on case118 bus 68). That is reported as
  pypowsybl's result, not corrected.
  Bus mapping: every `LINE-<f>-<t>` / `TWT-<f>-<t>` id carries the true
  MATPOWER endpoint numbers; bus ids (`VL-<n>_<k>`) do not, for transformer
  secondaries (established in gridoxide's accuracy_case_suite.py).
- cgmes: `network.load` on a zip of the case's profiles (built by
  `cases.prep`). Buses are mapped to TopologicalNodes through the CGMES
  terminal aliases PowSyBl keeps (`CGMES.Terminal<k>`) and the TP profile.

Settings, mirroring powsybl-benchmark's BASIC parameters:
- `voltage_init_mode=UNIFORM_VALUES`: flat start every solve (not PREVIOUS_VALUES).
- `distributed_slack=False`, `use_reactive_limits=False`.
- `phase_shifter_regulation_on=False`, `transformer_voltage_control_on=False`,
  `shunt_compensator_voltage_control_on=False`: outer-loop controls off.
- `voltageRemoteControl` left at its default (on): a CGMES generator that
  regulates a remote terminal has that as its PV constraint, as SSH defines
  it. Turning it off makes each generator hold the remote target at its own
  terminal, a different problem: on MicroGrid-BE the deviation from the
  published SV goes from 0.48% median / 2.6% max to 2.5% / 6.4%. MATPOWER
  cases have no remote regulation, so this changes nothing there.
- `connected_component_mode=MAIN`: OpenLoadFlow's default, only the main
  component is solved; the oracle's `n_checked` shows what was left out.
- `newtonRaphsonConvEpsPerEq=TOLERANCE_PU`: the common tolerance (OLF's
  default is 1e-4 p.u., four orders looser than the other tools).
- Slack: the case's own. The MATPOWER importer records the REF bus as a
  `slackTerminal` extension, which OpenLoadFlow reads. The CGMES importer
  records `SynchronousMachine.referencePriority` only as a
  `referencePriorities` extension, which OpenLoadFlow uses for the angle
  reference but not for the slack: it would pick its own slack
  (`MOST_MESHED`), a different problem (on a converted case14, a 6e-3 MW
  mismatch appears at bus 4 instead of the slack). So for CGMES input the
  priority-1 generator's voltage level is passed as
  `slackBusSelectionMode=NAME`. A case without priorities keeps
  OpenLoadFlow's choice.
- Not applied: powsybl-benchmark's `MatpowerUtil` phase-shift zeroing for
  the RTE cases. That would change the problem being solved.
"""
from adapters.cgmes_ids import by_node
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import CACHE, CASES, mat_path

SIDES = {"ONE": "1", "TWO": "2", "THREE": "3"}


class PypowsyblAdapter(SolverAdapter):
    name = "pypowsybl"
    display_name = "pypowsybl (OpenLoadFlow)"
    color = "#e87ba4"
    package = "pypowsybl"
    modules = ("pypowsybl", "pypowsybl.network", "pypowsybl.loadflow")
    language = "java"
    families = ("matpower", "cgmes", "converted-cimoxide", "converted-pypowsybl")
    settings = {"voltage_init_mode": "UNIFORM_VALUES", "distributed_slack": False, "use_reactive_limits": False,
                "outer_loop_controls": "off", "remote_voltage_control": True, "connected_component_mode": "MAIN", "tolerance_pu": TOLERANCE_PU,
                "max_iteration": MAX_ITERATIONS}

    def __init__(self):
        import pypowsybl.loadflow as lf
        self.params = lf.Parameters(
            voltage_init_mode=lf.VoltageInitMode.UNIFORM_VALUES,
            distributed_slack=False,
            use_reactive_limits=False,
            phase_shifter_regulation_on=False,
            transformer_voltage_control_on=False,
            shunt_compensator_voltage_control_on=False,
            connected_component_mode=lf.ConnectedComponentMode.MAIN,
            provider_parameters={"newtonRaphsonConvEpsPerEq": str(TOLERANCE_PU),
                                 "maxNewtonRaphsonIterations": str(MAX_ITERATIONS)},
        )

    def load(self, case):
        import pypowsybl.network as pn
        path = mat_path(case) if CASES[case]["family"] == "matpower" else CACHE / f"{case}.zip"
        network = pn.load(str(path))
        return {"network": network, "iterations": None, "params": self._params_for(network)}

    def _params_for(self, network):
        prio = network.get_extensions("referencePriorities")
        prio = prio[prio["priority"] > 0] if len(prio) else prio
        if not len(prio):
            return self.params
        import copy
        gen = prio["priority"].idxmin()
        params = copy.deepcopy(self.params)
        params.provider_parameters = {**params.provider_parameters, "slackBusSelectionMode": "NAME",
                                      "slackBusesIds": network.get_generators().at[gen, "voltage_level_id"]}
        return params

    def solve(self, model):
        import pypowsybl.loadflow as lf
        result = lf.run_ac(model["network"], parameters=model["params"])[0]
        if result.status != lf.ComponentStatus.CONVERGED:
            raise DidNotConverge(f"{result.status.name}: {result.status_text}")
        model["iterations"] = result.iteration_count

    def solution(self, model, case):
        net = model["network"]
        buses = net.get_buses()
        if CASES[case]["family"] == "matpower":
            nominal = net.get_voltage_levels()["nominal_v"]
            vm, va = {}, {}
            for df in (net.get_lines(), net.get_2_windings_transformers()):
                for elem_id, row in df.iterrows():
                    _, f, t = elem_id.split("-")
                    for bus_id, mp in ((row["bus1_id"], f), (row["bus2_id"], t.split("#")[0])):
                        if bus_id in buses.index:
                            b = buses.loc[bus_id]
                            vm[mp] = b["v_mag"] / nominal[b["voltage_level_id"]]
                            va[mp] = b["v_angle"]
            return Solution(vm, va, model["iterations"])

        aliases = net.get_aliases()
        aliases = aliases[aliases["alias_type"].str.match(r"CGMES\.Terminal\d?$")]
        terminals = net.get_terminals()
        side_bus = {(e, SIDES.get(s, "1")): b for e, s, b in
                    zip(terminals.index, terminals["element_side"], terminals["bus_id"])}
        per_terminal = {}
        for elem_id, row in aliases.iterrows():
            side = row["alias_type"][-1] if row["alias_type"][-1].isdigit() else "1"
            bus_id = side_bus.get((elem_id, side))
            if bus_id in buses.index:
                per_terminal[row["alias"]] = (buses.at[bus_id, "v_mag"], buses.at[bus_id, "v_angle"])
        vm, va = by_node(case, per_terminal, "terminal")
        return Solution(vm, va, model["iterations"])
