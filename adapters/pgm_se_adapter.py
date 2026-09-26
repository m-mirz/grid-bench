"""power-grid-model: `calculate_state_estimation`, Newton-Raphson.

Input: the PGM JSON the power-flow adapter reads (`cases.prep`, gridoxide's
converter; every node at a uniform `u_rated` of 1 V, powers in W on the case's
base power), plus sensors built from the case's measurement set. Joins:
- node = MATPOWER bus number (the converter's node ids);
- branch = the converter's id for that `.m` branch row, from
  `<case>.pgm.branch-ids.json` (written by `cases.prep` as the converter
  creates each line or transformer), asserted against its from and to node.
Sensors:
- |V|: `sym_voltage_sensor`, `u_measured` = |V| x `u_rated`, no angle.
- P and Q of one bus, or of one branch end, share one `sym_power_sensor`
  with their own `p_sigma` / `q_sigma`; `measured_terminal_type` is `node`
  for injections, `branch_from` for flows.
- The PF input's `voltage_regulator`s are left out: they are a power-flow
  control, not part of the network an estimator sees.

Known loss of the conversion, reported rather than fixed: the converter
moves a transformer's line charging into two `shunt`s at its end nodes, so a
`branch_from` sensor on such a transformer measures the flow without its
from-side charging, where MATPOWER's from-end flow includes it.

Settings:
- `calculation_method=newton_raphson`: PGM's WLS on the nonlinear equations.
  `iterative_linear` (the default) treats power sensors as currents at the
  previous voltage, a different estimator.
- `error_tolerance=SE_TOLERANCE` (on the voltage update), `max_iterations`.
- PGM chooses its own initial state (not settable).

Result: exact on every `~exact` case the conversion keeps intact (case300
and mvlv1004 included, where its power flow fails on PV buses: estimation
needs none). On `~noisy`, PGM treats every node with no load, generator or
source as an exact zero injection (a shunt does not count), overriding the
node's measurement: case14 bus 7 is estimated at 0 MW exactly, J = 42.1
against the optimum's 41.4. Checked outside the benchmark: with a zero-power
load on those nodes (which changes no equation), PGM reaches the WLS optimum
on case14, case118, case33bw, case4_dist and mvlv1004 (J = 41.36 and 489.9,
as the oracle tests' own Gauss-Newton). That is not done here: the zero-
injection rule is PGM's modelling choice, so it is reported. Beyond it,
case300 misses by the transformer charging above, and the PEGASE and RTE
cases by their phase shifts, which the converter rounds to zero (see the
power-flow adapter).
"""
import json

import numpy as np

from adapters.estimator_adapter import SE_TOLERANCE, EstimatorAdapter
from adapters.pgm_adapter import PgmAdapter
from adapters.solver_adapter import MAX_ITERATIONS, DidNotConverge, Solution
from cases import measurements
from cases.registry import CASES, measurements_path, pgm_branch_ids_path, pgm_json_path


class PgmEstimator(EstimatorAdapter):
    name = PgmAdapter.name
    display_name = PgmAdapter.display_name
    color = PgmAdapter.color
    package = PgmAdapter.package
    language = PgmAdapter.language
    modules = ("power_grid_model", "power_grid_model.utils", "power_grid_model.errors")
    settings = {"calculation_method": "newton_raphson", "error_tolerance": SE_TOLERANCE,
                "max_iterations": MAX_ITERATIONS}

    def load(self, case):
        from power_grid_model import ComponentType, DatasetType, MeasuredTerminalType, PowerGridModel, initialize_array
        from power_grid_model.utils import json_deserialize
        base = CASES[case]["base_case"]
        dataset = json_deserialize(pgm_json_path(base).read_text())
        dataset.pop(ComponentType.voltage_regulator, None)
        branch_ids = {int(r): i for r, i in json.loads(pgm_branch_ids_path(base).read_text()).items()}
        ends = {int(i): (int(f), int(t)) for comp in (ComponentType.line, ComponentType.transformer)
                if comp in dataset for i, f, t in zip(dataset[comp]["id"], dataset[comp]["from_node"],
                                                      dataset[comp]["to_node"])}
        u_rated = dict(zip(dataset[ComponentType.node]["id"].tolist(), dataset[ComponentType.node]["u_rated"]))
        meas = measurements.read(measurements_path(case))
        watt = 1e6   # MW -> W: the converter's powers are in W

        volt, power = [], {}   # power: (object, terminal) -> {p, q, p_sigma, q_sigma}
        for m in meas["measurements"]:
            if m["kind"] == "vm":
                volt.append((m["bus"], m["value"] * u_rated[m["bus"]], m["sigma"] * u_rated[m["bus"]]))
                continue
            if "bus" in m:
                key = (m["bus"], MeasuredTerminalType.node)
            else:
                obj = branch_ids[m["branch_row"]]
                assert ends[obj] == (m["from_bus"], m["to_bus"])
                key = (obj, MeasuredTerminalType.branch_from)
            pq = m["kind"][0]
            power.setdefault(key, {})[pq] = (m["value"] * watt, m["sigma"] * watt)

        next_id = max(int(v["id"].max()) for v in dataset.values() if len(v) and "id" in v.dtype.names) + 1
        vs = initialize_array(DatasetType.input, ComponentType.sym_voltage_sensor, len(volt))
        vs["id"] = np.arange(next_id, next_id + len(volt))
        for k, (bus, u, sigma) in enumerate(volt):
            vs[k]["measured_object"], vs[k]["u_measured"], vs[k]["u_sigma"] = bus, u, sigma
        ps = initialize_array(DatasetType.input, ComponentType.sym_power_sensor, len(power))
        ps["id"] = np.arange(next_id + len(volt), next_id + len(volt) + len(power))
        for k, ((obj, terminal), pq) in enumerate(power.items()):
            assert set(pq) == {"p", "q"}   # cases.measurements always measures P and Q together
            ps[k]["measured_object"], ps[k]["measured_terminal_type"] = obj, terminal
            ps[k]["p_measured"], ps[k]["p_sigma"] = pq["p"]
            ps[k]["q_measured"], ps[k]["q_sigma"] = pq["q"]
        dataset[ComponentType.sym_voltage_sensor] = vs
        dataset[ComponentType.sym_power_sensor] = ps
        return {"model": PowerGridModel(dataset), "node_ids": dataset[ComponentType.node]["id"], "result": None}

    def solve(self, model):
        from power_grid_model import CalculationMethod
        from power_grid_model.errors import PowerGridError
        try:
            model["result"] = model["model"].calculate_state_estimation(
                calculation_method=CalculationMethod.newton_raphson, symmetric=True,
                error_tolerance=SE_TOLERANCE, max_iterations=MAX_ITERATIONS)
        except PowerGridError as e:
            raise DidNotConverge(f"{type(e).__name__}: {str(e).splitlines()[0]}") from e

    def solution(self, model, case):
        node = model["result"]["node"]
        ids = [str(i) for i in model["node_ids"]]
        return Solution(dict(zip(ids, node["u_pu"])), dict(zip(ids, np.rad2deg(node["u_angle"]))))
