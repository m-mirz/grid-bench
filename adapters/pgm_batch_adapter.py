"""power-grid-model: batch power flow, the tool's own batch API (`native`).

PGM is built for this: a model is constructed once, and one
`calculate_power_flow(update_data=...)` call solves every scenario of a
batch, on as many threads as `threading` says. The power-flow adapter
(`adapters/pgm_adapter.py`) times one scenario; this one times the batch.

Input: the base case's PGM JSON, as in the power-flow adapter, with the same
conversion losses (phase shifts rounded to zero, the slack as a source
behind an impedance, the source's `u_ref` from the slack bus's `Vm`), which
the oracle reports on every scenario.

Scenarios: one update dataset for the whole sweep, scenarios x elements,
built in `load` (it is input data, like the network):
- `sym_load` `p_specified`, `q_specified`: the converter makes one load per
  bus with a nonzero `Pd` or `Qd`, on the node whose id is the MATPOWER bus
  number; joined by that node id (asserted one to one).
- `sym_gen` `p_specified`: the converter sums every online generator of a
  non-slack bus into one `sym_gen` on that node, so the update is the sum of
  the scenario's `Pg` over that bus's gen rows. The slack's `source` has no
  P setpoint to update.

Settings, as in the power-flow adapter (Newton-Raphson, voltage regulators
through `experimental_features="enabled"` on the private
`_calculate_power_flow`, no reactive limits, `calculation_initialization=
"flat"`, `error_tolerance=TOLERANCE_PU`, `max_iterations=MAX_ITERATIONS`),
and:
- `threading`: -1 (PGM's sequential mode, no thread pool) at one thread,
  otherwise the thread count. PGM splits the scenarios over its threads; each
  thread solves its scenarios with its own copy of the model.
- Every scenario starts from the flat start: a batch calculation does not
  carry one scenario's result into the next.
- `output_component_types=["node"]`: only the node voltages, which is all
  the oracle reads and what a sweep user would ask for; PGM otherwise
  computes and copies out every branch and appliance result too.
- `continue_on_batch_error=False`: a scenario that fails fails the batch,
  as a single solve would. PGM's batch error says only that the batch had
  errors; the record carries how many scenarios failed and the first one's
  own error.
"""
import numpy as np

from adapters.batch_adapter import BatchAdapter, BatchSolution, base_case, columns, scenarios
from adapters.pgm_adapter import PgmAdapter
from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge
from cases.registry import pgm_json_path


class PgmBatch(BatchAdapter):
    name = PgmAdapter.name
    display_name = PgmAdapter.display_name
    color = PgmAdapter.color
    package = PgmAdapter.package
    language = PgmAdapter.language
    modules = ("power_grid_model", "power_grid_model.utils", "power_grid_model.errors")
    mode = "native"
    settings = PgmAdapter.settings | {"mode": "native", "batch_api": "calculate_power_flow(update_data)",
                                      "threading": "-1 at 1 thread, else the thread count",
                                      "output_component_types": ["node"]}

    def load(self, case):
        from power_grid_model import ComponentType, DatasetType, PowerGridModel, initialize_array
        from power_grid_model.utils import json_deserialize
        dataset = json_deserialize(pgm_json_path(base_case(case)).read_text())
        sweep = scenarios(case)
        n = len(sweep["scale"])

        loads = dataset[ComponentType.sym_load]
        assert len(np.unique(loads["node"])) == len(loads) == len(sweep["load_bus"]), "one load per loaded bus"
        col = columns(loads["node"], sweep["load_bus"])
        load_upd = initialize_array(DatasetType.update, ComponentType.sym_load, (n, len(loads)))
        load_upd["id"] = loads["id"][None, :]
        load_upd["p_specified"] = sweep["pd"][:, col] * 1e6
        load_upd["q_specified"] = sweep["qd"][:, col] * 1e6

        gens = dataset[ComponentType.sym_gen]
        at = (sweep["gen_bus"][:, None] == gens["node"][None, :]).astype(float)   # gen rows x sym_gens
        assert (at.sum(axis=0) > 0).all(), "a sym_gen with no online generator"
        gen_upd = initialize_array(DatasetType.update, ComponentType.sym_gen, (n, len(gens)))
        gen_upd["id"] = gens["id"][None, :]
        gen_upd["p_specified"] = sweep["pg"] @ at * 1e6

        update = {ComponentType.sym_load: load_upd, ComponentType.sym_gen: gen_upd}
        return {"model": PowerGridModel(dataset), "update": update,
                "node_ids": [str(i) for i in dataset[ComponentType.node]["id"]], "result": None}

    def solve(self, model, threads=1):
        from power_grid_model import CalculationMethod, ComponentType
        from power_grid_model.errors import PowerGridBatchError, PowerGridError
        try:
            model["result"] = model["model"]._calculate_power_flow(  # noqa: SLF001, see the docstring
                calculation_method=CalculationMethod.newton_raphson, symmetric=True,
                error_tolerance=TOLERANCE_PU, max_iterations=MAX_ITERATIONS, calculation_initialization="flat",
                experimental_features="enabled", update_data=model["update"], threading=-1 if threads == 1 else threads,
                output_component_types=[ComponentType.node], continue_on_batch_error=False)
        except PowerGridBatchError as e:
            first = e.errors[0]
            raise DidNotConverge(f"{len(e.failed_scenarios)} of {len(e.failed_scenarios) + len(e.succeeded_scenarios)} "
                                 f"scenarios failed, first {e.failed_scenarios[0]}: "
                                 f"{type(first).__name__}: {str(first).splitlines()[0]}") from e
        except PowerGridError as e:
            raise DidNotConverge(f"{type(e).__name__}: {str(e).splitlines()[0]}") from e

    def solution(self, model, case):
        from power_grid_model import ComponentType
        node = model["result"][ComponentType.node]
        assert (node["id"] == np.array(model["node_ids"], dtype=node["id"].dtype)[None, :]).all()
        return BatchSolution(model["node_ids"], node["u_pu"], np.rad2deg(node["u_angle"]))
