"""power-grid-model: N-1 contingency analysis as a batch calculation
(`native`).

PGM's way to run contingencies is the same batch call as for a sweep, with
update data that switches the outaged branch off (`from_status`,
`to_status` = 0) in its scenario. Every scenario is independent and runs on
`threading` threads (see adapters/pgm_batch_adapter.py).

Start: PGM takes no initial voltages, so every outage starts from the flat
start (`calculation_initialization="flat"`), not from the base case's solution as the problem asks: a harder
start, never an easier one. The base case is not solved, since nothing would
use it.

Input and settings: the power-flow adapter's (`adapters/pgm_adapter.py`),
with its conversion losses and its experimental PV regulators, and
`output_component_types=["node"]`, `continue_on_batch_error=False` as in the
batch adapter.

Joins: `cases.prep` records each branch row's PGM id (`pgm_branch_ids_path`);
the update has one column per outaged element of each component (`line`,
`transformer`), switched off in its own scenario and on in the others.
"""
import json

import numpy as np

from adapters.batch_adapter import BatchSolution, ContingencyAdapter, base_case, outages
from adapters.pgm_adapter import PgmAdapter
from adapters.pgm_batch_adapter import PgmBatch
from cases.registry import pgm_branch_ids_path, pgm_json_path


class PgmN1(ContingencyAdapter):
    name = PgmAdapter.name
    display_name = PgmAdapter.display_name
    color = PgmAdapter.color
    package = PgmAdapter.package
    language = PgmAdapter.language
    modules = PgmBatch.modules
    mode = "native"
    settings = PgmBatch.settings | {"start": "flat (PGM takes no initial voltages)",
                                    "update": "from_status, to_status of the outaged branch"}
    solve = PgmBatch.solve

    def load(self, case):
        from power_grid_model import ComponentType, DatasetType, PowerGridModel, initialize_array
        from power_grid_model.utils import json_deserialize
        base = base_case(case)
        dataset = json_deserialize(pgm_json_path(base).read_text())
        ids = {int(r): int(i) for r, i in json.loads(pgm_branch_ids_path(base).read_text()).items()}
        rows = outages(case)["branch_row"]
        out_ids = np.array([ids[int(r)] for r in rows])
        update = {}
        for component in (ComponentType.line, ComponentType.transformer):
            mine = np.isin(out_ids, dataset[component]["id"]) if component in dataset else np.zeros(len(rows), bool)
            if not mine.any():
                continue
            cols = out_ids[mine]
            u = initialize_array(DatasetType.update, component, (len(rows), len(cols)))
            u["id"] = cols[None, :]
            status = np.ones((len(rows), len(cols)), dtype=np.int8)
            status[np.flatnonzero(mine), np.arange(len(cols))] = 0
            u["from_status"], u["to_status"] = status, status
            update[component] = u
        assert sum(int((u["from_status"] == 0).sum()) for u in update.values()) == len(rows), "every outage joined"
        return {"model": PowerGridModel(dataset), "update": update,
                "node_ids": [str(i) for i in dataset[ComponentType.node]["id"]], "result": None}

    def solution(self, model, case):
        return PgmBatch.solution(self, model, case)
