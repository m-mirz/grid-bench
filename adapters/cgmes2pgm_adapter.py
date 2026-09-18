"""power-grid-model fed by SOPTIM's cgmes2pgm (cgmes2pgm_suite / _converter).

cgmes2pgm reads CGMES only through a SPARQL endpoint. It is run the way its
suite runs it (`cgmes2pgm_suite.app`): an Apache Jena Fuseki server (the
suite's own image, `docker/fuseki/`, a sidecar container), a fresh in-memory
dataset, the profiles uploaded with `RdfXmlDirectoryImport` (one named graph
per profile), then `CgmesToPgmConverter(...).convert()` and a
`PowerGridModel`. All of that is the timed `import`: it is how this tool
gets from CGMES files to a model. Peak memory covers the Python process
only, not the Fuseki JVM.

cgmes2pgm is written for state estimation, and converts for it:
- every SynchronousMachine becomes a `sym_gen` of type `const_power` with its
  SSH p and q; the RegulatingControl target is read but only kept in
  `extra_info` (`_targetVoltage`), so no generator regulates voltage;
- the slack is a PGM `source` at the machine with the lowest
  `referencePriority` != 0, with `u_ref = 1` (nominal voltage), not the
  machine's voltage setpoint; a case without such a machine is rejected
  ("Grid has no SynchronousMachines or ExternalNetworkInjections"), which
  is the case for pypowsybl's CGMES export.
That is a different problem from the one every other tool here is given
(generator voltage regulation as the case defines it). It is benchmarked as
the tool is built, and the oracle reports where the solution departs from
the case. cgmes2pgm pins power-grid-model 1.12.x, so this column runs
its own image and cannot share the 1.13 PGM column's.

Settings:
- `ConverterOptions()` defaults, as the suite's example configuration.
- `split_profiles=True`: one named graph per profile, the suite's default.
- Power flow: the public `calculate_power_flow`, Newton-Raphson,
  `error_tolerance=TOLERANCE_PU`, `max_iterations=MAX_ITERATIONS` (no
  experimental features: the converter creates no voltage regulators).
- Bus mapping: each PGM node's `extra_info["_mrid"]` is its
  TopologicalNode mRID.
"""
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.registry import cgmes_files
from oracle.cgmes_sv import mrid

FUSEKI_URL = os.environ.get("FUSEKI_URL", "http://localhost:3030")
DATASET = "bench"
CIM100 = "http://iec.ch/TC57/CIM100#"


class Cgmes2pgmAdapter(SolverAdapter):
    name = "cgmes2pgm"
    display_name = "PGM via cgmes2pgm"
    color = "#4a3aa7"
    package = "cgmes2pgm_converter"
    modules = ("cgmes2pgm_converter", "cgmes2pgm_converter.common", "cgmes2pgm_suite.rdf_store",
               "power_grid_model", "power_grid_model.errors")
    language = "c++"
    families = ("cgmes", "converted-cimoxide", "converted-pypowsybl")
    settings = {"converter": "cgmes2pgm 0.4.3, default ConverterOptions", "rdf_store": "Jena Fuseki (in-memory)",
                "generators": "const_power (no voltage regulation)", "slack": "source, u_ref = 1 (nominal)",
                "calculation_method": "newton_raphson", "tolerance_pu": TOLERANCE_PU,
                "max_iteration": MAX_ITERATIONS}

    def tags(self) -> list[str]:
        return super().tags() + ["sparql", "state-estimation-converter"]

    def dependencies(self) -> dict[str, str]:
        from importlib.metadata import version
        return {**super().dependencies(), "cgmes2pgm_suite": version("cgmes2pgm_suite"),
                "power-grid-model": version("power-grid-model")}

    def _fuseki(self):
        from cgmes2pgm_suite.rdf_store import FusekiServer
        fuseki = FusekiServer(FUSEKI_URL)
        for _ in range(60):   # the sidecar may still be starting
            if fuseki.ping():
                return fuseki
            time.sleep(1)
        raise RuntimeError(f"Fuseki not reachable at {FUSEKI_URL}")

    def load(self, case):
        from cgmes2pgm_converter import CgmesToPgmConverter
        from cgmes2pgm_converter.common import CgmesDataset, ConverterOptions
        from cgmes2pgm_suite.rdf_store import RdfXmlDirectoryImport
        from power_grid_model import PowerGridModel

        fuseki = self._fuseki()
        fuseki.delete_dataset(DATASET)
        fuseki.create_dataset(DATASET)
        dataset = CgmesDataset(base_url=f"{FUSEKI_URL}/{DATASET}", cim_namespace=CIM100, split_profiles=True)
        with tempfile.TemporaryDirectory() as tmp:   # the importer takes one directory of profiles
            for f in cgmes_files(case):
                os.symlink(f, Path(tmp) / f.name)
            RdfXmlDirectoryImport(dataset=dataset, target_graph="default", base_iri=dataset.base_url,
                                  split_profiles=True).import_directory(tmp)
        input_data, extra_info = CgmesToPgmConverter(dataset, options=ConverterOptions()).convert()
        node_mrids = [mrid(extra_info[int(i)]["_mrid"]) for i in input_data["node"]["id"]]
        return {"model": PowerGridModel(input_data), "node_mrids": node_mrids, "result": None}

    def solve(self, model):
        from power_grid_model import CalculationMethod
        from power_grid_model.errors import PowerGridError
        try:
            model["result"] = model["model"].calculate_power_flow(
                calculation_method=CalculationMethod.newton_raphson,
                error_tolerance=TOLERANCE_PU, max_iterations=MAX_ITERATIONS)
        except PowerGridError as e:
            raise DidNotConverge(f"{type(e).__name__}: {str(e).splitlines()[0]}") from e

    def solution(self, model, case):
        node = model["result"]["node"]
        vm = dict(zip(model["node_mrids"], node["u"] / 1e3))            # V -> kV
        va = dict(zip(model["node_mrids"], np.rad2deg(node["u_angle"])))
        return Solution(vm, va)
