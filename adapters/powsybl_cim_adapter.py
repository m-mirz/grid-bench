"""PowSyBl's CGMES model (powsybl-core, `com.powsybl:powsybl-cgmes-model`):
CIM/XML loaded into an in-memory RDF4J triplestore, one context per file,
queried by PowSyBl's own SPARQL. This is the layer under PowSyBl's CGMES
import, before any conversion to its network model (pypowsybl's CIM
benchmark is the whole import). Through JPype (`adapters/jvm.py`), as in
cim-bench.

- import: `CgmesModelFactory.create` on a `DirectoryDataSource` over the
  case directory, every file in it (empty base name), the default
  triplestore implementation (RDF4J) and options. Both cases carry their
  boundary in that directory; a case with a separate boundary set would
  need the factory's boundary argument, so it is refused, not ignored.
- export: `CgmesModel.write` into a `DirectoryDataSource` over the output
  directory: every context back to a file of its own name.
- validate: none; PowSyBl has no CGMES validator.
- `counts`: triples over all contexts (SPARQL), and PowSyBl's own queries
  for lines (`acLineSegments`), generators (`synchronousMachinesAll`), loads
  (`energyConsumers`) and substations.
"""
from adapters import jvm
from adapters.cim_adapter import CimAdapter
from cases.registry import CASES


class PowsyblCim(CimAdapter):
    name = "powsybl"
    display_name = "PowSyBl (CgmesModel)"
    color = "#0f8fa9"   # tools/palette.py POWSYBL: slot 9 teal, dotted
    package = "powsybl-cgmes-model"
    modules = ("jpype", "jpype.imports")
    language = "java"
    settings = {"triplestore": "rdf4j (default)", "export": "CgmesModel.write, one file per context",
                "jvm_heap": jvm.HEAP}

    def __init__(self):
        jvm.start()

    def version(self):
        return jvm.jar_version("powsybl-cgmes-model")

    def dependencies(self):
        return jvm.dependencies("powsybl-triple-store-impl-rdf4j", "rdf4j-sail-memory")

    def load(self, case):
        from com.powsybl.cgmes.model import CgmesModelFactory
        from com.powsybl.commons.datasource import DirectoryDataSource
        from java.nio.file import Paths
        assert CASES[case]["boundary"] is None, f"{case}: a separate boundary set is not passed to CgmesModelFactory"
        return CgmesModelFactory.create(DirectoryDataSource(Paths.get(str(CASES[case]["dir"])), ""))

    def export(self, model, out_dir):
        from com.powsybl.commons.datasource import DirectoryDataSource
        from java.nio.file import Paths
        model.write(DirectoryDataSource(Paths.get(str(out_dir)), "export"))

    def counts(self, model):
        triples = model.tripleStore().query("SELECT (COUNT(*) AS ?n) WHERE { GRAPH ?g { ?s ?p ?o } }").get(0).get("n")
        return {"read": int(str(triples)), "unit": "triples", "lines": model.acLineSegments().size(),
                "generators": model.synchronousMachinesAll().size(), "loads": model.energyConsumers().size(),
                "substations": model.substations().size()}
