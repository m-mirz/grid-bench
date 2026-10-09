"""OpenCGMES: SOPTIM's CIM/XML parser and writer for Apache Jena (Java,
`de.soptim.opencgmes:cimxml`), through JPype (`adapters/jvm.py`).

The parser is given the CGMES 3.0 vocabularies first
(`parseAndRegisterCimProfile` on the ENTSO-E RDFS, `RDFS`), once per
process, untimed: that is how OpenCGMES is meant to be used, and only then
does it type literals by the schema (without it every value is a plain
string, and Svedala's validation reported 26,382 datatype violations on
values such as "6.307673"). The 2020 Header RDFS is not registered:
OpenCGMES rejects it (its header binds `cim` to another namespace); the
2019 one is.

- import: `CimXmlParser.parseCimModel` on each file, one Jena dataset per
  file (body in the default graph, header in a named graph), as OpenCGMES
  keeps CIM/XML documents.
- export: `CimXmlWriter.writeCimModel` of each dataset to its own file,
  unsorted (sorting costs time and changes nothing a reader needs).
- validate: parse the files, merge them (`fullModelToSingleGraph`, header
  included, into one `MultiUnion` with no copy), add the CGMES vocabulary
  graph, and run Jena SHACL (`ShaclValidator`, the engine OpenCGMES's
  validation UI uses) with the ENTSO-E CGMES 3.0 shapes
  (`cases.registry.CIM_SHAPES`; OpenCGMES ships no rules of its own). The
  vocabulary is in the data graph because SHACL's `sh:class` resolves
  subclasses through the data graph's `rdfs:subClassOf`: without it, a
  reference to a Breaker where a Switch is required fails (3,580 findings
  on Svedala). Shapes and vocabulary are parsed once per process, in the
  untimed warm-up, as triplets compiles its shapes once.
- Not done: no per-profile phase; the merged model is validated in one
  pass, as for triplets.
"""
from collections import Counter
from pathlib import Path

from adapters import jvm
from adapters.cim_adapter import CimAdapter
from cases.registry import CIM_SHAPES, DATA, cim_files

CIM = "http://iec.ch/TC57/CIM100#"
RDFS = [p for p in sorted((DATA / "application-profiles-library" / "CGMES" / "RDFS").glob("*.rdf"))
        if p.name != "61970-600-2_Header-AP-Voc-RDFS2020.rdf"]


class OpencgmesCim(CimAdapter):
    name = "opencgmes"
    display_name = "OpenCGMES"
    color = "#4a3aa8"   # tools/palette.py OPENCGMES: slot 7 violet, dotted
    package = "cimxml"
    modules = ("jpype", "jpype.imports")
    language = "java"
    validates = True
    settings = {"profiles": "CGMES 3.0 RDFS registered with the parser", "export": "CimXmlWriter, unsorted",
                "validator": "Jena SHACL on the merged model + vocabulary",
                "rules": "ENTSO-E CGMES 3.0 SHACL (CIM_SHAPES)", "jvm_heap": jvm.HEAP}

    def __init__(self):
        jvm.start()
        from de.soptim.opencgmes.cimxml.parser import CimXmlParser
        from java.nio.file import Paths
        self.parser = CimXmlParser()
        for p in RDFS:
            self.parser.parseAndRegisterCimProfile(Paths.get(str(p)))
        self.shapes = self.vocabulary = None

    def version(self):
        return jvm.jar_version("cimxml")

    def dependencies(self):
        return jvm.dependencies("jena-arq", "jena-shacl")

    def _parse(self, case):
        from java.nio.file import Paths
        return {p.name: self.parser.parseCimModel(Paths.get(str(p))) for p in cim_files(case)}

    def load(self, case):
        return self._parse(case)

    def export(self, model, out_dir):
        from de.soptim.opencgmes.cimxml.writer import CimXmlWriter
        from java.nio.file import Paths
        writer = CimXmlWriter()
        for name, dataset in model.items():
            writer.writeCimModel(Paths.get(str(Path(out_dir) / name)), dataset, False)

    def validate(self, case):
        from org.apache.jena.graph.compose import MultiUnion
        from org.apache.jena.shacl import ShaclValidator
        if self.shapes is None:
            self._read_rules()
        data = MultiUnion()
        for dataset in self._parse(case).values():
            data.addGraph(dataset.fullModelToSingleGraph())
        data.addGraph(self.vocabulary)
        entries = ShaclValidator.get().validate(self.shapes, data).getEntries()
        return {"violations": entries.size(),
                "by_severity": dict(Counter(str(e.severity().level().getLocalName()) for e in entries))}

    def _read_rules(self):
        from org.apache.jena.riot import Lang, RDFDataMgr
        from org.apache.jena.shacl import Shapes
        from org.apache.jena.sparql.graph import GraphFactory
        shapes = GraphFactory.createDefaultGraph()
        for p in CIM_SHAPES:
            RDFDataMgr.read(shapes, str(p))
        self.shapes = Shapes.parse(shapes)
        self.vocabulary = GraphFactory.createDefaultGraph()
        for p in RDFS:
            RDFDataMgr.read(self.vocabulary, str(p), Lang.RDFXML)

    def counts(self, model):
        from org.apache.jena.graph import Node, NodeFactory
        from org.apache.jena.vocabulary import RDF

        def n(cls):   # distinct objects: SSH, TP and SV repeat the type of what they describe
            node = NodeFactory.createURI(CIM + cls)
            return len({str(t.getSubject()) for d in model.values()
                        for t in d.getDefaultGraph().find(Node.ANY, RDF.type.asNode(), node).toList()})
        return {"read": sum(d.getDefaultGraph().size() for d in model.values()), "unit": "triples",
                "lines": n("ACLineSegment"), "generators": n("SynchronousMachine"),
                "loads": sum(n(c) for c in ("ConformLoad", "NonConformLoad", "EnergyConsumer")),
                "substations": n("Substation")}
