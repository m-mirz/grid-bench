"""triplets: CIM RDF/XML as one long table of (ID, KEY, VALUE, INSTANCE_ID)
triples, on polars, with CIM/XML export and SHACL validation.

Installed as `triplets[polars,sparql,oxigraph]`: triplets picks its engines
by what is installed (parser, export, SPARQL, validation alike), and these
are the extras its docs name for speed. Without `oxigraph`, the shapes'
`sh:sparql` constraints run on rdflib: Svedala's validation took 430 s
instead of 4 s, and found 10 more violations (17,361 against 17,351), a
difference between triplets' own engines, left as it is.

- import: `polars.read_RDF` on the case's files (the polars engine, as in
  cim-bench).
- export: `export_to_cimxml` with the CGMES 3.0 export schema
  (`ENTSOE_CGMES_3_0_0_552_ED1`), one XML file per input instance
  (`XML_PER_INSTANCE`, no zip), to memory, then written into the output
  directory. Written to memory because triplets 0.2.0 names the files it
  writes after their absolute source paths, so `export_base_path` is
  dropped by `os.path.join` and it writes over the source files.
- validate: `read_RDF` on the files, then `.shacl.validate` against the
  ENTSO-E CGMES 3.0 shapes for these profiles (`cases.registry.CIM_SHAPES`;
  triplets ships no rules of its own) with the same export schema as
  `rdf_map`, the default engine (polars, delegating `sh:sparql` to
  oxigraph). triplets compiles shapes once per process, keyed by content:
  the untimed warm-up compiles them and the timed rounds reuse them, as a
  user validating several models does. The polars engine skips what it
  cannot express (sequence paths, `sh:targetNode`, `sh:target`; it says so
  in warnings); `engine="pyshacl"` would cover them, at rdflib's speed.
  Where RealGrid's validation time goes (230 s): almost all of it in
  evaluating the shapes' `sh:sparql` queries on oxigraph, and 207 s in
  three of them (Terminal phases, and twice BaseVoltage against the
  equipment container). triplets binds `$this` by appending a `VALUES
  ?this {...}` block (6,683 or 7,561 focus nodes) at the end of each
  query's WHERE clause, after its OPTIONALs and FILTERs, and oxigraph joins
  it last: it evaluates the pattern for every equipment first. With the
  same block at the start of the WHERE clause (the same query under SPARQL
  semantics), each of the three runs in under 0.1 s instead of 41-119 s,
  with the same (empty) result. Jena, which binds `$this` before
  evaluating, does not pay this. Reported, not worked around.
- Not done: no `lexical=False`, no `context` enrichment of the report.
"""
from collections import Counter
from pathlib import Path

from adapters.cim_adapter import CimAdapter
from cases.registry import CIM_SHAPES, cim_files


class TripletsCim(CimAdapter):
    name = "triplets"
    display_name = "triplets"
    color = "#008301"   # tools/palette.py TRIPLETS: slot 6 green, dotted
    package = "triplets"
    modules = ("polars", "triplets", "triplets.export", "triplets.export_schema", "triplets.validation")
    language = "python"
    validates = True
    settings = {"engine": "polars (+ oxigraph for SPARQL)", "export_schema": "ENTSOE_CGMES_3_0_0_552_ED1",
                "export_type": "XML_PER_INSTANCE", "rules": "ENTSO-E CGMES 3.0 SHACL (CIM_SHAPES)"}

    def load(self, case):
        import polars
        import triplets  # noqa: F401 - registers read_RDF and the accessors on polars
        return polars.read_RDF([str(p) for p in cim_files(case)])

    def export(self, model, out_dir):
        from triplets.export import ExportType, export_to_cimxml
        from triplets.export_schema import schemas
        for f in export_to_cimxml(model, rdf_map=schemas.ENTSOE_CGMES_3_0_0_552_ED1,
                                  export_type=ExportType.XML_PER_INSTANCE, export_to_memory=True):
            f.seek(0)
            (Path(out_dir) / Path(f.name).name).write_bytes(f.read())

    def validate(self, case):
        from triplets.export_schema import schemas
        found = self.load(case).shacl.validate([str(p) for p in CIM_SHAPES], rdf_map=schemas.ENTSOE_CGMES_3_0_0_552_ED1)
        return {"violations": len(found), "by_severity": dict(Counter(found["SEVERITY"]))}

    def counts(self, model):
        def n(cls):
            table = model.type_tableview(cls, string_to_number=False)
            return 0 if table is None else len(table)
        return {"read": len(model), "unit": "triples", "lines": n("ACLineSegment"), "generators": n("SynchronousMachine"),
                "loads": sum(n(c) for c in ("ConformLoad", "NonConformLoad", "EnergyConsumer")),
                "substations": n("Substation")}
