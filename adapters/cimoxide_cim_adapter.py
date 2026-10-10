"""cimoxide: a Rust CGMES library (typed class tables generated from the
ENTSO-E RDFS, SHACL validation), through its PyO3 bindings.

- import: `cimoxide.decode_files` on the case's files, which parses them in
  parallel (one thread per file, GIL released) and merges them into one
  dataset. Its default, as a user calls it.
- export: `CimDataset.write_xml_files` with the case's profiles
  (`cases.registry.cim_profiles`), one file per profile, each with the
  header decoded for it. cimoxide assigns an object to a profile by the
  RDFS; EQBD, whose classes the RDFS also declares in EQ, falls back to
  plain class membership, so Svedala's 5 kB boundary file comes back as
  2 MB: every object of a boundary class, not only the boundary's. That
  is how cimoxide writes it, and it is left so.
- validate: `cimoxide.validate_files` with its defaults: its own rule set
  (the ENTSO-E CGMES 3.0 SHACL shapes compiled into it, plus hand-written
  SPARQL rules), per-profile phase then cross-profile phase, profiles and
  solved/not-solved auto-detected, no common or quality checks (both are
  opt-in extras beyond ENTSO-E's rules), nothing silenced.
- Not done: no `drop_sparql_store` or other tuning between rounds; nothing
  is cached between `load` calls.
"""
from collections import Counter

from adapters.cim_adapter import CimAdapter
from cases.registry import cim_files, cim_profiles


class CimoxideCim(CimAdapter):
    name = "cimoxide"
    display_name = "cimoxide"
    color = "#eda101"   # tools/palette.py CIMOXIDE: slot 4 yellow, dotted
    package = "cimoxide"
    modules = ("cimoxide",)
    language = "rust"
    validates = True
    settings = {"decode": "decode_files (parallel per file)", "export": "write_xml_files, one file per profile",
                "rules": "built-in (ENTSO-E CGMES 3.0 SHACL + SPARQL rules)", "common": False, "quality": False}

    def load(self, case):
        import cimoxide
        return {"ds": cimoxide.decode_files([str(p) for p in cim_files(case)]), "profiles": cim_profiles(case)}

    def export(self, model, out_dir):
        model["ds"].write_xml_files(str(out_dir), model["profiles"])

    def validate(self, case):
        import cimoxide
        found = cimoxide.validate_files([str(p) for p in cim_files(case)])
        return {"violations": len(found), "by_severity": dict(Counter(v.severity.removeprefix("sh:") for v in found))}

    def counts(self, model):
        ds = model["ds"]
        return {"read": len(ds), "unit": "objects", "lines": ds.count_type("ACLineSegment"),
                "generators": ds.count_type("SynchronousMachine"),
                "loads": sum(ds.count_type(c) for c in ("ConformLoad", "NonConformLoad", "EnergyConsumer")),
                "substations": ds.count_type("Substation")}
