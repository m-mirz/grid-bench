"""pypowsybl as a CIM library: PowSyBl's CGMES import into its IIDM network
model and its CGMES export from it (Java, compiled with GraalVM
native-image). Unlike the other CIM libraries it does not keep the RDF: it
converts CGMES into a network model and writes CGMES from that model, so
what it reads and writes is a power-system model, not the triples.

- import: `network.load` on the case directory as published (every profile,
  SV included); PowSyBl's CGMES importer finds the profiles there itself,
  so no zip is prepared for it. Importer parameters at their defaults.
- export: `network.save` with `format="CGMES"` into the output directory
  (files `export_<profile>.xml`), `cim-version` 100 (CGMES 3.0, the input's version;
  also PowSyBl's choice for a network read from CGMES 3.0, set so it does
  not depend on that), the default profiles (EQ, TP, SSH, SV). The
  boundary profile is not written back: PowSyBl exports an IGM's own
  profiles, as it is used.
- validate: none; PowSyBl has no CGMES validator.
- `counts`: IIDM identifiables (`get_identifiables`): switches, busbar
  sections, voltage levels and the rest, not CIM objects.
"""
from adapters.cim_adapter import CimAdapter
from cases.registry import CASES


class PypowsyblCim(CimAdapter):
    name = "pypowsybl"
    display_name = "pypowsybl"
    color = "#e87ba4"   # pypowsybl's slot (5, magenta), as in every other problem
    package = "pypowsybl"
    modules = ("pypowsybl", "pypowsybl.network")
    language = "java"
    settings = {"import": "network.load on the case directory, default parameters",
                "export": "network.save CGMES, cim-version 100, default profiles (EQ, TP, SSH, SV)"}

    def load(self, case):
        import pypowsybl.network as pn
        return pn.load(str(CASES[case]["dir"]))

    def export(self, model, out_dir):
        model.save(str(out_dir / "export"), format="CGMES", parameters={"iidm.export.cgmes.cim-version": "100"})

    def counts(self, model):
        return {"read": len(model.get_identifiables()), "unit": "network elements", "lines": len(model.get_lines()),
                "generators": len(model.get_generators()), "loads": len(model.get_loads()),
                "substations": len(model.get_substations())}
