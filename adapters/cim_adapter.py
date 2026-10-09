"""The interface of a CIM library: read CGMES, write it back, validate it.

Modelled on cim-bench's `ParserAdapter` (load, export; its query tests are
left out) and, unlike every other problem here, timed only: there is no
oracle verdict (AGENTS.md, rule 1, the documented exception). What a tool
read and what its validator found are recorded for information, never
graded: tools read CGMES into different things (a triplestore, typed
objects, a network model) and validate against different rules.

The same problem for every tool:

- Input: every profile of the case as published (`cases.registry.cim_files`,
  SV included), read from the uncompressed XML files, never from a zip
  prepared for the tool.
- `load` (timed as `import`): files to the tool's in-memory model.
- `export` (timed as `export`): that model to CGMES RDF/XML files in an
  empty directory, as the tool writes them; nothing is re-read or compared.
- `validate` (timed as `validate`, adapters with `validates` only): the
  files to a violation report in one call, parsing included. cimoxide's
  Python API validates files only, so this is the one definition every
  validator can meet; `import` beside it shows what parsing costs.
  cimoxide runs its own rules; triplets and OpenCGMES, which ship none,
  get the ENTSO-E CGMES 3.0 SHACL shapes for these profiles
  (`cases.registry.CIM_SHAPES`). PowSyBl and pypowsybl have no validator.
- `counts` (untimed): what the model holds, as cim-bench recorded it.
"""
from abc import abstractmethod
from pathlib import Path
from typing import Any

from adapters.solver_adapter import ToolAdapter


class CimAdapter(ToolAdapter):
    problem = "cim"
    families = ("cim-cgmes",)
    validates = False   # has a `validate`; the benchmark template adds the test only then

    def tags(self) -> list[str]:
        return ["cim", "cgmes"] + (["validator"] if self.validates else []) + [self.language]

    @abstractmethod
    def export(self, model: Any, out_dir: Path) -> None:
        """Write `model` as CGMES RDF/XML into the empty `out_dir`. Timed as `export`."""

    def validate(self, case: str) -> dict:
        """Read the case's files and validate them. Timed as `validate`.
        Returns {"violations": n, "by_severity": {severity: n}}."""
        raise NotImplementedError(f"{self.name} has no validator")

    @abstractmethod
    def counts(self, model: Any) -> dict:
        """{"read": n, "unit": what n counts, "lines", "generators", "loads",
        "substations"}: informational, untimed."""
