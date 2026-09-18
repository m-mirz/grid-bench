"""The interface every tool under test implements.

Modelled on cim-bench's `ParserAdapter`, but for solving rather than
counting. A tool is benchmarked as: `load` a case once (timed as `import`),
then call `solve` repeatedly on that one persistent model (timed as `solve`),
then read `solution` once (untimed) for the oracle.

Every adapter must configure its tool to solve the same problem as every
other: flat start on every solve, a single slack, no reactive limits, no
outer-loop controls (tap changers, phase shifters, switched shunts), and
generator voltage regulation exactly as the case defines it (for CGMES that
includes a remote regulated terminal). Convergence tolerance `TOLERANCE_PU`
on the power mismatch where the tool exposes one. Each adapter's docstring justifies each
setting and says what it deliberately does not do.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, requires, version
from typing import Any

TOLERANCE_PU = 1e-8   # max power mismatch at convergence, per unit
MAX_ITERATIONS = 30


class CaseUnsupported(Exception):
    """The tool cannot read this case at all (e.g. no CGMES importer)."""


class DidNotConverge(Exception):
    """The solver ran but did not converge. Recorded as a result."""


@dataclass
class Solution:
    """Bus voltages keyed by the case's own identifiers, never by array
    position: MATPOWER bus number (as str) for `matpower`, TopologicalNode
    mRID (see `oracle.cgmes_sv.mrid`) for `cgmes`."""
    vm: dict[str, float]        # p.u. for matpower, kV for cgmes
    va_deg: dict[str, float]
    iterations: int | None = None
    extra: dict = field(default_factory=dict)


class SolverAdapter(ABC):
    name: str                   # short id, used in file names and results
    display_name: str
    color: str                  # light-mode hex, a slot of tools/palette.py (fixed per tool)
    package: str                # PyPI distribution, for version capture
    modules: tuple[str, ...]    # every module load/solve imports (memory baseline)
    language: str               # implementation language of the solver core
    families: tuple[str, ...]   # case families this tool can read
    settings: dict = {}         # the solver settings, recorded with every result

    def tags(self) -> list[str]:
        return ["powerflow", "ac", "newton-raphson", self.language]

    def version(self) -> str:
        return version(self.package)

    def dependencies(self) -> dict[str, str]:
        out = {}
        for req in requires(self.package) or []:
            if "extra ==" in req:
                continue
            dep = req.split(";")[0].split("[")[0]
            for sep in "<>=!~ ":
                dep = dep.split(sep)[0]
            try:
                out[dep] = version(dep)
            except PackageNotFoundError:
                pass
        return out

    @abstractmethod
    def load(self, case: str) -> Any:
        """Read the case from disk into the tool's model. Timed as `import`."""

    @abstractmethod
    def solve(self, model: Any) -> None:
        """Solve on the persistent `model`. Timed as `solve`. Raises
        DidNotConverge rather than returning a non-converged state."""

    @abstractmethod
    def solution(self, model: Any, case: str) -> Solution:
        """Voltages after the last `solve`. Untimed."""
