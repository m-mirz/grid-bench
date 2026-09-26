"""Loads every result JSON in a directory into one structure. The single
reader all report generators share; none of them imports an adapter.

Record classification uses the explicit `operation` field each record
carries (cim-bench matched substrings of test names, which broke as soon as a
name contained "load").
"""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from cases.matpower import ISOLATED, parse_m
from cases.registry import CASES, DEFAULT_GROUPS, cgmes_sv_file
from oracle.cgmes_sv import published_voltages


@dataclass
class Record:
    tool: str
    case: str
    family: str
    operation: str          # "import" | "solve"
    median_ms: float
    min_ms: float
    rounds: int
    extra: dict


@dataclass
class Results:
    tools: dict = field(default_factory=dict)       # name -> metadata (display_name, color, version, ...)
    records: list = field(default_factory=list)
    failures: list = field(default_factory=list)    # {tool, case, operation, error}
    runs: list = field(default_factory=list)        # {tool, datetime, git_sha, image, machine}
    conversion: list = field(default_factory=list)  # oracle.check_conversion output, one entry per converted case

    def get(self, tool: str, case: str, operation: str) -> Record | None:
        return next((r for r in self.records if (r.tool, r.case, r.operation) == (tool, case, operation)), None)

    def failure(self, tool: str, case: str, operation: str) -> str | None:
        return next((f["error"] for f in self.failures
                     if (f["tool"], f["case"], f["operation"]) == (tool, case, operation)), None)

    def families(self) -> list[str]:
        """Families with results, in registry order."""
        from cases.registry import FAMILIES
        seen = {r.family for r in self.records} | {CASES[f["case"]]["family"] for f in self.failures}
        return [f for f in FAMILIES if f in seen]

    def _seen(self) -> set[str]:
        return {r.case for r in self.records} | {f["case"] for f in self.failures}

    def cases(self, family: str, robustness: bool = False) -> list[str]:
        """Cases with results, by size. Robustness-only cases (expected to fail
        everywhere) are kept out of headline tables unless asked for."""
        return sorted((c for c in self._seen() if CASES[c]["family"] == family
                       and (CASES[c]["groups"] == ["robustness"]) == robustness), key=lambda c: (case_size(c), c))

    def grids(self) -> list[str]:
        """Grids with results, in registry order."""
        from cases.registry import GRIDS
        seen = {CASES[c]["grid"] for c in self._seen()}
        return [g for g in GRIDS if g in seen]

    def grid_cases(self, grid: str, robustness: bool = False) -> list[str]:
        """Cases with results on one grid, by size, each conversion right
        after the case it was converted from."""
        def key(c):
            base = CASES[c].get("source_case", c)
            return case_size(c), base, CASES[c].get("converter", "")
        return sorted((c for c in self._seen() if CASES[c]["grid"] == grid
                       and (CASES[c]["groups"] == ["robustness"]) == robustness), key=key)

    def tool_order(self) -> list[str]:
        """Registry order, i.e. colour-slot order; never re-ranked."""
        from adapters import ADAPTERS  # a name list only; importing it loads no tool
        return [t for t in ADAPTERS if t in self.tools]


FAMILY_TITLES = {
    "matpower": "Transmission grids (MATPOWER .m)",
    "distribution": "Distribution grids (MATPOWER .m)",
    "cgmes": "CGMES conformity fixtures",
    "converted-cimoxide": "Transmission grids as CGMES, converted with cimoxide",
    "converted-pypowsybl": "Transmission grids as CGMES, converted with pypowsybl",
}

GRID_TITLES = {
    "transmission": "Transmission grids (meshed)",
    "distribution": "Distribution grids (radial)",
    "fixtures": "CGMES conformity fixtures",
}


def input_label(case: str) -> str:
    """How a tool reads the case: the `.m`, a converter's CGMES, or a fixture's own CGMES."""
    c = CASES[case]
    if "converter" in c:
        return f"CGMES ({c['converter']})"
    return "CGMES" if c["format"] == "cgmes" else "`.m`"


def graded(case: str) -> bool:
    """Tier 1 applies: there is a `.m` to grade against."""
    c = CASES[case]
    return c["format"] == "matpower" or "source_case" in c


def load(directory: Path) -> Results:
    """Every record of a case in the default groups. A case run by name
    (outside them) stays in its JSON but out of the published reports."""
    res = Results()
    conversion = Path(directory) / "conversion.json"
    if conversion.exists():
        res.conversion = json.loads(conversion.read_text())
    for path in sorted(Path(directory).glob("*.json")):
        if path.name == "conversion.json":
            continue
        data = json.loads(path.read_text())
        if "grid_bench" not in data:
            continue
        gb = data["grid_bench"]
        res.tools.update(gb["tools"])
        res.failures.extend(f for f in data.get("failures", []) if _published(f["case"]))
        for tool in gb["tools"]:
            res.runs.append({"tool": tool, "datetime": data.get("datetime"), "git_sha": gb.get("git_sha"),
                             "image": gb.get("container_image"), "machine": data.get("machine_info", {})})
        for b in data["benchmarks"]:
            e = b["extra_info"]
            if not _published(e["case"]):
                continue
            res.records.append(Record(e["tool"], e["case"], e["family"], e["operation"],
                                      b["stats"]["median"] * 1e3, b["stats"]["min"] * 1e3, b["stats"]["rounds"], e))
    return res


def _published(case: str) -> bool:
    return bool(set(CASES[case]["groups"]) & set(DEFAULT_GROUPS))


@lru_cache(maxsize=None)
def case_size(case: str) -> int:
    """Energized buses for MATPOWER (and cases converted from it), published
    TopologicalNodes for CGMES fixtures."""
    c = CASES[case]
    if "source_case" in c:
        return case_size(c["source_case"])
    if c["format"] == "matpower":
        return int((parse_m(c["file"])["bus"][:, 1] != ISOLATED).sum())
    return len(published_voltages(cgmes_sv_file(case)))


def load_solution(directory: Path, tool: str, case: str) -> dict | None:
    path = Path(directory) / "solutions" / tool / f"{case}.json"
    return json.loads(path.read_text()) if path.exists() else None
