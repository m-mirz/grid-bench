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
from cases.registry import CASES, cgmes_sv_file
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

    def get(self, tool: str, case: str, operation: str) -> Record | None:
        return next((r for r in self.records if (r.tool, r.case, r.operation) == (tool, case, operation)), None)

    def failure(self, tool: str, case: str, operation: str) -> str | None:
        return next((f["error"] for f in self.failures
                     if (f["tool"], f["case"], f["operation"]) == (tool, case, operation)), None)

    def cases(self, family: str, robustness: bool = False) -> list[str]:
        """Cases with results, by size. Robustness-only cases (expected to fail
        everywhere) are kept out of headline tables unless asked for."""
        seen = {r.case for r in self.records} | {f["case"] for f in self.failures}
        return sorted((c for c in seen if CASES[c]["family"] == family
                       and (CASES[c]["groups"] == ["robustness"]) == robustness), key=lambda c: (case_size(c), c))

    def tool_order(self) -> list[str]:
        """Registry order, i.e. colour-slot order; never re-ranked."""
        from adapters import ADAPTERS  # a name list only; importing it loads no tool
        return [t for t in ADAPTERS if t in self.tools]


def load(directory: Path) -> Results:
    res = Results()
    for path in sorted(Path(directory).glob("*.json")):
        data = json.loads(path.read_text())
        if "grid_bench" not in data:
            continue
        gb = data["grid_bench"]
        res.tools.update(gb["tools"])
        res.failures.extend(data.get("failures", []))
        for tool in gb["tools"]:
            res.runs.append({"tool": tool, "datetime": data.get("datetime"), "git_sha": gb.get("git_sha"),
                             "image": gb.get("container_image"), "machine": data.get("machine_info", {})})
        for b in data["benchmarks"]:
            e = b["extra_info"]
            res.records.append(Record(e["tool"], e["case"], e["family"], e["operation"],
                                      b["stats"]["median"] * 1e3, b["stats"]["min"] * 1e3, b["stats"]["rounds"], e))
    return res


@lru_cache(maxsize=None)
def case_size(case: str) -> int:
    """Energized buses for MATPOWER, published TopologicalNodes for CGMES."""
    c = CASES[case]
    if c["family"] == "matpower":
        return int((parse_m(c["file"])["bus"][:, 1] != ISOLATED).sum())
    return len(published_voltages(cgmes_sv_file(case)))


def load_solution(directory: Path, tool: str, case: str) -> dict | None:
    path = Path(directory) / "solutions" / tool / f"{case}.json"
    return json.loads(path.read_text()) if path.exists() else None
