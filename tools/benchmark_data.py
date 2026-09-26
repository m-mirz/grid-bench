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
    # State estimation, in the same structure (`tools` holds the estimators'
    # metadata), so every table and chart function works on either.
    se: "Results | None" = None

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


def reads(res: Results, tool: str, case: str) -> bool:
    return CASES[case]["family"] in res.tools[tool]["families"]


def scoreboard(res: Results) -> tuple[list[tuple[str, str | None]], list[tuple[str, list[str]]]]:
    """Per tool, one cell per grid and input: "✓ / ✗ / FAILED" counts of the
    solves, or "solved of all" where there is no verdict (fixtures); then the
    robustness cases, where failing is the expected outcome. `·` where the
    tool reads none of them. Returns the columns as (label, the report
    section that explains it, if any) and (tool, cells) rows, for
    comparison.md and the site."""
    columns = []   # (label, section, cases)
    for grid in res.grids():
        cases = res.grid_cases(grid)
        for label in dict.fromkeys(input_label(c) for c in cases):
            sub = [c for c in cases if input_label(c) == label]
            columns.append((f"{grid} {label}" if all(graded(c) for c in sub) else GRID_TITLES[grid], None, sub))
    robust = [c for g in res.grids() for c in res.grid_cases(g, robustness=True)]
    if robust:
        columns.append(("hard transmission cases", "hard-transmission-cases", robust))
    rows = []
    for t in res.tool_order():
        cells = []
        for _, _, cases in columns:
            mine = [c for c in cases if reads(res, t, c)]
            recs = [res.get(t, c, "solve") for c in mine]
            if not mine:
                cells.append("·")
            elif all(graded(c) for c in mine):
                ok = sum(1 for r in recs if r and r.extra["oracle_ok"])
                bad = sum(1 for r in recs if r and not r.extra["oracle_ok"])
                cells.append(f"{ok} / {bad} / {len(mine) - ok - bad}")
            else:
                cells.append(f"{sum(1 for r in recs if r)} of {len(mine)}")
        rows.append((t, cells))
    return [(label, section) for label, section, _ in columns], rows


def load(directory: Path) -> Results:
    """Every record of a case in the default groups. A case run by name
    (outside them) stays in its JSON but out of the published reports.
    Power flow at the top level, state estimation in `.se`: a tool has an
    entry in each, with that problem's settings."""
    res = Results(se=Results())
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
        for tool, meta in gb["tools"].items():
            part = res.se if meta.get("problem") == "se" else res
            part.tools[tool] = meta
            part.runs.append({"tool": tool, "datetime": data.get("datetime"), "git_sha": gb.get("git_sha"),
                              "image": gb.get("container_image"), "machine": data.get("machine_info", {})})
        for f in data.get("failures", []):
            if _published(f["case"]):
                _part(res, f["case"]).failures.append(f)
        for b in data["benchmarks"]:
            e = b["extra_info"]
            if not _published(e["case"]):
                continue
            _part(res, e["case"]).records.append(Record(e["tool"], e["case"], e["family"], e["operation"],
                                      b["stats"]["median"] * 1e3, b["stats"]["min"] * 1e3, b["stats"]["rounds"], e))
    return res


def _part(res: Results, case: str) -> Results:
    return res.se if CASES[case]["problem"] == "se" else res


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
