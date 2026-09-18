"""Every benchmark case, in one place. The only case configuration there is.

Two families:

- `matpower`: MATPOWER `.m` files from the `data/benchmark-grids` submodule
  (see its PROVENANCE.md). The `.m` is the case definition; the oracle checks
  every tool's solution against it.
- `cgmes`: ENTSO-E CGMES 3.0 conformity configurations from the
  `data/CGMES-Test-Configurations` submodule. These ship an SV profile with a
  published solution, used as the reference. The SV profile is never given to
  a tool: it would hand the solver the answer as its starting point.

Groups say what a case is *for* (a case can be in several):

- `smoke`: trivial; proves the pipeline end to end. Timing is call overhead.
- `scaling`: one family (PEGASE) at increasing size, plus the IEEE mid-size
  cases. The timing headline. Below ~1000 buses, timing mostly measures
  Python call overhead rather than the solver.
- `feature`: exercises modelling details that separate importers (transformer
  line charging, negative reactance, PV buses without generators, offline
  equipment, every branch encoded as a transformer). The accuracy headline.
- `robustness`: known not to converge from a flat start in any tool tested;
  reported separately, never in the timing headline.

A case with no group stays available by name but is not run by default.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
CACHE = DATA / ".case-cache"
MATPOWER_DIR = DATA / "benchmark-grids" / "matpower"
CGMES_DIR = DATA / "CGMES-Test-Configurations" / "v3.0"

DEFAULT_GROUPS = ("smoke", "scaling", "feature", "robustness")


def _mp(file: str, groups: list[str], source: str, note: str = "") -> dict:
    return {"family": "matpower", "file": MATPOWER_DIR / file, "groups": groups, "source": source, "note": note}


def _cgmes(directory: str, groups: list[str], note: str = "", boundary: str | None = None) -> dict:
    return {"family": "cgmes", "dir": CGMES_DIR / directory, "groups": groups, "source": "ENTSO-E CGMES 3.0",
            "boundary": CGMES_DIR / boundary if boundary else None, "note": note}


CASES: dict[str, dict] = {
    "case14": _mp("case14.m", ["smoke"], "IEEE"),
    "case118": _mp("case118.m", ["scaling"], "IEEE"),
    "case_illinois200": _mp("case_ACTIVSg200.m", [], "TAMU synthetic",
                            "no taps or shifters; adds little beyond case118/case300"),
    "case300": _mp("case300.m", ["scaling", "feature"], "IEEE",
                   "13 kV levels, a negative reactance, 18 transformers with line charging"),
    "case1354pegase": _mp("case1354pegase.m", ["scaling"], "PEGASE"),
    "case2869pegase": _mp("case2869pegase.m", ["scaling"], "PEGASE"),
    "case9241pegase": _mp("case9241pegase.m", ["scaling"], "PEGASE", "66 phase shifters, 16 negative reactances"),
    "case3120sp": _mp("case3120sp.m", ["feature"], "Polish summer peak",
                      "101 PV buses with no online generator, 207 offline generators"),
    "case2848rte": _mp("case2848rte.m", ["feature"], "RTE",
                       "every branch encoded as a transformer; the one RTE case that converges from flat start"),
    "case1888rte": _mp("case1888rte.m", ["robustness"], "RTE"),
    "case6495rte": _mp("case6495rte.m", ["robustness"], "RTE"),
    "case6515rte": _mp("case6515rte.m", [], "RTE", "same grid and structure as case6495rte, another snapshot"),

    "cgmes_powerflow": _cgmes("PowerFlow/PowerFlow", ["smoke"]),
    "cgmes_microgrid_be": _cgmes("MicroGrid/MicroGid-BaseCase/MicroGrid-BE-MAS", ["feature"],
                                 boundary="MicroGrid/MicroGid-BaseCase/MicroGrid-BD-MAS/20171002T0930Z_ENTSO-E_EQ_BD_2.xml"),
    "cgmes_minigrid": _cgmes("MiniGrid/MiniGrid-Merged", ["feature"]),
    "cgmes_smallgrid": _cgmes("SmallGrid/SmallGrid-Merged", ["scaling"]),
    "cgmes_svedala": _cgmes("Svedala/Svedala-Merged", ["scaling"]),
    "cgmes_realgrid": _cgmes("RealGrid/RealGrid-Merged", ["scaling"],
                             "published SV is itself inconsistent (transformers miss their own nameplate ratio); "
                             "large SV deviations here are a data defect, see gridoxide scripts/bench/README.md"),
}


def select(family: str, groups=DEFAULT_GROUPS, names=None) -> list[str]:
    """Case keys of one family, filtered by explicit names or by group."""
    keys = [k for k, c in CASES.items() if c["family"] == family]
    if names:
        return [k for k in keys if k in names]
    return [k for k in keys if set(CASES[k]["groups"]) & set(groups)]


def cgmes_files(key: str) -> list[Path]:
    """Every profile a tool is given: all XML in the case directory except SV
    (the reference solution) and the non-power-flow profiles, plus the
    boundary set if the case needs one."""
    case = CASES[key]
    skip = ("_SV", "_DL", "_GL", "_DY", "_OP", "_SC")
    files = sorted(p for p in case["dir"].glob("*.xml") if not any(s in p.name for s in skip))
    return files + ([case["boundary"]] if case["boundary"] else [])


def cgmes_sv_file(key: str) -> Path:
    return next(CASES[key]["dir"].glob("*_SV*.xml"))


def cgmes_tp_files(key: str) -> list[Path]:
    return [p for p in cgmes_files(key) if "_TP" in p.name]


def mat_path(key: str) -> Path:
    """The normalized `.mat` produced by `cases.prep` (see normalize_for_tools)."""
    return CACHE / f"{key}.mat"


def pgm_json_path(key: str) -> Path:
    return CACHE / f"{key}.pgm.json"
