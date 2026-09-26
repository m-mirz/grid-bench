"""Every benchmark case, in one place. The only case configuration there is.

Five families, two input formats (MATPOWER `.m`, CGMES 3.0):

- `matpower`: meshed transmission cases, MATPOWER `.m` files from the
  `data/benchmark-grids` submodule (see its PROVENANCE.md). The `.m` is the
  case definition; the oracle checks every tool's solution against it.
- `distribution`: radial distribution grids in the same format and graded
  the same way. The fifteen literature feeders MATPOWER bundles (4 to 141
  buses; most as the submodule's `matpower-plain/` copies, whose MATLAB
  unit conversions are evaluated, since no importer runs MATLAB code), and
  synthetic MV/LV grids from power-grid-model's generator (1,004 to 29,840
  buses, the submodule's `generated/`). High R/X, heavy loading (voltages
  down to 0.67 p.u. in the generated grids) and small base powers, where
  transmission cases have none of these.

  Only six run by default; the rest repeat what these show (same structure,
  sizes where timing is call overhead, near-duplicates such as case33mg =
  case33bw at another base power) and stay available by name. case33bw:
  base power 10 and open tie switches; case4_dist: a PV generator, a tap
  and a slack setpoint unlike its bus voltage; case18: two voltage levels
  and shunt capacitors; mvlv1004 / 10616 / 29840: the scaling series.
- `cgmes`: ENTSO-E CGMES 3.0 conformity configurations from the
  `data/CGMES-Test-Configurations` submodule. These ship an SV profile with a
  published solution, used as the reference. The SV profile is never given to
  a tool: it would hand the solver the answer as its starting point.
- `converted-cimoxide`, `converted-pypowsybl`: MATPOWER cases converted to
  CGMES 3.0 by two converters (`cases/matpower_to_cgmes.py` on cimoxide, and
  pypowsybl's MATPOWER import + CGMES export, `cases/convert_pypowsybl.py`).
  Read by the CGMES-capable tools and graded by the tier-1 residual against
  the original `.m` (TopologicalNodes are named `BUS-<n>`). The converters
  themselves are graded tool-free by `oracle.cgmes_model.fidelity`.
  Only cimoxide's output is solved by default: it is exact, so a residual
  there is the reading tool's. pypowsybl's export is not the same problem
  (no slack, Ybus off by up to 3e-4, setpoints missing), so tools on it
  would grade the converter again; it is converted and graded
  (`oracle/check_conversion.py`) but its cases are in no group, available
  by name.

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

Reports are organised by `grid`, the physical grid a case describes:
`transmission` (the `matpower` family and its conversions, which are the
same grids read another way), `distribution` and `fixtures` (the CGMES
conformity fixtures). The family says how a tool reads a case; the grid
says what it is.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
CACHE = DATA / ".case-cache"
MATPOWER_DIR = DATA / "benchmark-grids" / "matpower"
PLAIN_DIR = DATA / "benchmark-grids" / "matpower-plain"
GENERATED_DIR = DATA / "benchmark-grids" / "generated"
CGMES_DIR = DATA / "CGMES-Test-Configurations" / "v3.0"

DEFAULT_GROUPS = ("smoke", "scaling", "feature", "robustness")
CONVERTERS = ("cimoxide", "pypowsybl")
SOLVED_CONVERTERS = ("cimoxide",)  # the others are only graded, see the docstring
FAMILIES = ("matpower", "distribution", "cgmes") + tuple(f"converted-{c}" for c in CONVERTERS)
GRIDS = ("transmission", "distribution", "fixtures")


def _mp(file: str, groups: list[str], source: str, note: str = "") -> dict:
    return {"family": "matpower", "grid": "transmission", "format": "matpower", "file": MATPOWER_DIR / file,
            "groups": groups, "source": source, "note": note}


def _dist(name: str, groups: list[str], source: str, note: str = "", where: Path = PLAIN_DIR) -> dict:
    """`where`: PLAIN_DIR for MATPOWER's feeders that convert units in MATLAB
    code, MATPOWER_DIR for the two that are plain data as shipped."""
    return {"family": "distribution", "grid": "distribution", "format": "matpower", "file": where / f"{name}.m",
            "groups": groups, "source": source, "note": note}


def _cgmes(directory: str, groups: list[str], note: str = "", boundary: str | None = None) -> dict:
    return {"family": "cgmes", "grid": "fixtures", "format": "cgmes", "dir": CGMES_DIR / directory, "groups": groups, "source": "ENTSO-E CGMES 3.0",
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

    "case4_dist": _dist("case4_dist", ["feature"], "MATPOWER", "a PV generator and a tap-changing transformer",
                        MATPOWER_DIR),
    "case12da": _dist("case12da", [], "Das et al."),
    "case15da": _dist("case15da", [], "Das et al."),
    "case15nbr": _dist("case15nbr", [], "Battu et al."),
    "case18": _dist("case18", ["feature"], "Grady et al.", "two voltage levels, shunt capacitors", MATPOWER_DIR),
    "case18nbr": _dist("case18nbr", [], "Battu et al."),
    "case22": _dist("case22", [], "Raju et al."),
    "case28da": _dist("case28da", [], "Das et al."),
    "case33bw": _dist("case33bw", ["smoke", "feature"], "Baran & Wu", "5 open tie switches"),
    "case33mg": _dist("case33mg", [], "Kashem et al.", "5 open tie switches"),
    "case69": _dist("case69", [], "Baran & Wu"),
    "case85": _dist("case85", [], "Das et al."),
    "case118zh": _dist("case118zh", [], "Zhang et al.", "15 open tie switches"),
    "case136ma": _dist("case136ma", [], "Mantovani et al.", "21 open tie switches"),
    "case141": _dist("case141", [], "Khodr et al."),
    "mvlv1004": _dist("mvlv1004", ["scaling"], "power-grid-model generator", "1 LV grid", GENERATED_DIR),
    "mvlv2606": _dist("mvlv2606", [], "power-grid-model generator", "3 LV grids", GENERATED_DIR),
    "mvlv10616": _dist("mvlv10616", ["scaling"], "power-grid-model generator", "13 LV grids", GENERATED_DIR),
    "mvlv29840": _dist("mvlv29840", ["scaling"], "power-grid-model generator", "37 LV grids", GENERATED_DIR),

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


# The headline MATPOWER cases (not robustness), each converted by every converter.
CONVERTED_FROM = [k for k, c in CASES.items() if c["family"] == "matpower" and c["groups"]
                  and c["groups"] != ["robustness"]]
for _base in CONVERTED_FROM:
    for _conv in CONVERTERS:
        CASES[f"{_base}@{_conv}"] = {
            "family": f"converted-{_conv}", "grid": CASES[_base]["grid"], "format": "cgmes",
            "dir": CACHE / f"{_base}@{_conv}",
            "groups": CASES[_base]["groups"] if _conv in SOLVED_CONVERTERS else [],
            "source": f"{_base} via {_conv}", "source_case": _base, "converter": _conv, "boundary": None,
            "note": CASES[_base]["note"],
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


def is_cgmes(key: str) -> bool:
    """Read through a tool's CGMES importer (fixtures and converted cases)."""
    return CASES[key]["format"] == "cgmes"


def cgmes_sv_file(key: str) -> Path:
    return next(CASES[key]["dir"].glob("*_SV*.xml"))


def cgmes_tp_files(key: str) -> list[Path]:
    return [p for p in cgmes_files(key) if "_TP" in p.name]


def mat_path(key: str) -> Path:
    """The normalized `.mat` produced by `cases.prep` (see normalize_for_tools)."""
    return CACHE / f"{key}.mat"


def pgm_json_path(key: str) -> Path:
    return CACHE / f"{key}.pgm.json"
