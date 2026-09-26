"""PGLib-OPF's published reference objectives, read from the release's own
`BASELINE.md` (in the `data/benchmark-grids` submodule, with provenance), so
the numbers have one source and are joined to a case by its PGLib name.

They come from PowerModels.jl v0.19.9 and Ipopt 3.14.4 (AC polar): locally
optimal, not proven global optima, and printed to five significant digits,
so they resolve a relative objective difference of about 5e-5 at best.
"""
from functools import lru_cache

from cases.registry import PGLIB_DIR


@lru_cache(maxsize=None)
def reference_objectives() -> dict[str, float]:
    """`{case name: AC objective in $/h}` over every table of `BASELINE.md`.
    The AC column is found by its header, not by position."""
    out = {}
    ac = None
    for line in (PGLIB_DIR / "BASELINE.md").read_text().splitlines():
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if "Case Name" in cells:
            ac = cells.index(r"AC (\$/h)")   # the header as written in BASELINE.md
            continue
        if ac is not None and cells and cells[0].startswith("pglib_opf_"):
            assert cells[0] not in out, f"{cells[0]} listed twice in BASELINE.md"
            out[cells[0]] = float(cells[ac])
    return out


def reference_objective(case: str) -> float:
    return reference_objectives()[case]
