"""Reader for MATPOWER's plain-text `.m` case format, plus a `.mat` writer.

This is the canonical representation every MATPOWER-family case is defined
by: the oracle builds its Ybus from it, and every tool's input is derived
from the same `.m` file (see each adapter's docstring for the exact path).

Ported from gridoxide's `python/gridoxide/matpower.py::parse_matpower_m`
(Apache-2.0, same author). MATPOWER case files are consistent numeric-matrix
literals (`mpc.bus = [ ...; ...; ];`), not general MATLAB, and this parses
exactly that.
"""
import re
from pathlib import Path

import numpy as np

# MATPOWER column indices (0-based), as documented in MATPOWER's `caseformat`.
BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN = range(13)
GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN = range(10)
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, RATIO, ANGLE, BR_STATUS = range(11)

PQ, PV, REF, ISOLATED = 1, 2, 3, 4


def parse_m(path: Path) -> dict:
    """Returns `{"version", "baseMVA", "bus", "gen", "branch", "gencost"}`."""
    text = Path(path).read_text()
    mpc: dict = {"version": "2"}
    base = re.search(r"mpc\.baseMVA\s*=\s*([\d.eE+-]+)\s*;", text)
    mpc["baseMVA"] = float(base.group(1)) if base else 100.0
    for name in ("bus", "gen", "branch", "gencost"):
        block = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.DOTALL)
        rows = []
        for line in block.group(1).splitlines() if block else []:
            line = line.split("%", 1)[0].strip().rstrip(";").strip()
            if line:
                rows.append([float(x) for x in line.split()])
        # gencost rows may be ragged (piecewise-linear curves); pad with NaN.
        width = max((len(r) for r in rows), default=0)
        mpc[name] = np.array([r + [np.nan] * (width - len(r)) for r in rows]).reshape(len(rows), width)
    return mpc


def normalize_for_tools(mpc: dict) -> dict:
    """Returns a copy with two fields that do not enter the AC power-flow
    equations made importable by every tool:

    - `baseKV == 0` -> 1.0. MATPOWER's per-unit formulation never uses it, but
      tools that build a physical-unit model divide by it (case14 has 0 on
      every bus; pandapower raises FloatingPointError, lightsim2grid rejects
      a 0 kV voltage level).
    - branch `rateA == 0` (MATPOWER: "unlimited") -> 9999 MVA. pandapower
      3.3.3's `from_ppc` uses rateA as the per-unit base of an `impedance`
      element and crashes on 0 (it indexes the wrong array in its own
      zero-guard, `from_ppc.py:303`).

    The oracle always evaluates against the *raw* case, so if either change
    affected a solution it would show up as a residual.
    """
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in mpc.items()}
    out["bus"][out["bus"][:, BASE_KV] == 0, BASE_KV] = 1.0
    out["branch"][out["branch"][:, RATE_A] == 0, RATE_A] = 9999.0
    return out


def write_mat(mpc: dict, path: Path) -> None:
    """Writes a MATPOWER `.mat` (the `mpc` struct), which is what pypowsybl's
    and pandapower's MATPOWER importers read. `version` must be present."""
    import scipy.io

    out = {k: v for k, v in mpc.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
    scipy.io.savemat(str(path), {"mpc": out})


def energized_bus_ids(mpc: dict) -> np.ndarray:
    """Bus numbers of every non-isolated bus, in file order."""
    bus = mpc["bus"]
    return bus[bus[:, BUS_TYPE] != ISOLATED, BUS_I].astype(int)
