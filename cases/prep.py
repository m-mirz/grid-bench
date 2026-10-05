"""Derives every tool's input file from the case definitions.

    python -m cases.prep [case ...]

Writes into `data/.case-cache/`:

- `<case>.mat`: the case after `normalize_for_tools`, for pypowsybl,
  pandapower and lightsim2grid (each through its own MATPOWER importer).
- `<case>.zip` (cgmes cases): the profiles a tool is given, zipped, for
  pypowsybl, whose importer reads one file.
- `<case>@<converter>/` and `.zip`: the case converted to CGMES 3.0 by each
  converter (`cases/matpower_to_cgmes.py` on cimoxide, `cases/convert_pypowsybl.py`).
  A converter whose library is not installed is skipped, so the harness
  image converts with cimoxide and the pypowsybl image
  (`python -m cases.prep --family converted-pypowsybl`) with pypowsybl.
- `<case>.pgm.json`: power-grid-model input, converted by
  `gridoxide.matpower.convert` (vendored: cases/gridoxide_matpower.py). power-grid-model
  has no MATPOWER importer; this converter is the one gridoxide's own
  benchmark feeds PGM with. Its known loss: PGM's transformer `clock` cannot
  hold a continuous phase shift, so every MATPOWER phase shift is rounded to
  zero. The oracle reports that as a residual on the shifting branches.
  The slack's source is made the `.m`'s slack: ideal (`PGM_SOURCE_SK`), at
  its generators' `Vg`.
  Next to it, `<case>.pgm.branch-ids.json`: the PGM id of each branch row,
  the join for PGM's branch sensors in state estimation.
- `<case>~<scenario>.meas.json` (state-estimation cases): the measurement
  set (`cases.measurements`), generated from the case's own power flow
  (`cases.truth`), which is first checked by the oracle's tier-1 residual.
- `<case>#sweep.sweep.npz` (batch cases): the operating points
  (`cases.sweep`), every one solved by `cases.truth` and checked by tier 1.
- `<case>#n1.n1.npz` (contingency cases): the outages (`cases.contingency`),
  every one solved by `cases.truth` from the base case's solution and
  checked by tier 1.

Conversion happens here, not inside a tool's timed import, so a tool's
import time never includes our own conversion code.

A cached file is rebuilt when the source `.m`, this module, `matpower.py`, or
the converter changes: the cache key is a hash of all of them, not
the file's mere existence.
"""
import hashlib
import json
import shutil
import sys
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from cases import contingency, gridoxide_matpower, matpower, measurements, sweep, truth
from cases.registry import (CACHE, CASES, FAMILIES, cgmes_files, contingency_path, mat_path, measurements_path,
                            pgm_branch_ids_path, pgm_json_path, sweep_path)
from oracle import residual, ybus


def _cache_key(case: dict) -> str:
    h = hashlib.sha256()
    for p in (case["file"], Path(__file__), Path(matpower.__file__), Path(gridoxide_matpower.__file__)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def _zip(key: str) -> None:
    with zipfile.ZipFile(CACHE / f"{key}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in cgmes_files(key):
            z.write(f, f.name)


def prepare_cgmes(key: str) -> None:
    out = CACHE / f"{key}.zip"
    if out.exists() and out.stat().st_mtime >= max(f.stat().st_mtime for f in cgmes_files(key)):
        return
    _zip(key)
    print(f"prepared {key}", file=sys.stderr)


def prepare_converted(key: str) -> None:
    case = CASES[key]
    base, conv = case["source_case"], case["converter"]
    module = {"cimoxide": "cases.matpower_to_cgmes", "pypowsybl": "cases.convert_pypowsybl"}[conv]
    try:
        lib_version = version(conv)
    except PackageNotFoundError:
        print(f"skipped {key}: {conv} not installed here", file=sys.stderr)
        return
    stamp = CACHE / f"{key}.key"
    h = hashlib.sha256()
    for p in (CASES[base]["file"], Path(__file__), Path(matpower.__file__),
              Path(__file__).parent / f"{module.split('.')[-1]}.py"):
        h.update(Path(p).read_bytes())
    h.update(lib_version.encode())
    if stamp.exists() and stamp.read_text() == h.hexdigest() and (CACHE / f"{key}.zip").exists():
        return
    prepare(base)
    shutil.rmtree(case["dir"], ignore_errors=True)
    if conv == "cimoxide":
        from cases import matpower_to_cgmes
        matpower_to_cgmes.write(matpower_to_cgmes.convert(matpower.parse_m(CASES[base]["file"]), base),
                                case["dir"], base)
    else:
        from cases import convert_pypowsybl
        convert_pypowsybl.convert(mat_path(base), case["dir"])
    _zip(key)
    stamp.write_text(h.hexdigest())
    print(f"prepared {key}", file=sys.stderr)


def prepare_se(key: str) -> None:
    """The measurement set, after the base case's tool inputs (tools read the
    network from those and attach the measurements)."""
    case = CASES[key]
    prepare(case["base_case"])
    stamp = CACHE / f"{key}.key"
    h = hashlib.sha256()
    for p in (case["file"], Path(__file__), Path(matpower.__file__), Path(truth.__file__),
              Path(measurements.__file__), Path(residual.__file__), Path(ybus.__file__)):
        h.update(Path(p).read_bytes())
    if stamp.exists() and stamp.read_text() == h.hexdigest() and measurements_path(key).exists():
        return
    mpc = matpower.parse_m(case["file"])
    data = measurements.generate(mpc, key, case["scenario"])
    vm = {b: t[0] for b, t in data["true_state"].items()}
    va = {b: t[1] for b, t in data["true_state"].items()}
    r = residual.residual(mpc, vm, va)
    assert max(r.max_dp_mw, r.max_dq_mvar) < 1e-6 and r.max_dvm_pu < 1e-9 and r.n_checked == r.n_buses, \
        f"{key}: the true state does not solve the case ({r})"
    measurements.write(data, measurements_path(key))
    stamp.write_text(h.hexdigest())
    print(f"prepared {key}", file=sys.stderr)


def prepare_sweep(key: str) -> None:
    """The scenarios, after the base case's tool inputs (tools read the
    network from those and apply each scenario to it)."""
    case = CASES[key]
    prepare(case["base_case"])
    stamp = CACHE / f"{key}.key"
    h = hashlib.sha256()
    for p in (case["file"], Path(__file__), Path(matpower.__file__), Path(truth.__file__),
              Path(sweep.__file__), Path(residual.__file__), Path(ybus.__file__)):
        h.update(Path(p).read_bytes())
    if stamp.exists() and stamp.read_text() == h.hexdigest() and sweep_path(key).exists():
        return
    mpc = matpower.parse_m(case["file"])
    data = sweep.generate(mpc, key)
    for k in range(len(data["scale"])):
        mpc_k = sweep.scenario(mpc, data, k)
        ids, v = truth.solve_pf(mpc_k)
        on = ~np.isnan(v)
        vm = {str(b): float(abs(x)) for b, x in zip(ids[on], v[on])}
        va = {str(b): float(np.rad2deg(np.angle(x))) for b, x in zip(ids[on], v[on])}
        r = residual.residual(mpc_k, vm, va)
        assert max(r.max_dp_mw, r.max_dq_mvar) < 1e-6 and r.max_dvm_pu < 1e-9 and r.n_checked == r.n_buses, \
            f"{key}: scenario {k} has no solution from flat start ({r})"
    sweep.write(data, sweep_path(key))
    stamp.write_text(h.hexdigest())
    print(f"prepared {key}", file=sys.stderr)


def prepare_n1(key: str) -> None:
    """The outages, after the base case's tool inputs."""
    case = CASES[key]
    prepare(case["base_case"])
    stamp = CACHE / f"{key}.key"
    h = hashlib.sha256()
    for p in (case["file"], Path(__file__), Path(matpower.__file__), Path(truth.__file__),
              Path(contingency.__file__), Path(residual.__file__), Path(ybus.__file__)):
        h.update(Path(p).read_bytes())
    if stamp.exists() and stamp.read_text() == h.hexdigest() and contingency_path(key).exists():
        return
    mpc = matpower.parse_m(case["file"])
    _, v_base = truth.solve_pf(mpc)
    rows, skipped = [], 0
    for row in contingency.candidates(mpc, key):
        mpc_k = contingency.outage(mpc, {"branch_row": [row]}, 0)
        try:
            ids, v = truth.solve_pf(mpc_k, v_base)
        except RuntimeError:
            skipped += 1
            continue
        on = ~np.isnan(v)
        r = residual.residual(mpc_k, {str(b): float(abs(x)) for b, x in zip(ids[on], v[on])},
                              {str(b): float(np.rad2deg(np.angle(x))) for b, x in zip(ids[on], v[on])})
        assert max(r.max_dp_mw, r.max_dq_mvar) < 1e-6 and r.max_dvm_pu < 1e-9 and r.n_checked == r.n_buses, \
            f"{key}: outage of branch row {row} solved, but not to the case ({r})"
        rows.append(row)
        if len(rows) == contingency.N_OUTAGES:
            break
    branch = mpc["branch"]
    contingency.write({"branch_row": np.array(rows, dtype=np.int64),
                       "from_bus": branch[rows, matpower.F_BUS].astype(np.int64),
                       "to_bus": branch[rows, matpower.T_BUS].astype(np.int64), "n_skipped": np.int64(skipped)},
                      contingency_path(key))
    stamp.write_text(h.hexdigest())
    print(f"prepared {key} ({len(rows)} outages, {skipped} skipped: no convergence)", file=sys.stderr)


def prepare(key: str) -> None:
    case = CASES[key]
    if case["family"] == "cgmes":
        return prepare_cgmes(key)
    if "converter" in case:
        return prepare_converted(key)
    if case["problem"] == "se":
        return prepare_se(key)
    if case["problem"] == "batch":
        return prepare_sweep(key)
    if case["problem"] == "n1":
        return prepare_n1(key)
    stamp = CACHE / f"{key}.key"
    digest = _cache_key(case)
    if (stamp.exists() and stamp.read_text() == digest and mat_path(key).exists() and pgm_json_path(key).exists()
            and pgm_branch_ids_path(key).exists()):
        return
    convert = gridoxide_matpower.convert
    mpc = matpower.normalize_for_tools(matpower.parse_m(case["file"]))
    matpower.write_mat(mpc, mat_path(key))
    branch_ids = convert(mat_path(key), pgm_json_path(key))
    pgm_branch_ids_path(key).write_text(json.dumps(branch_ids))
    _strip_reactive_limits(pgm_json_path(key))
    _ideal_source(pgm_json_path(key), mpc)
    stamp.write_text(digest)
    print(f"prepared {key}", file=sys.stderr)


def _strip_reactive_limits(path: Path) -> None:
    """The converter writes each generator's Qmin/Qmax onto its
    `voltage_regulator`, which makes PGM enforce reactive limits. Every other
    tool runs with limits off, so they are removed here."""
    data = json.loads(path.read_text())
    for vr in data["data"].get("voltage_regulator", []):
        vr.pop("q_min", None)
        vr.pop("q_max", None)
    path.write_text(json.dumps(data))


# The slack is a PGM `source`, an ideal voltage behind an impedance of
# u_ref^2 / sk. The converter's sk = 1e10 VA leaves the slack 1e-3 to 0.4
# p.u. off its setpoint, a different problem from MATPOWER's ideal slack
# bus. The error falls as 1/sk; from 1e30 on it is 2e-16 p.u. on every
# case, and convergence and every other residual are unchanged up to 1e40.
PGM_SOURCE_SK = 1e40


def _ideal_source(path: Path, mpc: dict) -> None:
    """Makes every source the `.m`'s slack: short-circuit power
    PGM_SOURCE_SK, and `u_ref` the slack generators' `Vg`. The converter
    takes `u_ref` from the bus's `Vm` column, which MATPOWER's power flow
    never reads at a generator bus (`Vg` is the setpoint), so this changes
    no equation of the `.m`; case4_dist and case18 (Vm 1, Vg 1.05) and
    case3120sp (Vg 1.04) were solved at the wrong slack voltage."""
    gen = mpc["gen"][mpc["gen"][:, matpower.GEN_STATUS] > 0]
    data = json.loads(path.read_text())
    for source in data["data"]["source"]:
        vg = np.unique(gen[gen[:, matpower.GEN_BUS] == source["node"], matpower.VG])
        assert len(vg) == 1, f"slack bus {source['node']}: online generators' Vg {vg}"
        source["sk"] = PGM_SOURCE_SK
        source["u_ref"] = float(vg[0])
    path.write_text(json.dumps(data))


def main(argv: list[str]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if argv[:1] == ["--family"]:
        keys = [k for k, c in CASES.items() if c["family"] == argv[1]]
    else:
        keys = argv or [k for fam in FAMILIES for k, c in CASES.items() if c["family"] == fam]
    for key in keys:
        prepare(key)


if __name__ == "__main__":
    main(sys.argv[1:])
