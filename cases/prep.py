"""Derives every tool's input file from the case definitions.

    python -m cases.prep [case ...]

Writes into `data/.case-cache/`:

- `<case>.mat`: the case after `normalize_for_tools`, for pypowsybl,
  pandapower and lightsim2grid (each through its own MATPOWER importer).
- `<case>.zip` (cgmes cases): the profiles a tool is given, zipped, for
  pypowsybl, whose importer reads one file.
- `<case>.pgm.json`: power-grid-model input, converted by
  `gridoxide.matpower.convert` (see cases/pgm_converter.py). power-grid-model
  has no MATPOWER importer; this converter is the one gridoxide's own
  benchmark feeds PGM with. Its known loss: PGM's transformer `clock` cannot
  hold a continuous phase shift, so every MATPOWER phase shift is rounded to
  zero. The oracle reports that as a residual on the shifting branches.

Conversion happens here, not inside a tool's timed import, so a tool's
import time never includes our own conversion code.

A cached file is rebuilt when the source `.m`, this module, `matpower.py`, or
the converter changes: the cache key is a hash of all of them, not
the file's mere existence.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from cases import matpower, pgm_converter
from cases.registry import CACHE, CASES, cgmes_files, mat_path, pgm_json_path


def _cache_key(case: dict) -> str:
    h = hashlib.sha256()
    for p in (case["file"], Path(__file__), Path(matpower.__file__)):
        h.update(Path(p).read_bytes())
    h.update(pgm_converter.SDIST_SHA256.encode())
    return h.hexdigest()


def prepare_cgmes(key: str) -> None:
    out = CACHE / f"{key}.zip"
    files = cgmes_files(key)
    if out.exists() and out.stat().st_mtime >= max(f.stat().st_mtime for f in files):
        return
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    print(f"prepared {key}", file=sys.stderr)


def prepare(key: str) -> None:
    case = CASES[key]
    if case["family"] == "cgmes":
        return prepare_cgmes(key)
    stamp = CACHE / f"{key}.key"
    digest = _cache_key(case)
    if stamp.exists() and stamp.read_text() == digest and mat_path(key).exists() and pgm_json_path(key).exists():
        return
    convert = pgm_converter.converter().convert
    mpc = matpower.normalize_for_tools(matpower.parse_m(case["file"]))
    matpower.write_mat(mpc, mat_path(key))
    convert(mat_path(key), pgm_json_path(key))
    _strip_reactive_limits(pgm_json_path(key))
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


def main(argv: list[str]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for key in argv or list(CASES):
        prepare(key)


if __name__ == "__main__":
    main(sys.argv[1:])
