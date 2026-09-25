# CLAUDE.md

Contributor guide for grid-bench (people and coding agents alike).

## What this is

A benchmark of power system analysis software. v1 covers AC power flow for
pandapower, lightsim2grid, PyPSA, power-grid-model, pypowsybl, VeraGrid,
Sienna (PowerFlows.jl, Julia, through juliacall), Sparlectra.jl (Julia,
through juliacall), MATPOWER (GNU Octave), and
power-grid-model on CGMES through cgmes2pgm. The infrastructure follows cim-bench (adapters, one
container per tool, JSON as the only contract between measuring and
reporting). The
methodology follows gridoxide's `scripts/bench` (warm solves on persistent
models, justified settings, a tool-independent oracle).

## Rules that are not negotiable

1. **No speed number without a correctness number.** Every solve record
   carries the oracle's verdict. Never add a timing path that skips
   `oracle.evaluate`.
2. **The oracle imports no tool.** `oracle/` depends on numpy/scipy only.
   Anything that needs a tool object goes in `adapters/`. In `cases/`, only
   the converters (`matpower_to_cgmes.py`, `convert_pypowsybl.py`) import a
   library, lazily; the rest is numpy/scipy.
3. **Same problem for every tool** (see `adapters/solver_adapter.py`): flat
   start on every solve, single slack, no reactive limits, no outer-loop
   controls, generator voltage regulation as the case defines it, tolerance
   `TOLERANCE_PU`. Every setting in an adapter gets one sentence of
   justification in its docstring, including what it deliberately does not do.
4. **Report, don't fix.** When a tool's importer changes the problem (the
   oracle shows a residual), document it in the adapter docstring and leave it.
   Only change input data if the change provably leaves the power-flow
   equations untouched (`cases.matpower.normalize_for_tools`), and the oracle
   keeps using the raw case so that claim stays checked.
5. **Exact joins only.** Buses are matched to case identifiers by an explicit
   relationship (MATPOWER number, TP-profile terminal or connectivity node),
   never by nearest value, and never by array position without an assertion.
6. **Failures are data.** A tool raises; conftest.py records the real
   exception. Never catch-and-skip in an adapter.
7. **Generators never import adapters.** Reporting reads JSON only (the
   `adapters.ADAPTERS` name list is the one exception; it loads no tool).

## Layout

```
cases/       registry.py (every case, groups, families), matpower.py (.m reader), prep.py (tool inputs),
             gridoxide_matpower.py (vendored MATPOWER->PGM converter, gridoxide 0.0.2 + one fix),
             matpower_to_cgmes.py (cimoxide converter), convert_pypowsybl.py (pypowsybl converter)
oracle/      ybus.py, residual.py (tier 1), cgmes_sv.py (tier 2), evaluate.py (entry point),
             cgmes_model.py (tool-free CGMES reader: TN->bus join, converter fidelity),
             check_conversion.py (writes conversion.json)
adapters/    solver_adapter.py (the ABC), <tool>_adapter.py, cgmes_ids.py, memory.py,
             octave_session.py + matpower_octave/ (MATPOWER's Octave side, timed inside Octave)
benchmarks/  benchmark_template.py (generates tests), conftest.py (selection, failures, metadata),
             <tool>_benchmark.py (3 lines each)
tools/       benchmark_data.py (loader) + generate_{comparison,graphs,site,all}.py, palette.py
tool-configs/<tool>/pyproject.toml   dependencies of each image (tools pinned exactly)
tool-configs/matpower/Dockerfile      the official Octave image + uv Python + the MATPOWER release
tool-configs/sienna/Dockerfile, julia/   Julia on top of the base image; Project.toml, Manifest.toml,
             setup.jl (registry snapshot), GridBenchSienna (the adapter's Julia half, precompiled)
tool-configs/sparlectra/Dockerfile, julia/   the same for Sparlectra.jl (GridBenchSparlectra)
docker/      base.dockerfile, tool.dockerfile, docker-compose.yml, build.sh, run_*.sh
tests/       the oracle's own tests, and the converter's (exactness + planted errors)
data/        submodules: benchmark-grids (MATPOWER), CGMES-Test-Configurations
results-docker/  published results: <tool>.json, comparison.md, graphs/
docs/index.html  generated site
```

## Commands

```bash
docker/build.sh [tool ...]                     # images (base, harness, tools)
docker/lock.sh                                 # re-resolve every uv.lock and Julia Manifest.toml (review, then commit)
docker/run_benchmark.sh [tool ...] [-- --groups smoke]   # prep, oracle tests, tools, reports
docker/run_single.sh pandapower --cases case14,case300   # one tool, quick iteration
docker compose -f docker/docker-compose.yml run --rm reports   # regenerate reports only

./setup.sh && ./run_benchmarks.sh [tool ...]   # native, one environment, for development
uv run pytest tests                            # oracle tests
uv run python -m cases.prep [case ...]         # tool inputs into data/.case-cache/
```

Source is mounted into containers read-only, so editing an adapter needs no
rebuild; changing `tool-configs/` does. Tool containers have no network,
except cgmes2pgm, which reaches its Fuseki sidecar (`docker/fuseki/`) on an
internal compose network; the run scripts stop the sidecar afterwards.

## Adding a tool

1. `adapters/<tool>_adapter.py`: subclass `SolverAdapter`. Set `name`,
   `display_name`, `color` (the next unused slot in `tools/palette.py`; a
   tool keeps its colour for life; all nine are taken (the ninth past the
   validated eight), and a reference
   implementation uses `REFERENCE`, drawn dashed), `package`, `modules`
   (everything `load` and `solve` import, for the memory baseline), `language`,
   `families`, `settings`. Implement `load`, `solve`, `solution`. Docstring:
   input path, bus-id mapping, and every setting with its justification.
2. Register it in `adapters/__init__.py` (order = colour-slot order).
3. `benchmarks/<tool>_benchmark.py`: three lines, copy another.
4. `tool-configs/<tool>/pyproject.toml`: copy another, pin the tool exactly
   (a release at least 7 days old; `exclude-newer = "P7D"` enforces it).
   Then `docker/lock.sh` to write its `uv.lock`, and commit both: images
   install exactly the lockfile.
5. A service in `docker/docker-compose.yml`, copy another. A tool that
   needs more than Python packages brings `tool-configs/<tool>/Dockerfile`
   (built by `docker/build.sh` in place of `docker/tool.dockerfile`; see
   sienna's, which adds Julia).
6. `docker/build.sh <tool> && docker/run_single.sh <tool> --groups smoke`.
   Then check the oracle: on case14 a correct tool shows residuals around
   1e-9 MVA. If it does not, find out why before anything else.
7. `docker/run_single.sh <tool>` for all default cases, then regenerate reports.
8. Add the tool's smoke outcomes to `benchmarks/smoke_expectations.json` and
   the tool to the CI matrix (`.github/workflows/smoke.yml`). CI checks each
   smoke case against its known outcome (`tools/check_smoke.py`), including
   expected oracle rejections, so a documented finding is not a CI failure
   but a change in any outcome is. Update an entry only when a behaviour
   change is understood.

## Case families

`matpower` (meshed transmission), `distribution` (radial feeders and
generated MV/LV grids, same `.m` format and grading), `cgmes` (conformity
fixtures, graded against their SV), and `converted-<converter>` (MATPOWER
cases converted to CGMES, graded by the tier-1 residual against the original
`.m`). A tool declares the families it reads in `SolverAdapter.families`.
Converted cases are keyed `<case>@<converter>`. Branch on a case's input
format with `is_cgmes(case)` (the `format` field), never on its family.

`oracle/cgmes_model.py` must stay independent of both converters: it is
ElementTree only, and must not import cimoxide or pypowsybl. When a converter
and the checker disagree, check the reading of CGMES against a third party
(the pypowsybl export, a tool's importer) before changing either.

## Adding a case

Add it to `CASES` in `cases/registry.py` with its groups. MATPOWER cases
come from the `data/benchmark-grids` submodule, and new data goes there
first (with its provenance), never directly into this repo. A `.m` file
must be plain data: MATPOWER's distribution files convert units in MATLAB
code that no importer runs, so the registry reads the submodule's
`matpower-plain/` copies (`tests/test_cases.py` guards this). CGMES cases
need an SV profile, which is used as the reference and never given to a tool.

## Pinning

Everything a build fetches is pinned; keep it that way. Base images and the
uv image by digest, CPython by exact version (`docker/base.dockerfile`),
Python packages by the committed `tool-configs/*/uv.lock` (and `uv.lock` for
the native path), Julia by the official image's digest and Julia packages by
`tool-configs/*/julia/Manifest.toml`, resolved against the General
registry at a commit at least 7 days old (`REGISTRY_COMMIT` in `setup.jl`;
move it forward deliberately, like `exclude-newer`; the one exception is
sparlectra, pinned to its newest release on purpose, see its `setup.jl`), the Fuseki jar by
SHA-256 (`docker/fuseki/Dockerfile`), GNU Octave by the official image's
digest and the MATPOWER release zip by SHA-256
(`tool-configs/matpower/Dockerfile`), the CI checkout action by commit. Do not add `apt`/`apk` installs. To
update anything, change the pin deliberately and say why in the commit.

## Style

Flat over nested, functions over classes unless there is state, no
defensive try/except (tools raise, conftest records), no hardcoded tool
names outside adapters and the registry. Comments explain why, not what.
