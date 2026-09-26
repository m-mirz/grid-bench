# AGENTS.md

Contributor guide for grid-bench (people and coding agents alike).

## What this is

A benchmark of power system analysis software. v1 covers AC power flow for
pandapower, lightsim2grid, PyPSA, power-grid-model, pypowsybl, VeraGrid,
Sienna (PowerFlows.jl) and Sparlectra.jl (both Julia, through juliacall),
MATPOWER (GNU Octave), and power-grid-model on CGMES through cgmes2pgm, and
weighted least-squares state estimation for pandapower, power-grid-model,
VeraGrid and Sparlectra.jl. The
infrastructure follows cim-bench (adapters, one container per tool, JSON as
the only contract between measuring and reporting). The methodology follows
gridoxide's `scripts/bench` (warm solves on persistent models, justified
settings, a tool-independent oracle).

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
   `TOLERANCE_PU`. State estimation (`adapters/estimator_adapter.py`): plain
   WLS over the case's measurement set as given, each measurement with its
   own sigma, flat start, the slack angle as the only reference, no
   pseudo-measurements or zero-injection constraints, no bad-data handling,
   tolerance `SE_TOLERANCE` on the state update. Every setting in an adapter
   gets one sentence of justification in its docstring, including what it
   deliberately does not do.
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
cases/       registry.py (every case: groups, family, grid, problem), matpower.py (.m reader), prep.py (tool inputs),
             truth.py (the power flow behind a state-estimation case), measurements.py (its measurement sets),
             gridoxide_matpower.py (vendored MATPOWER->PGM converter, gridoxide 0.0.2),
             matpower_to_cgmes.py (cimoxide converter), convert_pypowsybl.py (pypowsybl converter)
oracle/      ybus.py, residual.py (tier 1), cgmes_sv.py (tier 2), wls.py (state estimation), evaluate.py (entry point),
             cgmes_model.py (tool-free CGMES reader: TN->bus join, converter fidelity),
             check_conversion.py (writes conversion.json)
adapters/    solver_adapter.py (the ABCs), <tool>_adapter.py, estimator_adapter.py + <tool>_se_adapter.py
             (state estimation), cgmes_ids.py, memory.py,
             octave_session.py + matpower_octave/ (MATPOWER's Octave side, timed inside Octave)
benchmarks/  benchmark_template.py (generates tests), conftest.py (selection, failures, metadata),
             <tool>_benchmark.py, <tool>_se_benchmark.py (3 lines each)
tools/       benchmark_data.py (loader, grids, scoreboard) + generate_{comparison,site,all}.py,
             palette.py, check_smoke.py (CI's smoke outcomes)
tool-configs/<tool>/pyproject.toml   dependencies of each image (tools pinned exactly)
tool-configs/matpower/Dockerfile      the official Octave image + uv Python + the MATPOWER release
tool-configs/sienna/Dockerfile, julia/   Julia on top of the base image; Project.toml, Manifest.toml,
             setup.jl (registry snapshot), GridBenchSienna (the adapter's Julia half, precompiled)
tool-configs/sparlectra/Dockerfile, julia/   the same for Sparlectra.jl (GridBenchSparlectra, se.jl: estimation)
docker/      base.dockerfile, tool.dockerfile, docker-compose.yml, build.sh, run_*.sh
tests/       the oracle's own tests (test_wls.py: the state-estimation oracle), and the converter's
             (exactness + planted errors)
data/        submodules: benchmark-grids (MATPOWER), CGMES-Test-Configurations
results-docker/  published results: <tool>.json, <tool>-se.json, comparison.md
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
   `display_name`, `color` (the next unused slot in `tools/palette.py`, kept
   for life; all nine slots are taken, and a tenth tool needs a different
   encoding, not another hue, see the palette's docstring; a reference
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
8. If the tool has a state estimator: `adapters/<tool>_se_adapter.py`
   subclassing `EstimatorAdapter` (identity taken from the power-flow
   adapter), registered in `ESTIMATORS`, and `benchmarks/<tool>_se_benchmark.py`
   (`create_benchmarks("<tool>", "se")`). The run scripts pick it up and
   write `<tool>-se.json`. On `case14~exact` a correct estimator shows J
   around 1e-20 and a step around 1e-15: signs, units and branch ends first.
9. Add the tool's smoke outcomes to `benchmarks/smoke_expectations.json` and
   the tool to the CI matrix (`.github/workflows/smoke.yml`). CI checks each
   smoke case against its known outcome (`tools/check_smoke.py`), including
   expected oracle rejections, so a documented finding is not a CI failure
   but a change in any outcome is. Update an entry only when a behaviour
   change is understood.

## Case families

`matpower` (meshed transmission), `distribution` (radial feeders and
generated MV/LV grids, same `.m` format and grading), `cgmes` (conformity
fixtures, graded against their SV), `converted-<converter>` (MATPOWER
cases converted to CGMES, graded by the tier-1 residual against the original
`.m`), and `se-matpower`, `se-distribution` (state estimation: a case plus
a measurement scenario, `exact` or `noisy`, generated from the case's own
power flow and graded by `oracle.wls`). A tool declares the families it
reads in `SolverAdapter.families`. Converted cases are keyed
`<case>@<converter>`, state-estimation cases `<case>~<scenario>`; a case's
`problem` ("pf" or "se") says which adapter solves it. Branch on a case's input
format with `is_cgmes(case)` (the `format` field), never on its family.

Only exact conversions are solved: `SOLVED_CONVERTERS` in the registry.
pypowsybl's export is converted and graded (`oracle/check_conversion.py`),
but its cases are in no group, because it is not the problem in the `.m`
(no slack, Ybus and setpoints off) and tools on it would only grade the
converter again. They stay runnable by name.

`oracle/cgmes_model.py` must stay independent of both converters: it is
ElementTree only, and must not import cimoxide or pypowsybl. When a converter
and the checker disagree, check the reading of CGMES against a third party
(the pypowsybl export, a tool's importer) before changing either.

## Reports

`tools/generate_all.py` writes `comparison.md` and `docs/index.html`
(whose charts are drawn in the browser from the data embedded in it) from the JSON in a results directory; run it (or the
`reports` compose service) after any change to `tools/`, and commit what it
writes. Conventions both pages share:

- Only cases in the default groups are published (`benchmark_data.load`);
  a case run by name stays in its JSON.
- Tables are split by `grid` (transmission, distribution, fixtures), not by
  family: a converted case is a row under the case it came from, with an
  input column, so what the input route changes is one row apart.
- The scoreboard (`benchmark_data.scoreboard`) is computed once for both
  pages. The `robustness` group is shown as "hard transmission cases".
- In `comparison.md`, notes are grouped by cause: a wrong solution per
  (tool, input), a failure per (tool, message with its numbers dropped).
- The site's chart shows one input at a time (one line per tool needs one
  input); for state estimation, the input is the measurement scenario.

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
sparlectra, pinned to its newest release on purpose, see its `setup.jl`),
the Fuseki jar by SHA-256 (`docker/fuseki/Dockerfile`), GNU Octave by the
official image's digest and the MATPOWER release zip by SHA-256
(`tool-configs/matpower/Dockerfile`), the CI checkout action by commit. Do
not add `apt`/`apk` installs. To update anything, change the pin
deliberately and say why in the commit.

## Style

Flat over nested, functions over classes unless there is state, no
defensive try/except (tools raise, conftest records), no hardcoded tool
names outside adapters and the registry. Comments explain why, not what.
