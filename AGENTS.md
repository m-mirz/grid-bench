# AGENTS.md

Contributor guide for grid-bench (people and coding agents alike).

## What this is

A benchmark of power system analysis software. v1 covers AC power flow for
pandapower, p3s (pandapower's parallel solver, C++/KLU), lightsim2grid, PyPSA,
power-grid-model, pypowsybl, VeraGrid, Sienna (PowerFlows.jl), ExaPF.jl (on
its CPU backend, and on its CUDA backend as `exapf_gpu`, which needs an
NVIDIA GPU) and Sparlectra.jl (all Julia, through juliacall), MATPOWER (GNU Octave),
gpusim2grid (lightsim2grid's GPU companion, CUDA and cuDSS, which needs an
NVIDIA GPU), and power-grid-model on CGMES through cgmes2pgm, and
weighted least-squares state estimation for pandapower, power-grid-model,
VeraGrid and Sparlectra.jl, and AC optimal power flow for MATPOWER,
pandapower, VeraGrid and PowerModels.jl (Ipopt; its own image, OPF only) on
PGLib-OPF cases, and batch power flow (a sweep of operating points per case)
for every tool that reads `.m`: through the tool's own batch API at 1..n
threads (power-grid-model, p3s, lightsim2grid) or on one thread (PyPSA,
VeraGrid, Sienna, ExaPF.jl), or on one GPU (ExaPF.jl's CUDA backend; gpusim2grid,
starting every scenario from the base case's solution, a base-case-start
variant never ranked with the flat-start tools), or a loop of single solves (pandapower, pypowsybl,
Sparlectra.jl, MATPOWER), and N-1 contingency analysis for the same tools
(200 branch outages per transmission case, through a native contingency API
where there is one), and CIM libraries reading, writing and validating CGMES
(import, export, validate on the Svedala and RealGrid conformity models) for
cimoxide, triplets, OpenCGMES, PowSyBl's Java CgmesModel (the last two
through JPype) and pypowsybl, timed only, as in cim-bench. The
infrastructure follows cim-bench (adapters, one container per tool, JSON as
the only contract between measuring and reporting). The methodology follows
gridoxide's `scripts/bench` (warm solves on persistent models, justified
settings, a tool-independent oracle).

## Rules that are not negotiable

1. **No speed number without a correctness number.** Every solve record
   carries the oracle's verdict. Never add a timing path that skips
   `oracle.evaluate`. The one exception is the CIM benchmark (problem
   `cim`, `adapters/cim_adapter.py`), timed only like cim-bench: the tools
   read CGMES into different models (triples, typed objects, a network
   model) and validate against different rules, so there is no one answer
   to grade. What each read and found is recorded and shown, never graded.
   No other problem gets this exception.
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
   tolerance `SE_TOLERANCE` on the state update. AC-OPF
   (`adapters/optimizer_adapter.py`): MATPOWER's formulation (polynomial
   cost; voltage, generator, branch MVA and angle-difference limits), flat
   start where settable, `OPF_TOLERANCE`. Batch power flow
   (`adapters/batch_adapter.py`): every scenario the power-flow problem
   (flat start per scenario, never the previous scenario's result), the
   tool's own thread setting with BLAS pinned to one thread, every scenario
   of every thread count graded. N-1 (`ContingencyAdapter`, same module):
   the same, except that every outage starts from the tool's own solution
   of the base case (solved from flat start in the same timed call), the
   one exception to the flat start, as contingency analysis is done; a tool
   that cannot start from it says what it does instead. CIM
   (`adapters/cim_adapter.py`): every profile of the case as published,
   from the uncompressed XML; export warm, into an emptied directory;
   validate from files to report in one call, parsing included; cimoxide
   with its own rules, the validators without any (triplets, OpenCGMES) with
   the ENTSO-E CGMES 3.0 shapes `cases.registry.CIM_SHAPES`. Every setting in an adapter
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
   `adapters.REGISTRIES` name lists are the one exception; they load no tool).

## Layout

```
cases/       registry.py (every case: groups, family, grid, problem), matpower.py (.m reader), prep.py (tool inputs),
             truth.py (the power flow behind a state-estimation case), measurements.py (its measurement sets),
             pglib.py (PGLib's reference OPF objectives, read from its BASELINE.md),
             sweep.py (the operating points of a batch case), contingency.py (the outages of an N-1 case),
             gridoxide_matpower.py (vendored MATPOWER->PGM converter, gridoxide 0.0.2),
             matpower_to_cgmes.py (cimoxide converter), convert_pypowsybl.py (pypowsybl converter)
oracle/      ybus.py, residual.py (tier 1), cgmes_sv.py (tier 2), wls.py (state estimation), opf.py (AC-OPF),
             batch.py (batch power flow, tier 1 per scenario),
             evaluate.py (entry point),
             cgmes_model.py (tool-free CGMES reader: TN->bus join, converter fidelity),
             check_conversion.py (writes conversion.json)
adapters/    solver_adapter.py (the ABCs), <tool>_adapter.py, estimator_adapter.py + <tool>_se_adapter.py
             (state estimation), optimizer_adapter.py + <tool>_opf_adapter.py (AC-OPF),
             batch_adapter.py + <tool>_batch_adapter.py (batch power flow) and <tool>_n1_adapter.py (N-1),
             cim_adapter.py + <tool>_cim_adapter.py (CIM import/export/validate), jvm.py (JPype, Java tools),
             cgmes_ids.py, memory.py,
             octave_session.py + matpower_octave/ (MATPOWER's Octave side, timed inside Octave)
benchmarks/  benchmark_template.py (generates tests), conftest.py (selection, failures, metadata),
             <tool>_benchmark.py, <tool>_se_benchmark.py, <tool>_opf_benchmark.py,
             <tool>_batch_benchmark.py, <tool>_n1_benchmark.py, <tool>_cim_benchmark.py (3 lines each)
tools/       benchmark_data.py (loader, grids, scoreboard) + generate_{site,all}.py,
             palette.py, check_smoke.py (CI's smoke outcomes)
tool-configs/<tool>/pyproject.toml   dependencies of each image (tools pinned exactly)
tool-configs/matpower/Dockerfile      the official Octave image + uv Python + the MATPOWER release
tool-configs/sienna/Dockerfile, julia/   Julia on top of the base image; Project.toml, Manifest.toml,
             setup.jl (registry snapshot), GridBenchSienna (the adapter's Julia half, precompiled)
tool-configs/exapf/Dockerfile, julia/   the same for ExaPF.jl (GridBenchExaPF, CPU backend)
tool-configs/exapf_gpu/Dockerfile, julia/   ExaPF.jl on CUDA: exapf's GridBenchExaPF plus CUDA.jl and
             CUDSS.jl, the CUDA runtime fixed by LocalPreferences.toml (built without a GPU);
             needs-gpu: left out of a default run_benchmark.sh sweep where nvidia-smi fails
tool-configs/gpusim2grid/Dockerfile   compiles gpusim2grid (CUDA, cuDSS; not on PyPI) and the lightsim2grid
             it is seeded from (with its C++ headers) from pinned sources; needs-gpu, like exapf_gpu
tool-configs/sparlectra/Dockerfile, julia/   the same for Sparlectra.jl (GridBenchSparlectra, se.jl: estimation)
tool-configs/p3s/Dockerfile           compiles p3s's C++/KLU extension (not on PyPI) from pinned sources,
             with OpenMP for its batch solver (libgomp from the same pinned gcc image)
tool-configs/<tool>/pom.xml, jars.sha256   a Java library (opencgmes, powsybl): exact versions, and the
             SHA-256 of every jar they resolve to; built by docker/java-tool.dockerfile
docker/      base.dockerfile, tool.dockerfile, java-tool.dockerfile, docker-compose.yml, build.sh, run_*.sh
tests/       the oracle's own tests (test_wls.py: state estimation, test_opf.py: AC-OPF, test_batch.py and
             test_contingency.py: sweeps and outages), and the converter's
             (exactness + planted errors)
data/        submodules: benchmark-grids (MATPOWER, PGLib-OPF), CGMES-Test-Configurations,
             application-profiles-library (ENTSO-E's CGMES RDFS and SHACL)
results-docker/  published results: <tool>.json, <tool>-se.json, <tool>-opf.json, <tool>-batch.json, <tool>-n1.json,
             <tool>-cim.json, conversion.json; gpu/: GPU tools and the CPU baselines run on their machine
docs/index.html  generated site
```

## Commands

```bash
docker/build.sh [tool ...]                     # images (base, harness, tools)
docker/lock.sh                                 # re-resolve every uv.lock and Julia Manifest.toml (review, then commit)
docker/run_benchmark.sh [tool ...] [-- --groups smoke]   # prep, oracle tests, tools, reports
docker/run_single.sh pandapower --cases case14,case300   # one tool, quick iteration
GRID_BENCH_RESULTS=gpu docker/run_single.sh exapf_gpu    # into results-docker/gpu/ (GPU tabs); a baseline likewise
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
   encoding, not another hue, see the palette's docstring (p3s is
   pandapower's blue, dashed: `P3S`; ExaPF.jl is Sienna's red, dotted: `EXAPF`, and on a
   GPU dash-dotted: `EXAPF_GPU`; gpusim2grid lightsim2grid's orange, dash-dotted: `GPUSIM2GRID`); a reference implementation uses `REFERENCE`, drawn dashed; PowerModels.jl, PGLib-OPF's
   reference solver, `REFERENCE_DOTTED`; the CIM libraries, which only
   share a chart with pypowsybl, slot hues dotted: `CIMOXIDE` etc.), `package`, `modules`
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
   An AC-OPF the same way: `adapters/<tool>_opf_adapter.py` on
   `OptimizerAdapter`, in `OPTIMIZERS`, `create_benchmarks("<tool>", "opf")`,
   `<tool>-opf.json`; the solution carries the dispatch of every online
   generator, keyed by gen row. On `pglib_opf_case14_ieee` a correct tool is
   feasible to about 1e-4 MVA and 9e-6 below PGLib's rounded reference.
   Batch power flow the same way: `adapters/<tool>_batch_adapter.py` on
   `BatchAdapter` (a native batch API, timed per thread count) or
   `LoopBatchAdapter` (the power-flow adapter's solve per scenario, one
   thread), in `BATCHES`, `create_benchmarks("<tool>", "batch")`,
   `<tool>-batch.json`. Scenarios are joined by bus number and gen row. On
   `case14#sweep` a correct tool shows residuals around 1e-9 MVA on all 100
   scenarios, at every thread count. N-1 likewise: `<tool>_n1_adapter.py`
   on `ContingencyAdapter` or `LoopContingencyAdapter`, in `CONTINGENCIES`,
   `create_benchmarks("<tool>", "n1")`, `<tool>-n1.json`; outages are joined
   by branch row (with from and to bus to assert). On `case14#n1` a correct
   tool passes all 19 outages. A CIM library (most have no power-flow
   adapter): `adapters/<tool>_cim_adapter.py` on `CimAdapter` (`load`,
   `export`, `counts`, and `validate` with `validates = True` if it has a
   validator), in `CIM`, `create_benchmarks("<tool>", "cim")`,
   `<tool>-cim.json`. Read from the published files, never a zip prepared
   for it; never write next to them (the checkout is read-only in the
   containers; natively, triplets once wrote over a fixture). A Java library
   is a `tool-configs/<tool>/pom.xml` (exact versions) on
   `docker/java-tool.dockerfile`, started through `adapters/jvm.py`; run
   `docker/lock.sh` for its `jars.sha256`. On `cgmes_svedala#cim` a library
   reads 90 lines, 39 generators, 73 loads and 56 substations.
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
power flow and graded by `oracle.wls`), and `opf-pglib` (AC-OPF on
PGLib-OPF v23.07, keyed by PGLib's names, graded by `oracle.opf` against
the `.m` and PGLib's reference objective), and `sweep-matpower`,
`sweep-distribution` (batch power flow: a case plus 100 operating points,
keyed `<case>#sweep`, each scenario graded by tier 1 in `oracle.batch`),
and `n1-matpower` (N-1: a transmission case plus its outages, keyed
`<case>#n1`, each graded the same way against the case with that branch
out; radial grids have no outage that keeps them connected), and
`cim-cgmes` (CIM import, export and validation: a CGMES fixture read whole,
SV included, keyed `<case>#cim`, timed and not graded). A tool declares the families it
reads in `SolverAdapter.families`. Converted cases are keyed
`<case>@<converter>`, state-estimation cases `<case>~<scenario>`; a case's
`problem` ("pf", "se", "opf", "batch", "n1" or "cim") says which adapter solves it. Branch on a case's input
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

`tools/generate_all.py` writes `docs/index.html` (whose charts and tables
are drawn in the browser from the data embedded in it) from the JSON in a
results directory; run it (or the `reports` compose service) after any
change to `tools/`, and commit what it writes. Conventions:

- Only cases in the default groups are published (`benchmark_data.load`);
  a case run by name stays in its JSON.
- Tables are split by `grid` (transmission, distribution, fixtures), not by
  family: a converted case is a row under the case it came from, with an
  input column, so what the input route changes is one row apart.
- The scoreboard (`benchmark_data.scoreboard`) shows the `robustness`
  group as "hard transmission cases".
- The chart shows one input at a time (one line per tool needs one
  input); for state estimation, the input is the measurement scenario, for
  OPF the operating condition (typical, congested, small angle difference).
- Batch power flow and N-1 are shown per scenario or outage (the call's
  median over its size), at one thread and at each tool's fastest thread
  count, with a thread-scaling plot (small multiples, one per case) for
  APIs that take a thread count.
  Scaling numbers depend on the machine: `grid_bench.cpus` in each JSON
  records the cores the run had.
- CIM has operations instead of a solve (Import, Export, Validate,
  Memory, the last with its own selector: import, import + export,
  validate, each measured in a fresh process by `adapters/memory.py`); its scoreboard is "done of all" per operation, and in place of
  the accuracy table it shows what each tool read and found, ungraded.
- GPU tabs (power flow, batch, N-1) read `results-docker/gpu/`, not the
  top level: GPU tools and the CPU baselines run on the same machine, which
  need not be the machine behind the other tabs, so they are compared only
  with each other. A tool is a GPU tool when its JSON records a device
  (`dependencies.gpu`); every other tool there is a baseline. Run both
  with `GRID_BENCH_RESULTS=gpu`.

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
sparlectra, pinned to its newest release on purpose, see its `setup.jl`;
p3s likewise, through `exclude-newer-package` in its `pyproject.toml`,
and gpusim2grid, which has no release, at the newest commit of its main
branch, see its Dockerfile),
the Fuseki jar by SHA-256 (`docker/fuseki/Dockerfile`), GNU Octave by the
official image's digest and the MATPOWER release zip by SHA-256
(`tool-configs/matpower/Dockerfile`), the gcc image by digest and the
SuiteSparse and p3s source tarballs by SHA-256 (`tool-configs/p3s/Dockerfile`),
NVIDIA's CUDA devel image by digest and the cuDSS, lightsim2grid sdist and
gpusim2grid tarballs by SHA-256 (`tool-configs/gpusim2grid/Dockerfile`),
the Maven and JRE images by digest and every jar of a Java library by
SHA-256 (`tool-configs/*/jars.sha256`, checked by
`docker/java-tool.dockerfile`; its pom pins versions at least 7 days old),
the ENTSO-E shapes and vocabularies by the `application-profiles-library`
submodule's commit, the CI checkout action by commit. Do not add `apt`/`apk` installs. To
update anything, change the pin deliberately and say why in the commit.

## Style

Flat over nested, functions over classes unless there is state, no
defensive try/except (tools raise, conftest records), no hardcoded tool
names outside adapters and the registry. Comments explain why, not what.
