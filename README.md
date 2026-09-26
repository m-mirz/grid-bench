# grid-bench

A benchmark of power system analysis software. Inspired by
[cim-bench](https://github.com/Haigutus/cim-bench), which compares how fast
tools *read* CGMES grid models, grid-bench compares what the tools are for:
**solving** them. It does not stop at speed. Every solve is checked by an
oracle that none of the tools under test takes part in.

v1 covers AC power flow in ten tool setups:
[pandapower](https://github.com/e2nIEE/pandapower),
[lightsim2grid](https://github.com/Grid2op/lightsim2grid),
[PyPSA](https://github.com/PyPSA/PyPSA),
[power-grid-model](https://github.com/PowerGridModel/power-grid-model),
[pypowsybl](https://github.com/powsybl/pypowsybl) (OpenLoadFlow),
[VeraGrid](https://github.com/SanPen/VeraGrid),
[Sienna](https://github.com/Sienna-Platform) (PowerFlows.jl),
[Sparlectra.jl](https://github.com/Welthulk/Sparlectra.jl),
[MATPOWER](https://matpower.org) on GNU Octave, and power-grid-model on CGMES
through [cgmes2pgm](https://github.com/SOPTIM/cgmes2pgm_suite).

**Results:** [`results-docker/comparison.md`](results-docker/comparison.md) ·
[site](docs/index.html) (sortable, filterable, with hover detail)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results-docker/graphs/solve_matpower-dark.svg">
  <img alt="Warm AC power-flow solve time versus buses, log-log, one line per tool" src="results-docker/graphs/solve_matpower.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results-docker/graphs/memory_matpower-dark.svg">
  <img alt="Peak memory added by loading and solving versus buses, log-log, one line per tool" src="results-docker/graphs/memory_matpower.svg">
</picture>

## What it found

Numbers are in [comparison.md](results-docker/comparison.md); each finding
is traced to its cause in the tool's adapter docstring (`adapters/`).

- **lightsim2grid is the fastest correct solver** on every MATPOWER case but
  one (17 ms for 9,241 buses), and that one is an importer difference, below.
- **Several importers solve a different problem than the case defines, and
  converge without complaint.** The oracle names each difference:
  - pypowsybl's MATPOWER importer puts a transformer's line charging on one
    side instead of splitting it; Sienna's keeps only one half.
  - PyPSA and VeraGrid let offline generators keep regulating voltage;
    lightsim2grid and VeraGrid make online generators on PQ-typed buses
    regulate.
  - pandapower's `from_mpc` rewrites branches into physical units and loses
    fidelity on 4 of 8 transmission cases.
  - VeraGrid solves a case whose `baseMVA` is not 100 wrongly; PyPSA imports
    open tie switches as closed.
  - Sienna builds Ybus in single precision (residuals of 1e-5 to 1e-3 MW
    where others reach 1e-9), and cannot parse pure phase shifters.
- **Read as CGMES, some of those problems disappear.** pypowsybl solves every
  cimoxide-converted case exactly (except case2848rte, which it cannot solve
  from the `.m` either), including the three its MATPOWER importer gets
  wrong. The CGMES importers have their own losses: pandapower drops the
  sign of negative reactances, VeraGrid drops phase shifts.
- **power-grid-model's** experimental voltage regulation either diverges or
  holds the slack off its setpoint, on every MATPOWER case.
- **Sparlectra.jl's model is exact** (every MATPOWER case it converges on passes), and
  it is the only tool to solve a hard case (case1888rte). Its rectangular
  Newton-Raphson diverges from the common flat start on case9241pegase.
- **cgmes2pgm is built for state estimation:** generators become fixed P/Q
  injections and the slack sits at nominal voltage, so its power flow misses
  the case by design.
- **CGMES fixtures:** only pypowsybl solves RealGrid (6,051 nodes).
- **Container isolation caught a mislabelled number:** with lightsim2grid
  installed, `pandapower.runpp` silently hands its solve to lightsim2grid,
  which made "pandapower" look 3x faster
  ([CONTAINERIZATION_VALIDATION.md](CONTAINERIZATION_VALIDATION.md)).

## Why an oracle

Comparing tools against each other cannot tell you which one is right. In
gridoxide's benchmark, five tools agreed on case14 to three decimals and all
five were wrong the same way, because their shared reference had a
conversion bug. A residual check against the case's own equations found
four real bugs the five-way comparison had missed.

So every result is graded in up to three tiers, strongest first:

1. **Residual (MATPOWER cases, and their CGMES conversions).** The benchmark
   builds its own Ybus from the `.m` file (`oracle/ybus.py`, following
   MATPOWER's `makeYbus.m`) and evaluates `ΔS = V·conj(Ybus·V) − S` with the
   tool's voltages: P and Q where the case specifies them, |V| against
   generator setpoints.
2. **Published solution (CGMES fixtures).** Deviation from the SV profile
   the fixture ships. Buses are joined through the case's own TP profile,
   never by nearest voltage.
3. **Agreement between tools.** Reported, and labelled the weakest evidence.

The oracle is tested before any tool (`tests/test_oracle.py`): Ybus against
a network written out by hand, and a planted reactance sign flip that must
be caught.

## Same problem for every tool

Every adapter configures its tool to solve the same problem, and its
docstring justifies each setting and what it deliberately does not do:

- the tool's *own* solver (pandapower would otherwise hand off to lightsim2grid)
- flat start on **every** solve, so repeated timings do not warm-start
- one slack bus, the case's own
- no reactive limits, no outer-loop controls (taps, phase shifters, switched shunts)
- generator voltage regulation as the case defines it, including a remote
  regulated terminal in CGMES
- convergence tolerance 1e-8 p.u. on the power mismatch where the tool
  exposes one

## How it is measured

- **Solve (the headline):** one persistent model per tool and case, one
  untimed warm-up solve (JIT, first symbolic factorization), then repeated
  solves until 2 s are filled; the median is reported.
- **Import:** file to model, median of 3 cold loads after one warm-up load.
  Input conversion the benchmark does itself (`cases/prep.py`) is never
  timed.
- **Memory:** peak RSS of a freshly spawned process after loading and
  solving, minus the peak after merely importing the tool.
- **Failures are results:** non-convergence, importer crashes and timeouts
  are recorded with the tool's real exception.
- **Isolation:** one container per tool, no network, run one at a time.
  Everything a build fetches is pinned (images by digest, packages by
  lockfile, downloads by SHA-256; see [CLAUDE.md](CLAUDE.md#pinning)), and
  tool releases are at least a week old (one documented exception).

## Cases

Chosen for what they exercise, not just their size (`cases/registry.py`):

| group | cases | purpose |
|---|---|---|
| smoke | case14, case33bw, CGMES PowerFlow | pipeline check; timing is call overhead |
| scaling | case118, case300, PEGASE 1354 / 2869 / 9241, generated MV/LV grids mvlv1004 / 10616 / 29840, CGMES SmallGrid / Svedala / RealGrid | the timing headline |
| feature | case300, case3120sp, case2848rte, feeders case4_dist / case18 / case33bw, CGMES MicroGrid-BE / MiniGrid | the accuracy headline: line charging, negative reactances, PV buses without generators, offline equipment; radial feeders with high R/X and heavy loading |
| robustness | case1888rte, case6495rte | "hard transmission cases": not expected to converge from a flat start, reported separately |

Reports group the cases by grid: meshed **transmission** grids, radial
**distribution** grids (literature feeders, and power-grid-model's generated
MV/LV grids down to 0.67 p.u.), and the ENTSO-E **CGMES fixtures**. MATPOWER
files come from the [benchmark-grids](https://github.com/m-mirz/benchmark-grids)
submodule (feeders as plain-data copies, since their unit conversions are
MATLAB code no importer runs); CGMES from
[CGMES-Test-Configurations](https://github.com/m-mirz/CGMES-Test-Configurations).

**Transmission cases as CGMES.** The fixtures have a weak reference and no
size ladder, so the transmission cases are also converted to CGMES 3.0 and
graded against the original `.m`. Two converters are checked without any
tool (`oracle/check_conversion.py`): the one written for this benchmark on
[cimoxide](https://github.com/m-mirz/cimoxide) (`cases/matpower_to_cgmes.py`)
is exact, so tools solve its output; pypowsybl's MATPOWER import + CGMES
export is not (no slack, Ybus off by up to 3e-4, setpoints missing), so its
output is graded but not solved by default.

## How each tool is driven

| tool | MATPOWER input | CGMES input | solver |
|---|---|---|---|
| pandapower | `from_mpc` (.mat) | `from_cim` | `runpp`, NR, numba |
| lightsim2grid | `init_from_matpower` (.mat) | — | `NR_KLU` |
| PyPSA | `import_from_pypower_ppc`, transformers as pi-model | — | `pf()` |
| power-grid-model | PGM JSON via `gridoxide.matpower` | — | NR with experimental voltage regulators |
| pypowsybl | `network.load` (.mat) | `network.load` (zip); slack from `referencePriority` | OpenLoadFlow |
| VeraGrid | `parse_matpower_file` (.m) | `open_cgmes` | NR |
| Sienna | `PowerSystems.System` (.m) via juliacall | — | PowerFlows.jl NR (KLU) |
| Sparlectra.jl | `createNetFromMatPowerFile` (.m) via juliacall | `importCGMES` (zip) | rectangular NR (UMFPACK) |
| MATPOWER | `loadcase` (.m), in GNU Octave | — | `runpf`, NR (UMFPACK) |
| PGM via cgmes2pgm | — | upload to a Fuseki sidecar, `CgmesToPgmConverter` | PGM 1.12 NR (generators as fixed P/Q) |

Every tool reads the case through its **own importer** wherever it has one.
The Julia tools run inside the benchmark process through
[juliacall](https://github.com/JuliaPy/PythonCall.jl), with their Julia half
precompiled into the image so no timing includes the JIT. MATPOWER, the
reference implementation (drawn dashed), is timed inside Octave with
tic/toc, so the bridge to it is in no number; it runs MATPOWER 8's default
MP-Core, which Octave executes slowly, so these are not MATLAB's numbers.

## Running it

```bash
git clone --recurse-submodules https://github.com/m-mirz/grid-bench && cd grid-bench
docker/build.sh                                  # base, harness and tool images
docker/run_benchmark.sh                          # everything: prep, oracle tests, tools, reports
docker/run_single.sh pypowsybl --cases case300   # one tool, some cases
```

For development without containers: `./setup.sh` (needs [uv](https://docs.astral.sh/uv/)),
then `./run_benchmarks.sh [tool ...] [-- --groups smoke]`.

Published numbers in `results-docker/` come from one full sweep on one
otherwise idle machine, recorded with its CPU, OS and git commit. CI only
checks that the harness works. For an A/B comparison, interleave the runs
(A, B, A, B).

## Adding a tool

See [CLAUDE.md](CLAUDE.md#adding-a-tool): one adapter, one three-line
benchmark file, one `tool-configs/<tool>/pyproject.toml`, one compose service.

## Credits

The adapter/container/report architecture follows
[cim-bench](https://github.com/Haigutus/cim-bench). The methodology (warm
timing, justified settings, the residual oracle, per-tool bus mappings)
comes from gridoxide's `scripts/bench`, from which `oracle/residual.py`,
`oracle/cgmes_sv.py` and `cases/matpower.py` are ported.

Licensed under Apache-2.0.
