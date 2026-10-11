"""JuliaGrid.jl: AC Newton-Raphson on its KLU factorization (Julia, through
juliacall), the power-flow column of a library that also does state
estimation and AC-OPF (adapters/juliagrid_se_adapter.py,
juliagrid_opf_adapter.py) and runs batch and N-1 as loops.

Input: JuliaGrid's own MATPOWER reader (`powerSystem`) on the case's `.m`,
power-flow data only (`optimal=false`), given a copy (`readable`) that
leaves the equations untouched, for two limits of that reader: it takes
`mpc.baseMVA` from the first line naming it, a comment in the plain
distribution files (`% Sbase = mpc.baseMVA * 1e6;`, which it fails to
parse), and rejects a row's trailing comment in `gencost` (PGLib's `% NG`),
so comments are dropped; and with OPF data it reads
generator columns 11-16 (ramp and capability) unconditionally, where
PGLib's files have MATPOWER's 10 core columns, so the gen matrix is padded
to 21 with zeros, which is what MATPOWER's `loadcase` does (and zero
capability columns are no curve, for JuliaGrid as for MATPOWER). `load`
asserts that the copy parses to the raw case's data; the oracle grades the
raw case. Bus ids: JuliaGrid keeps buses,
branches and generators in the `.m`'s row order and refers to buses by index
(its labels are the bus names where a case has `bus_name`, so they are no
join); `load` asserts that order against the case: every branch's from and
to bus and every generator's bus, by JuliaGrid's index, are the row's
(`join`). The Julia half is tool-configs/juliagrid/julia/GridBenchJuliaGrid.

Settings:
- `newtonRaphson(system, KLU)`: Newton-Raphson in polar coordinates, KLU
  for each step's linear system (JuliaGrid's default is UMFPACK's LU).
- Flat start: every bus voltage set to 1 p.u. and 0 rad in the system
  before the model is built (JuliaGrid otherwise starts from the `.m`'s VM
  and VA columns, often a solved operating point); `setInitialPoint!`
  restores it before every solve and puts generator setpoints on regulated
  buses.
- The loop of JuliaGrid's `powerFlow!` (`GridBenchJuliaGrid.iterate!`): stop
  when both the largest |ΔP| and |ΔQ| are below `TOLERANCE_PU` (p.u.), at
  most `MAX_ITERATIONS` Newton steps; written out because `powerFlow!`
  reports neither convergence nor the step count.
- Bus types as JuliaGrid sets them, as MATPOWER does: a PV bus without an
  in-service generator is PQ, a generator on a PQ bus stays a fixed P/Q
  injection. No reactive limits (`reactiveLimit!` is not called), one slack.
- Julia single-threaded, logs off (warnings and info).

Results (first run, AMD EPYC 7J13): every case accepted but three. The hard
cases case1888rte and case6495rte do not converge from the flat start, and
on case2848rte it converges (6.9e-7 MW) to the low-voltage root MATPOWER
and Sparlectra.jl reach from the same start, |V| down to 0.02 p.u., which
the oracle rejects (`VM_FLOOR_PU`).
"""
import re
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from adapters.solver_adapter import MAX_ITERATIONS, TOLERANCE_PU, DidNotConverge, Solution, SolverAdapter
from cases.matpower import BUS_I, F_BUS, GEN_BUS, T_BUS, parse_m
from cases.registry import CASES


GEN_COLUMNS = 21   # MATPOWER's full gen matrix


def readable(path: Path) -> str:
    """A copy of the `.m` for JuliaGrid's reader, in this process's temp
    directory: comments dropped (but on lines with a quoted name), the gen
    matrix padded with zero columns to MATPOWER's 21. Asserted to parse to
    the same data."""
    lines, in_gen = [], False
    for line in Path(path).read_text().splitlines():
        if "'" not in line:   # a quoted name (bus_name) may hold a %
            line = line.split("%")[0].rstrip()
        if not line.strip():
            continue
        if re.match(r"\s*mpc\.gen\s*=\s*\[", line):
            in_gen = True
        elif in_gen and line.strip().startswith("]"):
            in_gen = False
        elif in_gen and line.strip():
            values = line.replace(";", " ").split()
            line = "\t" + "\t".join(values + ["0"] * (GEN_COLUMNS - len(values))) + ";"
        lines.append(line)
    out = Path(tempfile.gettempdir()) / f"juliagrid-{Path(path).name}"
    out.write_text("\n".join(lines) + "\n")
    raw, copy = parse_m(path), parse_m(out)
    assert raw["baseMVA"] == copy["baseMVA"]
    for key in ("bus", "branch", "gencost"):
        assert np.array_equal(raw.get(key, np.zeros(0)), copy.get(key, np.zeros(0))), key
    gen = raw["gen"]
    assert np.array_equal(copy["gen"][:, :gen.shape[1]], gen) and not copy["gen"][:, gen.shape[1]:].any()
    return str(out)


def _gb():
    from adapters.juliagrid_julia import GB   # starts Julia on first use
    return GB


def join(case: str, system) -> list[str]:
    """MATPOWER bus numbers in JuliaGrid's bus order, the row order asserted
    through every branch's and generator's buses."""
    mpc = parse_m(CASES[case]["file"])
    number = [int(b) for b in mpc["bus"][:, BUS_I]]
    n, f, t, g = _gb().layout(system)
    assert int(n) == len(number), "one JuliaGrid bus per bus row"
    assert [number[int(i) - 1] for i in f] == [int(b) for b in mpc["branch"][:, F_BUS]], "branch from-buses"
    assert [number[int(i) - 1] for i in t] == [int(b) for b in mpc["branch"][:, T_BUS]], "branch to-buses"
    assert [number[int(i) - 1] for i in g] == [int(b) for b in mpc["gen"][:, GEN_BUS]], "generator buses"
    return [str(b) for b in number]


class JuliagridAdapter(SolverAdapter):
    name = "juliagrid"
    display_name = "JuliaGrid.jl"
    color = "#0f8fa7"   # tools/palette.py JULIAGRID: Sparlectra's teal, dashed
    package = "juliacall"
    modules = ("adapters.juliagrid_julia",)
    language = "julia"
    families = ("matpower", "distribution")
    settings = {"solver": "newtonRaphson", "formulation": "polar", "linear_solver": "klu", "init": "flat",
                "reactive_limits": False, "distributed_slack": False, "tolerance_pu": TOLERANCE_PU,
                "tolerance_norm": "max |ΔP|, |ΔQ|", "max_iteration": MAX_ITERATIONS}

    def version(self):
        return self.dependencies()["JuliaGrid"]

    def dependencies(self):
        from adapters.juliagrid_julia import JULIA_VERSION
        return {k: str(v) for k, v in _gb().versions().items()} | {"julia": JULIA_VERSION,
                                                                    "juliacall": version("juliacall")}

    def load(self, case):
        model = _gb().load(readable(CASES[case]["file"]))
        return {"jl": model, "bus_ids": join(case, model.system), "iterations": None}

    def solve(self, model):
        steps = _gb().solve_b(model["jl"], TOLERANCE_PU, MAX_ITERATIONS)   # juliacall spells solve! as solve_b
        if steps < 0:
            raise DidNotConverge(f"NR did not converge in {MAX_ITERATIONS} iterations")
        model["iterations"] = int(steps)

    def solution(self, model, case):
        vm, va = _gb().solution(model["jl"])
        ids = model["bus_ids"]
        return Solution(dict(zip(ids, map(float, vm))), dict(zip(ids, map(float, va))), model["iterations"])
