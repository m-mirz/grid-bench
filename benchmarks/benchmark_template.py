"""Generates the benchmark tests for one tool. A tool's benchmark file is:

    from benchmarks.benchmark_template import create_benchmarks
    create_benchmarks("pandapower")

which injects two tests into that module, each parametrized over every case
the tool can read (selected by `--groups` / `--cases`, see conftest.py):

- `test_import[case]`: cold load from disk into the tool's model.
  1 untimed warm-up round (lazy imports, first-call setup), then
  `IMPORT_ROUNDS` timed rounds, or 1 if the warm-up alone took longer than
  `SLOW_IMPORT_SECONDS` (cgmes2pgm uploading RealGrid to Fuseki would
  otherwise approach the 30-minute test timeout). Peak memory is measured separately in a
  fresh process (adapters/memory.py) and attached to this record.
- `test_solve[case]`: repeated solves on ONE persistent model, as every tool
  is used in practice and as every tool here supports. 1 untimed warm-up
  solve (JIT, first symbolic factorization), then enough timed rounds to
  fill `TARGET_SECONDS`, clamped to [MIN_ROUNDS, MAX_ROUNDS]. The solution
  of the last round is graded by the oracle and attached to the record.

A tool in a process of its own (`SolverAdapter.clock`) is timed by that
process: the records hold the time measured inside the tool.

Warm solve is the headline because cold numbers mostly measure one-time
setup: in gridoxide's bench, a 1.3-1.7x cold gap to lightsim2grid traced
entirely to symbolic factorization being redone.

A test that raises (non-convergence, unsupported input, timeout) is recorded
as a failure with its real exception by conftest.py; it never becomes a
blank cell.
"""
import inspect
import json
import math
import os
import time
from pathlib import Path

from adapters import get_adapter, memory
from cases.registry import CASES
from oracle.evaluate import evaluate

IMPORT_ROUNDS = 3
SLOW_IMPORT_SECONDS = 30.0
TARGET_SECONDS = 2.0
MIN_ROUNDS, MAX_ROUNDS = 5, 200
RESULTS = Path(os.environ.get("GRID_BENCH_RESULTS", Path(__file__).resolve().parent.parent / "results"))


def _record(benchmark, adapter, case: str, operation: str) -> None:
    benchmark.group = f"{operation}:{case}"
    if adapter.clock is not None:
        # pytest-benchmark (pinned) measures each round as the difference of
        # its `_timer` around the call: the tool's own clock makes that the
        # time measured inside the tool, bridge excluded.
        benchmark._timer = adapter.clock  # noqa: SLF001
    benchmark.extra_info.update({
        "tool": adapter.name, "case": case, "family": CASES[case]["family"],
        "groups": CASES[case]["groups"], "operation": operation,
    })


def _dump_solution(tool: str, case: str, sol) -> None:
    path = RESULTS / "solutions" / tool / f"{case}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"vm": sol.vm, "va_deg": sol.va_deg}))


def create_benchmarks(tool: str) -> None:
    adapter = get_adapter(tool)
    namespace = inspect.currentframe().f_back.f_globals
    namespace["ADAPTER"] = adapter

    def test_import(benchmark, case):
        _record(benchmark, adapter, case, "import")
        t0 = time.perf_counter()
        adapter.load(case)                        # warm-up, timed only to choose the round count
        rounds = 1 if time.perf_counter() - t0 > SLOW_IMPORT_SECONDS else IMPORT_ROUNDS
        benchmark.pedantic(adapter.load, args=(case,), rounds=rounds, iterations=1)
        benchmark.extra_info.update(memory.measure(tool, case))

    def test_solve(benchmark, case):
        _record(benchmark, adapter, case, "solve")
        model = adapter.load(case)
        adapter.solve(model)                      # warm-up; raises on non-convergence
        t0 = time.perf_counter()
        adapter.solve(model)                      # calibration
        rounds = min(MAX_ROUNDS, max(MIN_ROUNDS, math.ceil(TARGET_SECONDS / (time.perf_counter() - t0))))
        benchmark.pedantic(adapter.solve, args=(model,), rounds=rounds, iterations=1)
        sol = adapter.solution(model, case)
        _dump_solution(tool, case, sol)
        benchmark.extra_info.update({"iterations": sol.iterations, "n_reported": len(sol.vm)})
        benchmark.extra_info.update(evaluate(case, sol.vm, sol.va_deg))

    namespace["test_import"] = test_import
    namespace["test_solve"] = test_solve
