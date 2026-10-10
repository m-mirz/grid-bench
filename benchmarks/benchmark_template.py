"""Generates the benchmark tests for one tool. A tool's benchmark file is:

    from benchmarks.benchmark_template import create_benchmarks
    create_benchmarks("pandapower")

(a state-estimation benchmark file passes `"se"`, an optimal-power-flow one
`"opf"`: the same tests, on the tool's `EstimatorAdapter` or
`OptimizerAdapter`, over that problem's cases, graded by `oracle.wls` or
`oracle.opf`; a batch one `"batch"` and an N-1 one `"n1"`, see below)

which injects two tests into that module, each parametrized over every case
the tool can read (selected by `--groups` / `--cases`, see conftest.py):

- `test_import[case]`: cold load from disk into the tool's model.
  1 untimed warm-up round (lazy imports, first-call setup), then
  `IMPORT_ROUNDS` timed rounds, or 1 if the warm-up alone took longer than
  `SLOW_IMPORT_SECONDS` (cgmes2pgm uploading RealGrid to Fuseki would
  otherwise approach the 30-minute test timeout). Peak memory is measured separately in a
  fresh process (adapters/memory.py) and attached to this record, before
  this process loads the case: the two never hold it at once, which a
  model of half the machine's memory needs (ExaPF's batch of
  case9241pegase, 16 GB). For the same reason the timed rounds never hold
  two models: each round's model is released before the next round starts,
  outside the timer. A case that does not load leaves no memory record;
  this process's own load then raises and is recorded.
- `test_solve[case]`: repeated solves on ONE persistent model, as every tool
  is used in practice and as every tool here supports. 1 untimed warm-up
  solve (JIT, first symbolic factorization), then enough timed rounds to
  fill `TARGET_SECONDS`, clamped to [MIN_ROUNDS, MAX_ROUNDS], or 1 round
  and no calibration solve if the warm-up alone took longer than
  `SLOW_SOLVE_SECONDS` (an OPF of a few thousand buses in a Python solver
  takes a minute; eight of them per case made one tool's sweep take an
  hour, and repeats do not change a minute-long median). Outside OPF it
  applies to VeraGrid's state estimation on mvlv29840~exact (152 s) and can
  to PyPSA's power flow on mvlv29840 (29 s). The solution of the last round is
  graded by the oracle and attached to the record.

A tool in a process of its own (`SolverAdapter.clock`) is timed by that
process: the records hold the time measured inside the tool.

Batch power flow and N-1 (`"batch"`, `"n1"`, `adapters.batch_adapter`): `test_solve` is
also parametrized by `threads`, the adapter's `thread_counts()`. A round is
one call that solves every scenario of the sweep, timed the same way; every
thread count's solution is graded, scenario by scenario
(`oracle.evaluate.evaluate_batch`), since a race between threads would show
only there. One round is a whole batch, so these tests get
`BATCH_TIMEOUT_SECONDS` instead of the 30 minutes of pyproject.toml: a loop
of PyPSA's 5 s solve over 200 outages of case9241pegase takes 17 minutes a
round, warm-up included twice that.

CIM libraries (`"cim"`, `adapters.cim_adapter`) get no `test_solve`. They
get `test_import`, with what the model holds attached (`counts`), and:

- `test_export[case]`: one model, repeated exports of it into an emptied
  directory (emptied untimed, before each round), rounds chosen like a
  solve's. Files and bytes written are attached.
- `test_validate[case]` (validators only): files to violation report,
  rounds like an import's. The violation counts are attached, and the peak
  memory of one validate in a fresh process (`adapters/memory.py`).

These are timed, not graded: the one exception to "no speed number without
a correctness number" (AGENTS.md, rule 1), as in cim-bench.

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
import shutil
import time
from pathlib import Path

import pytest

from adapters import get, memory
from cases.registry import CASES
from oracle.evaluate import evaluate, evaluate_batch

IMPORT_ROUNDS = 3
SLOW_IMPORT_SECONDS = 30.0
SLOW_SOLVE_SECONDS = 30.0
TARGET_SECONDS = 2.0
MIN_ROUNDS, MAX_ROUNDS = 5, 200
BATCH_TIMEOUT_SECONDS = 3600
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
        "groups": CASES[case]["groups"], "operation": operation, "problem": CASES[case]["problem"],
    })


def _dump_solution(tool: str, case: str, sol) -> None:
    path = RESULTS / "solutions" / tool / f"{case}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"vm": sol.vm, "va_deg": sol.va_deg}
                               | ({"pg_mw": sol.pg_mw, "qg_mvar": sol.qg_mvar} if sol.pg_mw is not None else {})))


def create_benchmarks(tool: str, problem: str = "pf") -> None:
    adapter = get(tool, problem)
    namespace = inspect.currentframe().f_back.f_globals
    namespace["ADAPTER"] = adapter

    def test_import(benchmark, case):
        _record(benchmark, adapter, case, "import")
        benchmark.extra_info.update(memory.measure(tool, case))   # first: see the module docstring
        t0 = time.perf_counter()
        adapter.load(case)                        # warm-up, timed only to choose the round count
        rounds = 1 if time.perf_counter() - t0 > SLOW_IMPORT_SECONDS else IMPORT_ROUNDS
        # pytest-benchmark holds a round's return value until the next round
        # returns, i.e. two models at once; `setup` (untimed, before every
        # round) releases the previous one instead.
        held = []
        benchmark.pedantic(lambda: held.append(adapter.load(case)), setup=held.clear, rounds=rounds, iterations=1)
        if problem == "cim":
            benchmark.extra_info.update(adapter.counts(held[0]))

    def test_solve(benchmark, case):
        _record(benchmark, adapter, case, "solve")
        model = adapter.load(case)
        t0 = time.perf_counter()
        adapter.solve(model)                      # warm-up; raises on non-convergence
        if time.perf_counter() - t0 > SLOW_SOLVE_SECONDS:
            rounds = 1
        else:
            t0 = time.perf_counter()
            adapter.solve(model)                  # calibration
            rounds = min(MAX_ROUNDS, max(MIN_ROUNDS, math.ceil(TARGET_SECONDS / (time.perf_counter() - t0))))
        benchmark.pedantic(adapter.solve, args=(model,), rounds=rounds, iterations=1)
        sol = adapter.solution(model, case)
        _dump_solution(tool, case, sol)
        benchmark.extra_info.update({"iterations": sol.iterations, "n_reported": len(sol.vm)})
        benchmark.extra_info.update(evaluate(case, sol.vm, sol.va_deg, sol.pg_mw, sol.qg_mvar))

    @pytest.mark.timeout(BATCH_TIMEOUT_SECONDS)
    def test_solve_batch(benchmark, case, threads):
        _record(benchmark, adapter, case, "solve")
        benchmark.extra_info.update({"threads": threads, "mode": adapter.mode})
        model = adapter.load(case)
        t0 = time.perf_counter()
        adapter.solve(model, threads)             # warm-up; raises on non-convergence
        if time.perf_counter() - t0 > SLOW_SOLVE_SECONDS:
            rounds = 1
        else:
            t0 = time.perf_counter()
            adapter.solve(model, threads)         # calibration
            rounds = min(MAX_ROUNDS, max(MIN_ROUNDS, math.ceil(TARGET_SECONDS / (time.perf_counter() - t0))))
        benchmark.pedantic(adapter.solve, args=(model, threads), rounds=rounds, iterations=1)
        sol = adapter.solution(model, case)
        benchmark.extra_info.update(evaluate_batch(case, sol.bus_ids, sol.vm, sol.va_deg))

    def test_export(benchmark, case, tmp_path):
        _record(benchmark, adapter, case, "export")
        model = adapter.load(case)
        out = tmp_path / "export"

        def emptied():
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir()
            return (model, out), {}

        t0 = time.perf_counter()
        adapter.export(*emptied()[0])             # warm-up
        if time.perf_counter() - t0 > SLOW_SOLVE_SECONDS:
            rounds = 1
        else:
            t0 = time.perf_counter()
            adapter.export(*emptied()[0])         # calibration
            rounds = min(MAX_ROUNDS, max(MIN_ROUNDS, math.ceil(TARGET_SECONDS / (time.perf_counter() - t0))))
        benchmark.pedantic(adapter.export, setup=emptied, rounds=rounds, iterations=1)
        written = [p for p in out.rglob("*") if p.is_file()]
        assert written, f"{tool} wrote nothing"
        benchmark.extra_info.update({"files_written": len(written), "bytes_written": sum(p.stat().st_size for p in written)})

    def test_validate(benchmark, case):
        _record(benchmark, adapter, case, "validate")
        t0 = time.perf_counter()
        adapter.validate(case)                    # warm-up
        rounds = 1 if time.perf_counter() - t0 > SLOW_IMPORT_SECONDS else IMPORT_ROUNDS
        report = benchmark.pedantic(adapter.validate, args=(case,), rounds=rounds, iterations=1)
        benchmark.extra_info.update(report)
        benchmark.extra_info.update(memory.measure(tool, case, "validate"))

    namespace["test_import"] = test_import
    if problem == "cim":
        namespace["test_export"] = test_export
        if adapter.validates:
            namespace["test_validate"] = test_validate
    else:
        namespace["test_solve"] = test_solve_batch if problem in ("batch", "n1") else test_solve
