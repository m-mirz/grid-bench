"""Peak memory of loading and solving one case, in a fresh interpreter.

cim-bench measured an RSS delta inside the benchmarking process after N
timed rounds, which mixes in everything the rounds accumulated. Here each
measurement runs in a newly spawned process:

- `rss_baseline_mb`: resident memory after importing every module the
                     adapter uses (`SolverAdapter.modules`), so lazy imports
                     inside `load` are not charged to the case.
- `rss_import_mb`:   peak resident memory while loading the case.
- `rss_solve_mb`:    peak resident memory over loading plus one solve.
- `rss_export_mb`:   a CIM library's (no solve): peak resident memory over
                     loading plus one export, into a temporary directory.
- `rss_validate_mb`: a CIM validator's peak resident memory over one
                     validate (files to report), measured on its own:
                     `measure(..., operation="validate")` spawns another
                     process, with its own `rss_baseline_mb`, since
                     validate reads the files itself and holds no model
                     from an import.

Peaks come from the kernel's high-water mark (`VmHWM`), reset after the tool
import by writing 5 to `/proc/self/clear_refs`. `getrusage().ru_maxrss` is not
usable here: Linux carries it across fork and exec, so a spawned child
starts with its parent's peak (the pytest process, which grows as larger
cases are loaded), which inflated baselines from 0.2 to 1.9 GB.

A tool running in a process of its own (`SolverAdapter.tool_pid`, MATPOWER
in Octave) is measured in that process instead, the same way.
"""
import multiprocessing
from importlib import import_module
from pathlib import Path


def _status_mb(field: str, pid: int | str = "self") -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024   # kB
    raise RuntimeError(f"{field} not in /proc/self/status")


def _child(tool: str, case: str, operation: str) -> dict:
    import tempfile

    from adapters import get
    from adapters.solver_adapter import SolvingAdapter
    from cases.registry import CASES

    adapter = get(tool, CASES[case]["problem"])
    for name in adapter.modules:
        try:
            import_module(name)
        except Exception as e:
            # Re-raised as a plain exception, since the parent unpickles what
            # the child raises: a tool's own exception type can need the tool
            # to unpickle (juliacall's JuliaError starts Julia in the pool's
            # result thread, which hangs: exapf_gpu without a usable GPU).
            raise RuntimeError(f"{type(e).__name__}: {e}") from None
    pid = adapter.tool_pid() or "self"
    Path(f"/proc/{pid}/clear_refs").write_text("5")   # reset VmHWM to current RSS
    out = {"rss_baseline_mb": _status_mb("VmRSS", pid)}
    if operation == "validate":
        adapter.validate(case)
        return out | {"rss_validate_mb": _status_mb("VmHWM", pid)}
    try:
        model = adapter.load(case)
        out["rss_import_mb"] = _status_mb("VmHWM", pid)
        if isinstance(adapter, SolvingAdapter):
            adapter.solve(model)
            out["rss_solve_mb"] = _status_mb("VmHWM", pid)
        else:
            with tempfile.TemporaryDirectory() as out_dir:
                adapter.export(model, Path(out_dir))
                out["rss_export_mb"] = _status_mb("VmHWM", pid)
    except Exception:  # noqa: BLE001 - the import, solve and export tests record why; here only memory matters
        pass
    return out


def measure(tool: str, case: str, operation: str = "import", timeout: float = 1800) -> dict:
    """`operation` "import": load, then one solve or export. "validate": one
    validate, in a process of its own."""
    # close and join, not the context manager's terminate: SIGTERM to a child
    # running Julia (juliacall handles signals) can deadlock it on exit, and
    # the parent then waits for it forever (seen with Sparlectra 0.30.1).
    # A child that failed or timed out is still terminated.
    pool = multiprocessing.get_context("spawn").Pool(1)
    try:
        out = pool.apply_async(_child, (tool, case, operation)).get(timeout)
    except BaseException:
        pool.terminate()
        raise
    pool.close()
    pool.join()
    return out
