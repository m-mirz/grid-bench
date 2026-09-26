"""Peak memory of loading and solving one case, in a fresh interpreter.

cim-bench measured an RSS delta inside the benchmarking process after N
timed rounds, which mixes in everything the rounds accumulated. Here each
measurement runs in a newly spawned process:

- `rss_baseline_mb`: resident memory after importing every module the
                     adapter uses (`SolverAdapter.modules`), so lazy imports
                     inside `load` are not charged to the case.
- `rss_import_mb`:   peak resident memory while loading the case.
- `rss_solve_mb`:    peak resident memory over loading plus one solve.

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


def _child(tool: str, case: str) -> dict:
    from adapters import get_adapter, get_estimator
    from cases.registry import CASES

    adapter = (get_estimator if CASES[case]["problem"] == "se" else get_adapter)(tool)
    for name in adapter.modules:
        import_module(name)
    pid = adapter.tool_pid() or "self"
    Path(f"/proc/{pid}/clear_refs").write_text("5")   # reset VmHWM to current RSS
    out = {"rss_baseline_mb": _status_mb("VmRSS", pid)}
    model = adapter.load(case)
    out["rss_import_mb"] = _status_mb("VmHWM", pid)
    try:
        adapter.solve(model)
        out["rss_solve_mb"] = _status_mb("VmHWM", pid)
    except Exception:  # noqa: BLE001 - the solve test records why; here only memory matters
        pass
    return out


def measure(tool: str, case: str, timeout: float = 1800) -> dict:
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply_async(_child, (tool, case)).get(timeout)
