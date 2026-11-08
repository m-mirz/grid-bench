"""The report's loader: what it publishes, and what it refuses."""
import json
from pathlib import Path

import pytest

from adapters.gpusim2grid_adapter import COMMIT
from cases.registry import CASES
from tools.benchmark_data import load, machines, scoreboard

DEFAULT = "case14#sweep"
LARGE = next(c for c, k in CASES.items() if k["problem"] == "batch" and "large" in k["groups"])
EPYC, RYZEN, A100 = "AMD EPYC 7J13", "AMD Ryzen 7 250", "NVIDIA A100"


def write(path, tool, cases=(DEFAULT,), device=None, cpu=EPYC, failures=()):
    meta = {"problem": "batch", "language": "C++", "families": ["sweep-matpower"], "display_name": tool,
            "dependencies": {"gpu": device} if device else {}, "tags": []}
    benchmarks = [{"stats": {"median": .001, "min": .001, "rounds": 3},
                   "extra_info": {"tool": tool, "case": c, "family": "sweep-matpower", "operation": "solve",
                                  "oracle_ok": True, "threads": 1}} for c in cases]
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"benchmarks": benchmarks, "failures": list(failures), "datetime": "2026-11-08",
                                "machine_info": {"cpu": {"brand_raw": cpu, "count": 30}},
                                "grid_bench": {"tools": {tool: meta}}}))


def test_load_publishes_gpu_tools_and_not_cpu_baselines(tmp_path):
    write(tmp_path / "lightsim2grid-batch.json", "lightsim2grid")
    write(tmp_path / "gpu" / "lightsim2grid-batch.json", "lightsim2grid")
    write(tmp_path / "gpu" / "gpusim2grid-batch.json", "gpusim2grid", device=A100)
    res = load(tmp_path).batch
    assert set(res.tools) == {"lightsim2grid", "gpusim2grid"}
    assert len(res.records) == 2   # the baseline in gpu/ is not counted again


def test_load_refuses_a_tool_with_two_results(tmp_path):
    write(tmp_path / "gpusim2grid-batch.json", "gpusim2grid", device=A100)
    write(tmp_path / "gpu" / "gpusim2grid-batch.json", "gpusim2grid", device=A100)
    with pytest.raises(ValueError, match="already has results"):
        load(tmp_path)


def test_cpu_and_gpu_tools_may_differ_in_machine(tmp_path):
    write(tmp_path / "lightsim2grid-batch.json", "lightsim2grid", cpu=RYZEN)
    write(tmp_path / "gpu" / "gpusim2grid-batch.json", "gpusim2grid", device=A100)
    found = machines(load(tmp_path).batch)
    assert found["cpu"]["cpu"] == RYZEN
    assert (found["gpu"]["cpu"], found["gpu"]["device"]) == (EPYC, A100)


def test_cpu_tools_of_one_problem_may_not_differ(tmp_path):
    write(tmp_path / "lightsim2grid-batch.json", "lightsim2grid")
    write(tmp_path / "pgm-batch.json", "pgm", cpu=RYZEN)
    with pytest.raises(ValueError, match="different machines"):
        machines(load(tmp_path).batch)


def test_scoreboard_separates_not_run_from_failed(tmp_path):
    write(tmp_path / "lightsim2grid-batch.json", "lightsim2grid", cases=(DEFAULT, LARGE))
    write(tmp_path / "pgm-batch.json", "pgm", cases=(LARGE,))   # the default case is missing: a run cut short
    write(tmp_path / "veragrid-batch.json", "veragrid",
          failures=[{"tool": "veragrid", "case": LARGE, "operation": "solve", "error": "x"}])
    columns, rows = scoreboard(load(tmp_path).batch)
    cells = {tool: dict(zip(columns, row)) for tool, row in rows}
    default = next(c for c in columns if "100 scenarios" in c)
    large = next(c for c in columns if c != default)
    assert cells["lightsim2grid"][large] == "1 / 0 / 0"
    assert cells["pgm"][default] == "0 / 0 / 1"
    assert cells["veragrid"][large] == "0 / 0 / 1"   # asked to run it, and it raised
    write(tmp_path / "veragrid-batch.json", "veragrid")
    columns, rows = scoreboard(load(tmp_path).batch)
    assert dict(rows)["veragrid"][columns.index(large)] == "·"   # never asked


def test_gpusim2grid_records_the_commit_its_image_builds():
    assert f"/{COMMIT}.tar.gz" in Path("tool-configs/gpusim2grid/Dockerfile").read_text()
