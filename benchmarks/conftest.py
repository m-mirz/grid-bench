"""pytest plumbing: case selection, failure recording, result metadata.

- `--groups smoke,scaling` / `--cases case14,case300` select cases
  (default: every case in `cases.registry.DEFAULT_GROUPS`).
- Every failed test (non-convergence, import error, timeout) is recorded
  with its tool, case, operation and real exception line, and written into
  the pytest-benchmark JSON under `failures`, so a report shows
  `FAILED (IterationDiverge: ...)` instead of a missing cell.
- Tool metadata (version, dependencies, settings, tags, colour) and the git
  commit are written once per JSON under `grid_bench`, so the report
  generators never import an adapter.
"""
import os
import re

import pytest

from cases.registry import DEFAULT_GROUPS, select

SCHEMA_VERSION = 1


def pytest_addoption(parser):
    parser.addoption("--groups", default=",".join(DEFAULT_GROUPS), help="comma-separated case groups")
    parser.addoption("--cases", default="", help="comma-separated case names (overrides --groups)")


def pytest_configure(config):
    config.grid_bench_failures = []


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    adapter = metafunc.module.ADAPTER
    groups = metafunc.config.getoption("groups").split(",")
    names = [c for c in metafunc.config.getoption("cases").split(",") if c]
    cases = [c for fam in adapter.families for c in select(fam, groups, names)]
    metafunc.parametrize("case", cases)


def extract_error(text: str) -> str:
    """The most specific `SomeError: message` line of a traceback. The last
    line is often a generic trailer shared across unrelated exceptions (PGM's
    "Try validate_input_data()..." hint); prefer the real one. From gridoxide's
    run_case_suite.py."""
    lines = [re.sub(r"^E\s+", "", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and "validate_input_data" not in ln and "validate_batch_data" not in ln]
    for ln in reversed(lines):
        if re.match(r"^[\w.]+(Error|Exception|Diverge|Converge|Unsupported)\w*:\s", ln) or ln.startswith("Failed: Timeout"):
            return re.sub(r"^(\w+\.)+(?=\w+:)", "", ln)[:300]   # drop the module path of the class
    return (lines[-1] if lines else "failed")[:300]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.failed and report.when in ("setup", "call") and hasattr(item, "callspec"):
        item.config.grid_bench_failures.append({
            "tool": item.module.ADAPTER.name,
            "case": item.callspec.params["case"],
            "operation": item.originalname.removeprefix("test_"),
            "error": extract_error(report.longreprtext),
        })


def pytest_benchmark_update_json(config, benchmarks, output_json):
    adapters = {id(item.module.ADAPTER): item.module.ADAPTER for item in config.grid_bench_items}
    output_json["grid_bench"] = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": os.environ.get("GIT_SHA", "unknown"),
        "container_image": os.environ.get("GRID_BENCH_IMAGE", "native"),
        "tools": {a.name: {
            "display_name": a.display_name, "color": a.color, "tags": a.tags(), "language": a.language,
            "version": a.version(), "dependencies": a.dependencies(), "settings": a.settings,
            "families": list(a.families),
        } for a in adapters.values()},
    }
    output_json["failures"] = config.grid_bench_failures


def pytest_collection_modifyitems(config, items):
    config.grid_bench_items = [i for i in items if hasattr(i.module, "ADAPTER")]
