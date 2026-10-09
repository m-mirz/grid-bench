"""Checks a smoke run against the known outcome of each smoke case.

    python3 tools/check_smoke.py <tool> [results_dir]

Every smoke case must end the way benchmarks/smoke_expectations.json says:
solved and accepted by the oracle (`ok`), solved to a different problem
(`rejected`, a documented finding), a recorded failure (`failed`), or solved
without a tier-1 verdict (`solved`, CGMES fixtures). A missing case, or any
change in either direction, fails the check: a regression, or a fix that
should be recorded as the new expectation. A tool with a state estimator has
its state-estimation smoke cases (`<case>~<scenario>`) in the same entry,
read from `<tool>-se.json`, a tool with an OPF its OPF smoke case,
from `<tool>-opf.json`, and its batch smoke cases (`<case>#sweep`) from
`<tool>-batch.json`, and its N-1 smoke case (`case14#n1`) from
`<tool>-n1.json`. A batch or N-1 case has a record per thread count; their
outcomes must agree (`inconsistent` otherwise, which no expectation names:
a thread count that changes the answer is a race). A CIM library's smoke
case (`<case>#cim`, from `<tool>-cim.json`) is timed, not graded: `done`
when every operation the tool has (import, export, validate) ran, `failed`
when any of them recorded a failure. Standard library only, so CI can run
it on the host.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def outcomes(results: dict) -> dict[str, str]:
    seen: dict[str, set[str]] = {}
    for b in results["benchmarks"]:
        e = b["extra_info"]
        if e.get("problem") == "cim":   # older records carry no problem
            seen.setdefault(e["case"], set()).add("done")
        elif e["operation"] == "solve":
            seen.setdefault(e["case"], set()).add(
                "solved" if "oracle_ok" not in e else ("ok" if e["oracle_ok"] else "rejected"))
    for f in results["failures"]:
        if f["operation"] == "solve" or f["case"].endswith("#cim"):
            seen.setdefault(f["case"], set()).add("failed")
    for case, o in seen.items():
        if case.endswith("#cim") and "failed" in o:   # one failed operation fails the case
            o.discard("done")
    return {case: next(iter(o)) if len(o) == 1 else "inconsistent" for case, o in seen.items()}


def main(tool: str, results_dir: str = "results-docker") -> int:
    expected = json.loads((ROOT / "benchmarks" / "smoke_expectations.json").read_text())[tool]
    paths = [Path(results_dir) / f"{tool}{suffix}.json" for suffix in ("", "-se", "-opf", "-batch", "-n1", "-cim")]
    runs = [json.loads(p.read_text()) for p in paths if p.exists()]
    results = {"benchmarks": [b for r in runs for b in r["benchmarks"]],
               "failures": [f for r in runs for f in r["failures"]]}
    actual = outcomes(results)
    wrong = {case: {"expected": want, "actual": actual.get(case, "missing")}
             for case, want in expected.items() if actual.get(case) != want}
    for case, want in expected.items():
        print(f"{tool:14s} {case:22s} expected {want:9s} actual {actual.get(case, 'missing')}")
    if wrong:
        print(f"smoke check FAILED for {tool}: {wrong}")
        for f in results["failures"]:
            print(f"  recorded failure: {f['case']} {f['operation']}: {f['error']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
