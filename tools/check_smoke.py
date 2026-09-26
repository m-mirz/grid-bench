"""Checks a smoke run against the known outcome of each smoke case.

    python3 tools/check_smoke.py <tool> [results_dir]

Every smoke case must end the way benchmarks/smoke_expectations.json says:
solved and accepted by the oracle (`ok`), solved to a different problem
(`rejected`, a documented finding), a recorded failure (`failed`), or solved
without a tier-1 verdict (`solved`, CGMES fixtures). A missing case, or any
change in either direction, fails the check: a regression, or a fix that
should be recorded as the new expectation. A tool with a state estimator has
its state-estimation smoke cases (`<case>~<scenario>`) in the same entry,
read from `<tool>-se.json`. Standard library only, so CI can run it on the
host.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def outcomes(results: dict) -> dict[str, str]:
    out = {}
    for b in results["benchmarks"]:
        e = b["extra_info"]
        if e["operation"] == "solve":
            out[e["case"]] = "solved" if "oracle_ok" not in e else ("ok" if e["oracle_ok"] else "rejected")
    for f in results["failures"]:
        if f["operation"] == "solve":
            out.setdefault(f["case"], "failed")
    return out


def main(tool: str, results_dir: str = "results-docker") -> int:
    expected = json.loads((ROOT / "benchmarks" / "smoke_expectations.json").read_text())[tool]
    paths = [Path(results_dir) / f"{tool}.json", Path(results_dir) / f"{tool}-se.json"]
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
