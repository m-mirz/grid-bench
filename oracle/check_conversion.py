"""Grades every MATPOWER -> CGMES conversion without any tool, and writes the
result next to the benchmark results.

    python -m oracle.check_conversion <results_dir>

For each converted case present in the cache, `oracle.cgmes_model.fidelity`
rebuilds Ybus, the specified injections and the voltage setpoints from the
CGMES files and compares them with the original `.m`. Also records whether a
slack is defined (any `referencePriority > 0`): a converted case without one
leaves each tool to pick its own slack bus, which changes the problem.
Output: `<results_dir>/conversion.json`, read by the report generators.
"""
import json
import sys
from pathlib import Path

from cases.matpower import parse_m
from cases.registry import CASES, cgmes_files
from oracle.cgmes_model import fidelity, of_class, read


def check(key: str) -> dict:
    case = CASES[key]
    paths = cgmes_files(key)
    if not paths:
        return {"case": key, "error": "not converted (converter not run)"}
    out = {"case": key, "source_case": case["source_case"], "converter": case["converter"]}
    out.update(fidelity(parse_m(CASES[case["source_case"]]["file"]), paths))
    machines = of_class(read(paths), "SynchronousMachine")
    out["slack_defined"] = any(float(m.get("SynchronousMachine.referencePriority") or 0) > 0 for m in machines.values())
    return out


def main(results_dir: str) -> None:
    keys = [k for k, c in CASES.items() if "converter" in c]
    report = [check(k) for k in keys]
    Path(results_dir, "conversion.json").write_text(json.dumps(report, indent=1))
    for r in report:
        print(r["case"], {k: v for k, v in r.items() if k not in ("case", "source_case", "converter")})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
