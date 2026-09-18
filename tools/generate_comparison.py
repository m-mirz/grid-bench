"""Writes `comparison.md`: the cross-tool tables, from result JSON only.

Every timing cell carries its oracle verdict, so no speed number appears
without a correctness number next to it:

    0.234 ✓        warm solve median in ms; the oracle's residual check passed
    5.087 ✗³       converged, but the solution does not solve the case (see note 3)
    FAILED¹        the tool raised; note 1 has the real exception
    —              the tool cannot read this case family
"""
import math
from pathlib import Path

import numpy as np

from cases.registry import CASES
from tools.benchmark_data import FAMILY_TITLES, Results, case_size, load_solution


class Notes:
    """Numbered footnotes, deduplicated by text."""

    def __init__(self):
        self.items: list[str] = []

    def ref(self, text: str) -> str:
        if text not in self.items:
            self.items.append(text)
        return "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(d)] for d in str(self.items.index(text) + 1))

    def render(self) -> str:
        return "\n".join(f"{i}. {t}" for i, t in enumerate(self.items, 1))


def fmt_ms(ms: float) -> str:
    return f"{ms:.3f}" if ms < 10 else f"{ms:.1f}" if ms < 1000 else f"{ms:,.0f}"


FAILING = {
    "pv_without_online_gen": "PV buses whose generators are all offline (MATPOWER solves them as PQ; "
                             "an offline generator is still regulating)",
    "pq_with_online_gen": "PQ buses with an online generator (MATPOWER: a fixed P/Q injection; "
                          "the generator is regulating voltage against the case)",
    "pq_with_offline_gen": "PQ buses with only offline generators (an offline generator is still regulating)",
    "pq": "other PQ buses", "pv": "PV buses", "ref": "the slack bus",
}


def residual_note(e: dict) -> str:
    failing = e.get("residual_failing", {})
    classes = "; ".join(f"{n} {FAILING.get(k, k)}" for k, n in sorted(failing.items(), key=lambda kv: -kv[1]))
    parts = [f"fails at {classes}"] if classes else []
    parts += [f"max |ΔP| {e['residual_max_dp_mw']:.3g} MW, max |ΔQ| {e['residual_max_dq_mvar']:.3g} MVAr"]
    if e["residual_max_dvm_pu"] > 1e-6:
        parts.append(f"|V| off its setpoint by {e['residual_max_dvm_pu']:.3g} p.u.")
    if e["residual_n_checked"] < e["residual_n_buses"]:
        parts.append(f"only {e['residual_n_checked']} of {e['residual_n_buses']} buses checkable")
    if "residual_zero_shift_max_dp_mw" in e and max(e["residual_zero_shift_max_dp_mw"],
                                                    e["residual_zero_shift_max_dq_mvar"]) < 1e-3:
        parts.append("residual vanishes if phase shifts are zeroed: the tool dropped them")
    return "; ".join(parts) + f" (worst: bus {e['residual_worst_bus']})"


def cell(res: Results, notes: Notes, tool: str, case: str, operation: str) -> str:
    if CASES[case]["family"] not in res.tools[tool]["families"]:
        return "—"
    rec = res.get(tool, case, operation)
    if rec is None:
        err = res.failure(tool, case, operation)
        return f"FAILED{notes.ref(f'{tool} on {case}: `{err}`')}" if err else "not run"
    text = fmt_ms(rec.median_ms)
    if operation != "solve" or "oracle_ok" not in rec.extra:
        return text
    if rec.extra["oracle_ok"]:
        return f"{text} ✓"
    return f"{text} ✗{notes.ref(f'{tool} on {case}: {residual_note(rec.extra)}')}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def family_tools(res: Results, family: str) -> list[str]:
    return [t for t in res.tool_order() if family in res.tools[t]["families"]]


def timing_section(res: Results, operation: str, family: str, notes: Notes) -> str:
    tools = family_tools(res, family) if family != "matpower" else res.tool_order()
    rows = [[f"{c}", f"{case_size(c):,}"] + [cell(res, notes, t, c, operation) for t in tools]
            for c in res.cases(family)]
    return table(["case", "nodes" if family == "cgmes" else "buses"] +
                 [res.tools[t]["display_name"] for t in tools], rows)


def per_family(res: Results, render, *args) -> str:
    parts = []
    for fam in res.families():
        parts += [f"### {FAMILY_TITLES[fam]}", "", render(res, *args, fam) if args else render(res, fam), ""]
    return "\n".join(parts)


def robustness_section(res: Results, notes: Notes) -> str:
    tools = res.tool_order()
    rows = [[c, f"{case_size(c):,}"] + [cell(res, notes, t, c, "solve") for t in tools]
            for c in res.cases("matpower", robustness=True)]
    return table(["case", "buses"] + [res.tools[t]["display_name"] for t in tools], rows) if rows else "Not run."


def memory_section(res: Results, family: str) -> str:
    tools = family_tools(res, family)
    rows = []
    for c in res.cases(family):
        row = [c]
        for t in tools:
            rec = res.get(t, c, "import")
            if rec is None or "rss_import_mb" not in rec.extra:
                row.append("—")
                continue
            e = rec.extra
            peak = e.get("rss_solve_mb", e["rss_import_mb"])
            row.append(f"{peak - e['rss_baseline_mb']:.0f} (+{e['rss_baseline_mb']:.0f})")
        rows.append(row)
    return table(["case"] + [res.tools[t]["display_name"] for t in tools], rows)


def accuracy_residual(res: Results, family: str = "matpower") -> str:
    tools = family_tools(res, family)
    rows = []
    for c in res.cases(family):
        row = [c]
        for t in tools:
            rec = res.get(t, c, "solve")
            if rec is None:
                row.append("failed")
                continue
            e = rec.extra
            worst = max(e["residual_max_dp_mw"], e["residual_max_dq_mvar"])
            row.append(f"{worst:.1e}" + ("" if e["oracle_ok"] else " ✗"))
        rows.append(row)
    return table(["case"] + [res.tools[t]["display_name"] for t in tools], rows)


def accuracy_cgmes(res: Results) -> str:
    tools = family_tools(res, "cgmes")
    rows = []
    for c in res.cases("cgmes"):
        row = [c]
        for t in tools:
            rec = res.get(t, c, "solve")
            if rec is None:
                row.append("failed")
                continue
            e = rec.extra
            if not e.get("sv_n"):
                row.append("n=0")
                continue
            row.append(f"{e['sv_dv_median']:.3%} / {e['sv_dv_max']:.2%} (n={e['sv_n']}/{e['sv_n_published']})")
        rows.append(row)
    return table(["case"] + [res.tools[t]["display_name"] for t in tools], rows)


def cross_tool(res: Results, directory: Path) -> str:
    """Tier 3, the weakest evidence: each tool's largest |V| deviation from
    the per-bus median of every tool that solved the case."""
    tools = res.tool_order()
    rows = []
    for c in res.cases("matpower"):
        sols = {t: s for t in tools if res.get(t, c, "solve") and (s := load_solution(directory, t, c))}
        if len(sols) < 3:
            continue
        buses = set.intersection(*(set(s["vm"]) for s in sols.values()))
        consensus = {b: float(np.median([s["vm"][b] for s in sols.values()])) for b in buses}
        row = [c, str(len(buses))]
        for t in tools:
            if t not in sols:
                row.append("—")
                continue
            dev = max((abs(sols[t]["vm"][b] - v) for b, v in consensus.items()), default=math.nan)
            row.append(f"{dev:.1e}")
        rows.append(row)
    return table(["case", "shared buses"] + [res.tools[t]["display_name"] for t in tools], rows)


def conversion_section(res: Results) -> str:
    if not res.conversion:
        return "Not run."
    rows = []
    for r in res.conversion:
        if "error" in r:
            rows.append([r["case"], "—", "—", "—", "—", "—", r["error"]])
            continue
        rows.append([r["source_case"], r["converter"], f"{r['buses']}/{r['buses_expected']}",
                     f"{r['max_dy_rel']:.1e}", f"{r['max_ds_mva']:.1e}",
                     f"{r['setpoints_missing']} / {r['setpoints_extra']}", "yes" if r["slack_defined"] else "**no**"])
    rows.sort(key=lambda row: (case_size(row[0]) if row[0] in CASES else 0, row[0], row[1]))
    return table(["case", "converter", "buses", "max rel. |ΔYbus|", "max |ΔS| MVA",
                  "setpoints missing / extra", "slack defined"], rows)


def environment(res: Results) -> str:
    rows = []
    for t in res.tool_order():
        m = res.tools[t]
        run = next((r for r in res.runs if r["tool"] == t), {})
        rows.append([m["display_name"], m["version"], m["language"],
                     ", ".join(f"{k}={v}" for k, v in m["settings"].items()),
                     str(run.get("git_sha", "?"))[:12], str(run.get("datetime", "?"))[:16]])
    machine = next((r["machine"] for r in res.runs if r["machine"]), {})
    cpu = machine.get("cpu", {})
    head = (f"Machine: {cpu.get('brand_raw', '?')}, {cpu.get('count', '?')} logical CPUs; "
            f"{machine.get('system', '?')} {machine.get('release', '')}; "
            f"Python {machine.get('python_version', '?')}.")
    return head + "\n\n" + table(["tool", "version", "core", "settings", "commit", "run"], rows)


def generate(directory: Path, res: Results) -> str:
    notes = Notes()
    parts = [
        "# grid-bench results",
        "",
        "Generated by `tools/generate_comparison.py` from the JSON in this directory. Do not edit by hand.",
        "",
        "## AC power flow: warm solve",
        "",
        "Median of repeated solves on one persistent model, flat start every time, in ms, per case family. "
        "✓: the solution satisfies the case's power-flow equations to 1e-3 MVA at every bus and holds every "
        "generator voltage setpoint (oracle tier 1, independent of every tool). "
        "✗: the tool converged, but to a solution of a different problem; the note says where.",
        "",
        per_family(res, lambda r, fam: timing_section(r, "solve", fam, notes)),
        "## Robustness",
        "",
        "Cases known not to converge from a flat start in any tool tested here. Kept out of the tables above; "
        "a tool that solves one of these (and passes the oracle) is doing something the others do not.",
        "",
        robustness_section(res, notes),
        "",
        "## Import: file to model",
        "",
        "Median of 3 cold loads after one warm-up load, in ms.",
        "",
        per_family(res, lambda r, fam: timing_section(r, "import", fam, notes)),
        "## Memory",
        "",
        "Peak RSS added by loading and solving the case, in MB, measured in a fresh process; "
        "in parentheses, the peak after merely importing the tool.",
        "",
        per_family(res, memory_section),
        "## Accuracy",
        "",
        "### Tier 1: residual against the MATPOWER case (MVA, worst bus)",
        "",
        "Also applied to converted cases: tool voltages are mapped back to MATPOWER buses by TopologicalNode name "
        "(`BUS-<n>`) and graded against the original `.m`.",
        "",
        "Largest |ΔP| or |ΔQ| of `V·conj(Ybus·V) − S` over the buses where it is specified, "
        "with Ybus built from the `.m` file by `oracle/ybus.py`. Every tool was asked to converge to 1e-8 p.u.",
        "",
        "\n\n".join(f"**{FAMILY_TITLES[fam]}**\n\n" + accuracy_residual(res, fam)
                    for fam in res.families() if fam != "cgmes"),
        "",
        "### Tier 2: deviation from the published CGMES solution",
        "",
        "|ΔV|/V against `SvVoltage`, median / max, and matched / published TopologicalNodes. The published "
        "solution comes from the fixture author's own tool and settings; it is a reference, not ground truth.",
        "",
        accuracy_cgmes(res),
        "",
        "### Converting MATPOWER to CGMES",
        "",
        "Each converter's output checked without any tool (`oracle/check_conversion.py`): Ybus, specified injections "
        "and voltage setpoints rebuilt from the CGMES files by a parser that shares no code with either converter, "
        "compared with the original `.m`. A missing slack leaves every tool to choose its own slack bus.",
        "",
        conversion_section(res),
        "",
        "### Tier 3: agreement between tools (weakest evidence)",
        "",
        "Largest |ΔV| (p.u.) from the per-bus median over the tools that solved the case. Agreement says "
        "nothing about who is right: if every tool shows the same deviation, suspect the reference, not the tools.",
        "",
        cross_tool(res, directory),
        "",
        "## Notes",
        "",
        notes.render(),
        "",
        "## Environment",
        "",
        environment(res),
        "",
    ]
    return "\n".join(parts)
