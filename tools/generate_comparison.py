"""Writes `comparison.md`: the cross-tool tables, from result JSON only.

Every timing cell carries its oracle verdict, so no speed number appears
without a correctness number next to it:

    0.234 ✓        warm solve median in ms; the oracle's residual check passed
    **0.010 ✓**    the fastest ✓ in its row
    5.087 ✗³       converged, but the solution does not solve the case (see note 3)
    FAILED¹        the tool raised; note 1 has the real exception
    ·              the tool does not read this input

Tables are split by grid (`cases.registry.GRIDS`). A transmission case read
as CGMES is a row under the case it was converted from, so each row
differing from the `.m` row shows what the input route changed.
"""
import math
import re
from pathlib import Path

import numpy as np

from cases.registry import CASES
from tools.benchmark_data import (GRID_TITLES, Results, case_size, graded, input_label, load_solution, reads,
                                  scoreboard)


class Notes:
    """Numbered footnotes, one per cause, each listing the cases it covers:
    a wrong solution per (tool, input), a failure per (tool, message)."""

    def __init__(self):
        self.cases: dict[str, dict[str, str]] = {}   # head -> {case: detail}

    def ref(self, head: str, case: str, detail: str = "") -> str:
        self.cases.setdefault(head, {}).setdefault(case, detail)
        return "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(d)] for d in str(list(self.cases).index(head) + 1))

    def render(self) -> str:
        out = []
        for i, (head, cases) in enumerate(self.cases.items(), 1):
            if any(cases.values()):
                out += [f"{i}. {head}"] + [f"    - {c}: {d}" for c, d in cases.items()]
            else:
                out.append(f"{i}. {head}: {', '.join(cases)}")
        return "\n".join(out)


def normalize_error(err: str) -> str:
    """Drops the numbers that differ per case (deviations, iteration counts),
    so one message on several cases is one note. Numbers inside identifiers
    (mRIDs) stay."""
    return re.sub(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", "…", err)


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


def failed(res: Results, notes: Notes, tool: str, case: str, operation: str) -> str:
    err = res.failure(tool, case, operation)
    if err is None:
        return "not run"
    return "FAILED" + notes.ref(f"FAILED · **{res.tools[tool]['display_name']}**: `{normalize_error(err)}`", case)


def cell(res: Results, notes: Notes, tool: str, case: str, operation: str) -> str:
    if not reads(res, tool, case):
        return "·"
    rec = res.get(tool, case, operation)
    if rec is None:
        return failed(res, notes, tool, case, operation)
    text = fmt_ms(rec.median_ms)
    if operation != "solve":
        return text
    if "oracle_ok" not in rec.extra:   # a fixture: its tier-2 deviation, no verdict
        return f"{text} · {rec.extra['sv_dv_median']:.3%}" if rec.extra.get("sv_n") else text
    if rec.extra["oracle_ok"]:
        return f"{text} ✓"
    head = f"✗ · **{res.tools[tool]['display_name']}, {input_label(case)}**"
    return f"{text} ✗{notes.ref(head, case, residual_note(rec.extra))}"


def fastest_ok(res: Results, tools: list[str], case: str) -> str | None:
    ok = [r for t in tools if (r := res.get(t, case, "solve")) and r.extra.get("oracle_ok")]
    return min(ok, key=lambda r: r.median_ms).tool if ok else None


def table(header: list[str], rows: list[list[str]], left: tuple[int, ...] = (0,)) -> str:
    """`left`: the text columns (left-aligned); the rest are numbers."""
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" if i in left else "---:" for i in range(len(header))) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def grid_tools(res: Results, cases: list[str]) -> list[str]:
    """Tools that read at least one of these cases; a column that would be all `·` is left out."""
    return [t for t in res.tool_order() if any(reads(res, t, c) for c in cases)]


def row_head(cases: list[str], c: str, with_input: bool) -> list[str]:
    """Case name and size on the first row of a case; its conversions below show only the input."""
    base = CASES[c].get("source_case", c)
    first = base == c or base not in cases
    head = [base if first else "", f"{case_size(c):,}" if first else ""]
    return head + ([input_label(c)] if with_input else [])


def grid_table(res: Results, cases: list[str], render) -> str:
    """One row per case; `render(tools, case)` gives the tool cells."""
    tools = grid_tools(res, cases)
    with_input = len({input_label(c) for c in cases}) > 1
    unit = "buses" if all(graded(c) for c in cases) else "nodes"
    header = ["case", unit] + (["input"] if with_input else []) + [res.tools[t]["display_name"] for t in tools]
    rows = [row_head(cases, c, with_input) + render(tools, c) for c in cases]
    return table(header, rows, (0, 2) if with_input else (0,))


def timing_cells(res: Results, notes: Notes, operation: str):
    def render(tools, case):
        best = fastest_ok(res, tools, case) if operation == "solve" else None
        return [f"**{text}**" if t == best else text
                for t in tools for text in [cell(res, notes, t, case, operation)]]
    return render


def per_grid(res: Results, render, graded_only: bool = False) -> str:
    parts = []
    for grid in res.grids():
        cases = res.grid_cases(grid)
        if cases and not (graded_only and not all(graded(c) for c in cases)):
            parts += [f"### {GRID_TITLES[grid]}", "", grid_table(res, cases, render), ""]
    return "\n".join(parts)


def robustness_section(res: Results, notes: Notes) -> str:
    cases = [c for g in res.grids() for c in res.grid_cases(g, robustness=True)]
    return grid_table(res, cases, timing_cells(res, notes, "solve")) if cases else "Not run."


def scoreboard_section(res: Results) -> str:
    columns, rows = scoreboard(res)
    header = [f"[{label}](#{section})" if section else label for label, section in columns]
    return table(["tool"] + header, [[res.tools[t]["display_name"]] + cells for t, cells in rows])


def memory_cells(res: Results):
    def render(tools, case):
        row = []
        for t in tools:
            if not reads(res, t, case):
                row.append("·")
                continue
            rec = res.get(t, case, "import")
            if rec is None or "rss_import_mb" not in rec.extra:
                row.append("—")
                continue
            e = rec.extra
            peak = e.get("rss_solve_mb", e["rss_import_mb"])
            row.append(f"{peak - e['rss_baseline_mb']:.0f} (+{e['rss_baseline_mb']:.0f})")
        return row
    return render


def residual_cells(res: Results):
    def render(tools, case):
        row = []
        for t in tools:
            rec = res.get(t, case, "solve")
            if not reads(res, t, case):
                row.append("·")
            elif rec is None:
                row.append("failed")
            else:
                worst = max(rec.extra["residual_max_dp_mw"], rec.extra["residual_max_dq_mvar"])
                row.append(f"{worst:.1e}" + ("" if rec.extra["oracle_ok"] else " ✗"))
        return row
    return render


def sv_cells(res: Results):
    def render(tools, case):
        row = []
        for t in tools:
            rec = res.get(t, case, "solve")
            if not reads(res, t, case):
                row.append("·")
            elif rec is None:
                row.append("failed")
            elif not rec.extra.get("sv_n"):
                row.append("n=0")
            else:
                e = rec.extra
                row.append(f"{e['sv_dv_median']:.3%} / {e['sv_dv_max']:.2%} (n={e['sv_n']}/{e['sv_n_published']})")
        return row
    return render


def accuracy_sv(res: Results) -> str:
    cases = [c for g in res.grids() for c in res.grid_cases(g) if not graded(c)]
    return grid_table(res, cases, sv_cells(res)) if cases else "Not run."


def cross_tool(res: Results, directory: Path) -> str:
    """Tier 3, the weakest evidence: each tool's largest |V| deviation from
    the per-bus median of every tool that solved the case."""
    cases = [c for g in res.grids() for c in res.grid_cases(g) if CASES[c]["format"] == "matpower"]
    tools = grid_tools(res, cases)
    rows = []
    for c in cases:
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
    # Tables first, so notes are numbered in reading order; the scoreboard needs none.
    solve = per_grid(res, timing_cells(res, notes, "solve"))
    robust = robustness_section(res, notes)
    imports = per_grid(res, timing_cells(res, notes, "import"))
    parts = [
        "# grid-bench results",
        "",
        "Generated by `tools/generate_comparison.py` from the JSON in this directory. Do not edit by hand.",
        "",
        "✓: the solution satisfies the case's power-flow equations to 1e-3 MVA at every bus and holds every "
        "generator voltage setpoint (oracle tier 1, independent of every tool). "
        "✗: the tool converged, but to a solution of a different problem. FAILED: the tool raised. "
        "`·`: the tool does not read this input. Superscripts point to the [notes](#notes); "
        "a shared note is a shared cause.",
        "",
        "## Scoreboard",
        "",
        "AC power flow on the default cases: ✓ / ✗ / FAILED per grid and input. CGMES fixtures have no verdict "
        "(their reference is someone else's solution): cases solved. Hard cases: transmission cases that are not "
        "expected to converge from a flat start, so FAILED is the normal outcome and a ✓ stands out.",
        "",
        scoreboard_section(res),
        "",
        "## AC power flow: warm solve",
        "",
        "Median of repeated solves on one persistent model, flat start every time, in ms; the fastest ✓ in each "
        "row in bold. Transmission cases are also read as CGMES converted from the `.m` (a row under the case), "
        "graded against the `.m` like the original: a row that differs from the `.m` row shows what the input "
        "route changed. CGMES fixtures: time · median |ΔV|/V against the published `SvVoltage` (tier 2).",
        "",
        solve,
        "## Robustness",
        "",
        "Cases known not to converge from a flat start in any tool tested here. Kept out of the tables above; "
        "a tool that solves one of these (and passes the oracle) is doing something the others do not.",
        "",
        robust,
        "",
        "## Import: file to model",
        "",
        "Median of 3 cold loads after one warm-up load, in ms.",
        "",
        imports,
        "## Memory",
        "",
        "Peak RSS added by loading and solving the case, in MB, measured in a fresh process; "
        "in parentheses, the peak after merely importing the tool. Charts: `graphs/memory_<family>.svg`. "
        "Only the benchmark process is measured: cgmes2pgm's Fuseki server is not included.",
        "",
        per_grid(res, memory_cells(res)),
        "## Accuracy",
        "",
        "### Tier 1: residual against the MATPOWER case (MVA, worst bus)",
        "",
        "Largest |ΔP| or |ΔQ| of `V·conj(Ybus·V) − S` over the buses where it is specified, "
        "with Ybus built from the `.m` file by `oracle/ybus.py`. Every tool was asked to converge to 1e-8 p.u. "
        "Converted cases are graded the same way: tool voltages are mapped back to MATPOWER buses by "
        "TopologicalNode name (`BUS-<n>`).",
        "",
        per_grid(res, residual_cells(res), graded_only=True).replace("### ", "#### "),
        "### Tier 2: deviation from the published CGMES solution",
        "",
        "|ΔV|/V against `SvVoltage`, median / max, and matched / published TopologicalNodes. The published "
        "solution comes from the fixture author's own tool and settings; it is a reference, not ground truth.",
        "",
        accuracy_sv(res),
        "",
        "### Converting MATPOWER to CGMES",
        "",
        "Each converter's output checked without any tool (`oracle/check_conversion.py`): Ybus, specified injections "
        "and voltage setpoints rebuilt from the CGMES files by a parser that shares no code with either converter, "
        "compared with the original `.m`. A missing slack leaves every tool to choose its own slack bus. "
        "Only exact conversions are solved by the tools (see `cases/registry.py`).",
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
        "Wrong solutions are grouped by tool and input, failures by tool and message (numbers that differ per case "
        "shown as …). Each note lists every case it covers.",
        "",
        notes.render(),
        "",
        "## Environment",
        "",
        environment(res),
        "",
    ]
    return "\n".join(parts)
