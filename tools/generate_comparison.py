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


def se_note(e: dict) -> str:
    """Why an estimate is not the WLS optimum of its measurement set (`oracle.wls`)."""
    if e["se_n_reported"] < e["se_n_buses"]:
        return f"only {e['se_n_reported']} of {e['se_n_buses']} buses reported"
    parts = [f"a Gauss-Newton step from the estimate still moves it by {e['se_max_step']:.2g} p.u./rad"]
    if e["se_J"] > e["se_J_true"]:
        parts.append(f"J = {e['se_J']:.4g} exceeds J at the true state ({e['se_J_true']:.4g})")
    return "; ".join(parts) + f" (|ΔV| from the true state up to {e['se_max_dvm_true_pu']:.2g} p.u.)"


OPF_LIMITS = (("opf_max_vm_violation_pu", "|V| outside its limits by {:.2g} p.u."),
              ("opf_max_pg_violation_mw", "generator P outside its limits by {:.2g} MW"),
              ("opf_max_qg_violation_mvar", "generator Q outside its limits by {:.2g} MVAr"),
              ("opf_max_flow_violation_mva", "branch flow over its limit by {:.2g} MVA"),
              ("opf_max_angle_violation_deg", "branch angle difference outside its limits by {:.2g} degrees"))


def opf_note(e: dict) -> str:
    """Why an OPF result is not accepted (`oracle.opf`)."""
    if e["opf_n_reported_buses"] < e["opf_n_buses"] or e["opf_n_reported_gens"] < e["opf_n_gens"]:
        return (f"only {e['opf_n_reported_buses']} of {e['opf_n_buses']} buses and {e['opf_n_reported_gens']} "
                f"of {e['opf_n_gens']} generators reported")
    parts = []
    if max(e["opf_max_dp_mw"], e["opf_max_dq_mvar"]) > 1e-3:
        parts.append(f"power balance off by {e['opf_max_dp_mw']:.2g} MW / {e['opf_max_dq_mvar']:.2g} MVAr")
    parts += [text.format(e[k]) for k, text in OPF_LIMITS if e[k] > (1e-3 if "deg" in k else 1e-5)]
    parts.append(f"cost {e['opf_gap']:+.2%} against the reference")
    return "; ".join(parts)


def batch_note(e: dict, unit: str = "scenario") -> str:
    """Why a batch is not accepted (`oracle.batch`): how many scenarios
    (or outages), then the worst one."""
    return (f"{e['scenarios_failed']} of {e['scenarios']} {unit}s fail; worst, {unit} {e['worst_scenario']}: "
            + residual_note(e))


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
    return f"{text} {verdict(res, notes, rec)}"


def verdict(res: Results, notes: Notes, rec) -> str:
    """✓, or ✗ with the note that says why, per problem."""
    if rec.extra["oracle_ok"]:
        return "✓"
    tool, case = rec.tool, rec.case
    if CASES[case]["problem"] == "se":
        head = f"✗ · **{res.tools[tool]['display_name']}, state estimation**"
        return f"✗{notes.ref(head, case, se_note(rec.extra))}"
    if CASES[case]["problem"] == "opf":
        head = f"✗ · **{res.tools[tool]['display_name']}, optimal power flow**"
        return f"✗{notes.ref(head, case, opf_note(rec.extra))}"
    if CASES[case]["problem"] == "batch":
        head = f"✗ · **{res.tools[tool]['display_name']}, batch power flow**"
        return f"✗{notes.ref(head, case, batch_note(rec.extra))}"
    if CASES[case]["problem"] == "n1":
        head = f"✗ · **{res.tools[tool]['display_name']}, N-1**"
        return f"✗{notes.ref(head, case, batch_note(rec.extra, 'outage'))}"
    head = f"✗ · **{res.tools[tool]['display_name']}, {input_label(case)}**"
    return f"✗{notes.ref(head, case, residual_note(rec.extra))}"


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


def se_cells(res: Results):
    def render(tools, case):
        row = []
        for t in tools:
            rec = res.get(t, case, "solve")
            if not reads(res, t, case):
                row.append("·")
            elif rec is None:
                row.append("failed")
            elif rec.extra["se_n_reported"] < rec.extra["se_n_buses"]:
                row.append(f"{rec.extra['se_n_reported']}/{rec.extra['se_n_buses']} buses ✗")
            else:
                e = rec.extra
                ratio = e["se_J"] / e["se_J_true"] if e["se_J_true"] > 1e-9 else math.nan
                j = f"J/J* {ratio:.3f}" if math.isfinite(ratio) else f"J {e['se_J']:.0e}"
                row.append(f"{e['se_max_step']:.0e} · {j} · {e['se_max_dvm_true_pu']:.0e}"
                           + ("" if e["oracle_ok"] else " ✗"))
        return row
    return render


def se_section(se: Results, notes: Notes) -> str:
    """State estimation: the same tables as power flow, graded by `oracle.wls`."""
    solve = per_grid(se, timing_cells(se, notes, "solve")).replace("### ", "#### ")
    imports = per_grid(se, timing_cells(se, notes, "import")).replace("### ", "#### ")
    return "\n".join([
        "## State estimation (WLS)",
        "",
        "Each case is a MATPOWER case with a measurement set generated from its own power flow "
        "(`cases/measurements.py`): `~exact`, |V| and P/Q injections at every bus without noise, whose optimum "
        "is the true state; `~noisy`, |V| at generator buses, P/Q injections at every bus and P/Q flows at the "
        "from end of every branch, with Gaussian noise. Every tool gets the same measurements with the same "
        "sigmas, flat start, no bad-data handling. ✓: the estimate is the weighted least-squares optimum of "
        "that measurement set (a Gauss-Newton step from it moves it by at most 1e-6 p.u./rad, and its J is "
        "no larger than J at the true state; `oracle/wls.py`, independent of every tool).",
        "",
        "### Scoreboard",
        "",
        scoreboard_section(se),
        "",
        "### Warm estimate",
        "",
        "Median of repeated estimates on one persistent model, flat start every time, in ms; the fastest ✓ in "
        "each row in bold.",
        "",
        solve,
        "### Import: file to model, with measurements",
        "",
        imports,
        "### Accuracy",
        "",
        "Largest entry of one Gauss-Newton step from the estimate (p.u./rad; 0 at the optimum) · J over J at "
        "the true state (below 1 with noise: the optimum fits the measurements better than the truth; "
        "`~exact` cases show J itself) · largest |ΔV| from the true state (p.u., information only).",
        "",
        per_grid(se, se_cells(se)).replace("### ", "#### "),
        "### Environment",
        "",
        environment(se),
        "",
    ])


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


def opf_cells(res: Results):
    def render(tools, case):
        row = []
        for t in tools:
            rec = res.get(t, case, "solve")
            if not reads(res, t, case):
                row.append("·")
            elif rec is None:
                row.append("failed")
            elif math.isnan(rec.extra["opf_gap"]):
                row.append("incomplete ✗")
            else:
                e = rec.extra
                worst = max(max(e["opf_max_dp_mw"], e["opf_max_dq_mvar"]) / 100, e["opf_max_vm_violation_pu"],
                            e["opf_max_pg_violation_mw"] / 100, e["opf_max_qg_violation_mvar"] / 100,
                            e["opf_max_flow_violation_mva"] / 100)
                row.append(f"{e['opf_gap']:+.1e} · {worst:.0e}" + ("" if e["oracle_ok"] else " ✗"))
        return row
    return render


def opf_section(opf: Results, notes: Notes) -> str:
    """AC optimal power flow: the same tables, graded by `oracle.opf`."""
    solve = per_grid(opf, timing_cells(opf, notes, "solve")).replace("### ", "#### ")
    imports = per_grid(opf, timing_cells(opf, notes, "import")).replace("### ", "#### ")
    return "\n".join([
        "## AC optimal power flow",
        "",
        "PGLib-OPF v23.07 cases: typical operating conditions from 14 to 2,869 buses, and the congested (`__api`) "
        "and small angle-difference (`__sad`) variants of case14, case118 and case300. Every tool solves MATPOWER's "
        "AC-OPF: polynomial cost, the power-flow equations, voltage, generator, branch MVA and angle-difference "
        "limits, flat start, tolerance 1e-6. ✓: the solution is feasible for the case (power balance within 1e-3 "
        "MVA, every limit within 1e-5 p.u. or 1e-3 degrees) and its cost, recomputed from the case, is at most "
        "0.01 % above PGLib's published reference (`oracle/opf.py`, independent of every tool). The reference is a "
        "local optimum given to five digits.",
        "",
        "### Scoreboard",
        "",
        scoreboard_section(opf),
        "",
        "### Warm solve",
        "",
        "Median of repeated solves on one persistent model, flat start every time, in ms; the fastest ✓ in each "
        "row in bold.",
        "",
        solve,
        "### Import: file to model",
        "",
        imports,
        "### Accuracy",
        "",
        "Cost against the reference (relative; negative is cheaper, which a solution violating a limit can be) · "
        "largest violation of balance or a limit, in p.u. of 100 MVA.",
        "",
        per_grid(opf, opf_cells(opf)).replace("### ", "#### "),
        "### Environment",
        "",
        environment(opf),
        "",
    ])


def per_scenario_cells(res: Results, notes: Notes, fastest: bool):
    """Median time of the whole batch over its scenarios, ms per scenario,
    with the verdict. `fastest`: at the thread count where it is lowest, and
    which (`@8`); otherwise at one thread. Bold: the fastest ✓ in the row."""
    def best(t, case):
        recs = [res.get(t, case, "solve", n) for n in res.thread_counts(t, case)] if fastest else []
        return min(recs, key=lambda r: r.median_ms, default=None) or res.get(t, case, "solve")

    def render(tools, case):
        recs = {t: best(t, case) for t in tools if reads(res, t, case)}
        ok = [r for r in recs.values() if r and r.extra["oracle_ok"]]
        top = min(ok, key=lambda r: r.median_ms).tool if ok else None
        row = []
        for t in tools:
            rec = recs.get(t)
            if not reads(res, t, case):
                row.append("·")
                continue
            if rec is None:
                row.append(failed(res, notes, t, case, "solve"))
                continue
            text = fmt_ms(rec.median_ms / rec.extra["scenarios"]) + (f" @{rec.extra['threads']}" if fastest else "")
            row.append(f"{f'**{text}**' if t == top else text} {verdict(res, notes, rec)}")
        return row
    return render


def scaling_table(res: Results) -> str:
    """Speedup over the same tool's one-thread batch, per tool whose batch
    API takes a thread count, and case; ✗ where that thread count's verdict
    differs from the one-thread run's."""
    counts = sorted({n for r in res.records if r.operation == "solve" for n in [r.extra.get("threads")] if n})
    rows = []
    for grid in res.grids():
        for c in res.grid_cases(grid):
            for t in res.tool_order():
                one = res.get(t, c, "solve", 1)
                if one is None or len(res.thread_counts(t, c)) < 2:   # loops and one-thread batch APIs
                    continue
                cells = []
                for n in counts:
                    r = res.get(t, c, "solve", n)
                    cells.append("" if r is None else f"{one.median_ms / r.median_ms:.2f}×"
                                 + ("" if r.extra["oracle_ok"] == one.extra["oracle_ok"] else " ✗"))
                rows.append([c.partition("#")[0], f"{case_size(c):,}", res.tools[t]["display_name"]] + cells)
    if not rows:
        return "No native batch results."
    return table(["case", "buses", "tool"] + [f"{n} thread{'s' if n > 1 else ''}" for n in counts], rows, (0, 2))


def cpus(res: Results) -> str:
    c = next((r["cpus"] for r in res.runs if r.get("cpus")), {})
    return (f"{c.get('available', '?')} of {c.get('logical', '?')} logical CPUs available to the run "
            f"({c.get('model', '?')})")


BATCH_TEXT = {
    "batch": {
        "title": "Batch power flow", "unit": "scenario",
        "intro": "Each case is a MATPOWER case and 100 operating points on its topology (`cases/sweep.py`): every "
                 "bus's demand scaled along one period of a daily curve between 60 % and 100 % of the case, with 5 % "
                 "noise per bus, generators redispatched in proportion. Every tool solves all of them in one timed "
                 "call, each scenario the power-flow problem above (flat start, tolerance 1e-8 p.u.), through its "
                 "batch API where it has one",
        "loop": "so the call is a loop of the tool's warm single solve after writing the scenario into its model",
        "verdict": "✓: every scenario's solution satisfies the case with that scenario's demand, at every bus (tier 1 "
                   "per scenario, `oracle/batch.py`).",
        "time": "Median time of the whole batch divided by its 100 scenarios, in ms; the fastest ✓ in each row in "
                "bold. Compare with the warm single solve above: the difference is what the batch API saves (or a "
                "loop adds).",
    },
    "n1": {
        "title": "N-1 contingency analysis", "unit": "outage",
        "intro": "Each transmission case with 200 single-branch outages that keep the grid connected and whose power "
                 "flow converges from the base case (`cases/contingency.py`; all 19 such of case14), chosen in a "
                 "seeded order. Every tool solves the base case and every outage in one timed call, each outage the "
                 "power-flow problem above with that branch out of service, started from the tool's own solution "
                 "of the base case (the one exception to the flat start, as contingency analysis is done; "
                 "power-grid-model takes no start voltages and starts each outage flat), through its contingency "
                 "API where it has one",
        "loop": "so the call is a loop of the tool's single solve with the branch taken out",
        "verdict": "✓: every outage's solution satisfies the case with that branch out of service, at every bus "
                   "(tier 1 per outage, `oracle/batch.py`).",
        "time": "Median time of the whole call (base case included) divided by its number of outages, in ms; the "
                "fastest ✓ in each row in bold.",
    },
}


def batch_section(batch: Results, notes: Notes, problem: str = "batch") -> str:
    """Batch power flow (one sweep of operating points per case) or N-1 (the
    outages of a case): the same tables, graded per scenario by `oracle.batch`."""
    text = BATCH_TEXT[problem]
    unit = text["unit"]
    loop = [batch.tools[t]["display_name"] for t in batch.tool_order() if batch.tools[t]["settings"].get("mode") == "loop"]
    return "\n".join([
        f"## {text['title']}",
        "",
        text["intro"] + ("; " + f"{', '.join(loop)} {'has' if len(loop) == 1 else 'have'} none, " + text["loop"]
                         + " (`loop`). " if loop else ". ") + text["verdict"],
        "",
        "### Scoreboard",
        "",
        scoreboard_section(batch),
        "",
        f"### Time per {unit}, one thread",
        "",
        text["time"],
        "",
        per_grid(batch, per_scenario_cells(batch, notes, fastest=False)).replace("### ", "#### "),
        f"### Time per {unit}, fastest thread count",
        "",
        f"As above, at the thread count where each tool is fastest (`@n`). {cpus(batch)}; a loop runs on one.",
        "",
        per_grid(batch, per_scenario_cells(batch, notes, fastest=True)).replace("### ", "#### "),
        "### Thread scaling",
        "",
        f"Speedup of each batch API that takes a thread count over its own one-thread run, same case, same "
        f"{unit}s. Each thread count's solution is graded on its own; ✗ where its verdict differs from the "
        "one-thread run's.",
        "",
        scaling_table(batch),
        "",
        f"### Import: file to model, with {unit}s",
        "",
        per_grid(batch, timing_cells(batch, notes, "import")).replace("### ", "#### "),
        "### Environment",
        "",
        environment(batch),
        "",
    ])


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
    se = se_section(res.se, notes) if res.se and (res.se.records or res.se.failures) else ""
    opf = opf_section(res.opf, notes) if res.opf and (res.opf.records or res.opf.failures) else ""
    batch = batch_section(res.batch, notes) if res.batch and (res.batch.records or res.batch.failures) else ""
    n1 = batch_section(res.n1, notes, "n1") if res.n1 and (res.n1.records or res.n1.failures) else ""
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
        "(their reference is someone else's solution): cases solved. Hard transmission cases: cases that are not "
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
        "## Hard transmission cases",
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
        "in parentheses, the peak after merely importing the tool. Charts: the site (`docs/index.html`). "
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
        *([se] if se else []),
        *([opf] if opf else []),
        *([batch] if batch else []),
        *([n1] if n1 else []),
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
