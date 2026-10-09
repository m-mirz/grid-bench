"""Writes `docs/index.html`: one self-contained page (GitHub Pages), from
result JSON only. Data is embedded; the chart and tables are drawn in the
browser so the filters apply to both. No external requests.
"""
import html
import json
from pathlib import Path

from cases.registry import CASES
from oracle.evaluate import VM_FLOOR_PU
from tools.benchmark_data import GRID_TITLES, Results, case_size, graded, input_label, scoreboard
from tools.palette import DARK, DASH, LIGHT, LIGHT_TO_DARK

KEEP = ("iterations", "oracle_ok", "residual_max_dp_mw", "residual_max_dq_mvar", "residual_max_dvm_pu",
        "residual_worst_bus", "residual_min_vm_pu", "residual_n_checked", "residual_n_buses", "sv_n", "sv_n_published",
        "sv_dv_median", "sv_dv_max", "sv_da_max_deg", "rss_baseline_mb", "rss_import_mb", "rss_solve_mb",
        "rss_export_mb", "rss_validate_mb",
        "se_J", "se_J_true", "se_max_step", "se_max_dvm_true_pu", "se_n_reported", "se_n_buses",
        "opf_gap", "opf_objective", "opf_max_dp_mw", "opf_max_dq_mvar", "opf_max_vm_violation_pu",
        "opf_max_pg_violation_mw", "opf_max_qg_violation_mvar", "opf_max_flow_violation_mva",
        "opf_max_angle_violation_deg", "opf_n_buses", "opf_n_reported_buses",
        "threads", "scenarios", "scenarios_failed", "worst_scenario",
        "read", "unit", "lines", "generators", "loads", "substations", "files_written", "bytes_written",
        "violations", "by_severity")
OPF_CONDITION = {"": "typical", "api": "congested", "sad": "small angle difference"}


def _input(case: str) -> str:
    """What the page's input selector offers: the input format for power flow,
    the measurement scenario for state estimation (one line per scenario:
    `~exact` and `~noisy` of a case are the same size, different problems),
    the operating condition for OPF (typical, `__api`, `__sad`)."""
    c = CASES[case]
    if c["problem"] == "se":
        return f"{c['scenario']} measurements"
    if c["problem"] == "opf":
        return OPF_CONDITION[case.partition("__")[2]]
    return input_label(case).strip("`")


def payload(res: Results) -> dict:
    tools = [{"name": t, "display": m["display_name"], "color": m["color"],
              "colorDark": LIGHT_TO_DARK.get(m["color"], m["color"]), "dash": DASH.get(m["color"]),
              "version": m["version"], "tags": m["tags"],
              "language": m["language"], "families": m["families"], "settings": m["settings"]}
             for t in res.tool_order() for m in [res.tools[t]]]
    # `order`: the report's row order (by size, each conversion under its source case).
    ordered = [c for g in res.grids() for rob in (False, True) for c in res.grid_cases(g, rob)]
    cases = {c: {"family": CASES[c]["family"], "grid": CASES[c]["grid"], "input": _input(c),
                 "base": CASES[c].get("source_case", CASES[c].get("base_case", c.partition("__")[0])),
                 "graded": graded(c), "order": i, "size": case_size(c),
                 "groups": CASES[c]["groups"], "source": CASES[c]["source"], "note": CASES[c]["note"]}
             for i, c in enumerate(ordered)}
    rows = [{"tool": r.tool, "case": r.case, "op": r.operation, "median": r.median_ms, "min": r.min_ms,
             "rounds": r.rounds, **{k: r.extra[k] for k in KEEP if k in r.extra}} for r in res.records]
    run = next((r for r in res.runs if r["machine"]), {})
    cpu = run.get("machine", {}).get("cpu", {})
    grids = [[g, {"transmission": "Transmission", "distribution": "Distribution", "fixtures": "CGMES fixtures"}[g],
              GRID_TITLES[g]] for g in res.grids()]
    columns, board = scoreboard(res)
    return {"tools": tools, "cases": cases, "rows": rows, "failures": res.failures, "grids": grids,
            "scoreboard": {"columns": [label.replace("`", "") for label in columns], "rows": board},
            "run": {"cpu": cpu.get("brand_raw", "?"), "cores": cpu.get("count", "?"),
                    "cpus": next((r["cpus"] for r in res.runs if r.get("cpus")), {}),
                    "os": f"{run.get('machine', {}).get('system', '?')} {run.get('machine', {}).get('release', '')}",
                    "git": str(run.get("git_sha", "?"))[:12], "date": str(run.get("datetime", "?"))[:10]}}


def generate(directory: Path, res: Results) -> str:
    part = lambda r: payload(r) if r and (r.records or r.failures) else None
    data = json.dumps({"pf": payload(res), "se": part(res.se), "opf": part(res.opf), "batch": part(res.batch),
                       "n1": part(res.n1), "cim": part(res.cim)},
                      separators=(",", ":")).replace("</", "<\\/")
    return (TEMPLATE.replace("__DATA__", data).replace("__VM_FLOOR__", repr(VM_FLOOR_PU))
            .replace("__LIGHT__", "".join(f"--{k}:{v};" for k, v in LIGHT.items()))
            .replace("__DARK__", "".join(f"--{k}:{v};" for k, v in DARK.items()))
            .replace("__TITLE__", html.escape("grid-bench")))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;__LIGHT__--accent:#2a78d6;--ok:#008300;--bad:#c4312f;--chip:#efeeea;--border:#dcdbd6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;__DARK__--accent:#3987e5;--ok:#3fae3f;--bad:#e66767;--chip:#2a2a27;--border:#3a3a36}}
:root[data-theme="dark"]{color-scheme:dark;__DARK__--accent:#3987e5;--ok:#3fae3f;--bad:#e66767;--chip:#2a2a27;--border:#3a3a36}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:20px;margin:40px 0 8px}
p,li{color:var(--text2);max-width:78ch}
.lede{font-size:17px;margin:0 0 20px}
.meta{font-size:13px;color:var(--text2)}
.controls{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;margin:24px 0 12px}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{border:0;background:transparent;color:var(--text2);padding:6px 12px;font:inherit;font-size:14px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--chip);color:var(--text);font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--border);border-radius:999px;padding:3px 10px;background:transparent;color:var(--text);font:inherit;font-size:13px;cursor:pointer}
.chip[aria-pressed="false"]{opacity:.45}
.chip i{width:10px;height:10px;border-radius:50%;display:inline-block}
.card{border:1px solid var(--border);border-radius:12px;padding:16px;background:var(--surface)}
#chart{width:100%;height:auto;display:block}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:13px;box-shadow:0 4px 16px rgba(0,0,0,.15);max-width:320px}
.tip b{display:block;color:var(--text)}.tip span{color:var(--text2)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-size:14px;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--text2);font-weight:600;cursor:pointer;user-select:none}
th:hover{color:var(--text)}
td.ok::after{content:" ✓";color:var(--ok)}
td.bad{color:var(--bad)}td.bad::after{content:" ✗"}
td.fail{color:var(--bad);font-size:12px}
td.na{color:var(--axis)}
td.best{font-weight:700}
.seg[hidden]{display:none}
.legend-note{font-size:13px;color:var(--text2);margin:8px 0 0}
.multiples{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px;margin-top:12px}
.multiples .card{padding:10px 12px}.multiples h3{font-size:14px;margin:0 0 4px}.multiples svg{width:100%;height:auto;display:block}
#s-legend .chip{cursor:default}
details{margin:6px 0}summary{cursor:pointer;color:var(--text)}
code{font-size:13px;background:var(--chip);padding:1px 5px;border-radius:4px}
</style>
</head>
<body>
<main>
<h1>grid-bench</h1>
<p class="lede">Power-flow speed of open-source power system tools, where every timing is graded by an oracle that no tool under test takes part in. MATPOWER cases are also converted to CGMES, and tools are graded on those against the original case. State estimation (weighted least squares), AC optimal power flow, batch power flow (a sweep of operating points, through each tool's batch API on as many threads as it can use) and N-1 contingency analysis are benchmarked the same way, and CIM libraries on reading, writing and validating CGMES (timed only): switch below.</p>
<p class="meta" id="meta"></p>

<div class="controls"><div class="seg" role="group" aria-label="Problem" id="problem"></div></div>
<h2>Scoreboard</h2>
<p id="sb-desc">AC power flow on the default cases: ✓ / ✗ / FAILED per grid and input. CGMES fixtures have no verdict (their reference is someone else's solution): cases solved. Hard transmission cases: cases that are not expected to converge from a flat start, so FAILED is the normal outcome and a ✓ stands out; they are in the Transmission tables, below the others.</p>
<div class="scroll"><table id="scoreboard"></table></div>

<div class="controls">
  <div class="seg" role="group" aria-label="Operation" id="op"></div>
  <div class="seg" role="group" aria-label="Grid" id="grid"></div>
  <div class="seg" role="group" aria-label="Input plotted" id="input"></div>
  <div class="seg" role="group" aria-label="Threads" id="threads"></div>
  <div class="seg" role="group" aria-label="Memory of" id="memop"></div>
  <div class="chips" id="toolchips" aria-label="Tools"></div>
</div>
<div class="card"><svg id="chart" viewBox="0 0 960 440" role="img" aria-label="Time versus case size, one line per tool"></svg>
<p class="legend-note" id="legend-note"></p></div>

<h2 id="t-title"></h2>
<p id="t-desc"></p>
<div class="scroll"><table id="timing"></table></div>

<section id="scaling-sec" hidden>
<h2>Thread scaling</h2>
<p id="s-desc"></p>
<div class="chips" id="s-legend"></div>
<div class="multiples" id="scaling-plots"></div>
<p class="legend-note">One panel per case, same y-axis. Dashed grey: ideal scaling (speedup equal to the thread count). Filled point: the oracle verified that thread count's solution; hollow: it did not. Hover a point for the time.</p>
<details><summary>As a table</summary><div class="scroll"><table id="scaling"></table></div></details>
</section>

<h2 id="a-title">Accuracy</h2>
<p id="a-desc"></p>
<div class="scroll"><table id="accuracy"></table></div>

<h2>Failures</h2>
<p>Every run that raised, with the tool's own exception.</p>
<div class="scroll"><table id="failures"></table></div>

<h2>How this is measured</h2>
<ul>
<li><b>Same problem for every tool.</b> Flat start on every solve, one slack, no reactive limits, no tap/phase-shifter/shunt control, generator voltage regulation as the case defines it, convergence tolerance 1e-8 p.u. Each adapter's docstring justifies its settings.</li>
<li><b>Solve</b> is timed warm: one persistent model, one untimed warm-up solve, then repeated solves (median shown). <b>Import</b> is the median of 3 cold loads after one warm-up load.</li>
<li><b>Tier 1 oracle</b> (MATPOWER): the residual <code>V·conj(Ybus·V) − S</code> with Ybus built from the <code>.m</code> file by the benchmark itself. <b>Tier 2</b> (CGMES): deviation from the case's published SV solution. <b>Tier 3</b>: tools against each other, the weakest evidence.</li>
<li><b>State estimation</b>: weighted least squares on each case with a measurement set generated from its own power flow. <code>exact</code>: |V| and P/Q injections at every bus without noise, so the optimum is the true state. <code>noisy</code>: |V| at generator buses, P/Q injections at every bus, P/Q flows at every branch's from end, with Gaussian noise. Same measurements and sigmas for every tool, flat start, no bad-data handling. The oracle accepts an estimate when one Gauss-Newton step from it moves it by at most 1e-6 and its J is no larger than J at the true state.</li>
<li><b>Optimal power flow</b>: MATPOWER's AC-OPF on PGLib-OPF v23.07 cases (polynomial cost; power-flow equations; voltage, generator, branch MVA and angle-difference limits), flat start, tolerance 1e-6. The oracle checks the solution's feasibility against the <code>.m</code> and recomputes its cost, accepted at most 0.01% above PGLib's published reference (a local optimum given to five digits).</li>
<li><b>Batch power flow</b>: each case with 100 operating points (every bus's demand along a daily curve between 60% and 100% of the case, 5% noise per bus, generators redispatched in proportion), all solved in one timed call, each scenario the power-flow problem above. Tools with a batch API run it at 1, 2, 4, … threads, up to every core available; a tool without one runs a loop of its warm single solve, on one thread. Times are per scenario. The oracle grades every scenario of every thread count.</li>
<li><b>N-1 contingencies</b>: each transmission case with 200 single-branch outages that keep the grid connected (all 19 of case14), solved in one timed call with the base case, each outage started from the tool's own base-case solution (power-grid-model takes no start voltages and starts flat). Contingency APIs that take a thread count run at 1, 2, 4, … threads. Times are per outage. The oracle grades every outage of every thread count against the case with that branch out.</li>
<li><b>CIM import/export</b>: CIM libraries on the Svedala and RealGrid CGMES 3.0 conformity models, every profile (SV included) read from the published XML. <b>Import</b>: files to the library's model. <b>Export</b>: that model back to CGMES RDF/XML, warm, into an empty directory. <b>Validate</b>: files to violation report in one call, parsing included (cimoxide validates files only); cimoxide runs its own rules, triplets and OpenCGMES (with Jena SHACL) the ENTSO-E CGMES 3.0 SHACL shapes. These are timed, not graded, as in cim-bench: what each tool read and found is shown, but the tools read CGMES into different models and validate against different rules. The Java libraries run in-process through JPype, JVM included in memory.</li>
<li><b>Memory</b> is the peak RSS of a fresh process loading and solving the case, minus the peak after importing the tool (values below 1 MB are drawn at 1 MB on the log axis). Only the benchmark process counts: cgmes2pgm's Fuseki server is not included.</li>
</ul>
<div class="tip" id="tip" hidden></div>
</main>
<script>
const ALL = __DATA__;
const VM_FLOOR = __VM_FLOOR__;
let D = ALL.pf;
const state = {problem: "pf", op: "solve", grid: (D.grids[0] || ["transmission"])[0], input: null, threads: "1", memop: "import", off: new Set()};
const se = () => state.problem === "se";
const cim = () => state.problem === "cim";
const OP_NAME = {solve: "Solve", import: "Import", export: "Export", validate: "Validate", memory: "Memory"};
const validator = t => (t.tags || []).includes("validator");
// The oracle's verdict detail for a record, per problem.
const opfVerdict = r => r.opf_gap === undefined || r.opf_gap === null
  ? `only ${r.opf_n_reported_buses} of ${r.opf_n_buses} buses reported`
  : `cost ${(r.opf_gap * 100).toFixed(4)}% against the reference · balance ${Math.max(r.opf_max_dp_mw, r.opf_max_dq_mvar).toExponential(1)} MVA · |V| ${r.opf_max_vm_violation_pu.toExponential(1)} p.u. · P/Q ${Math.max(r.opf_max_pg_violation_mw, r.opf_max_qg_violation_mvar).toExponential(1)} · flow ${r.opf_max_flow_violation_mva.toExponential(1)} MVA · angle ${r.opf_max_angle_violation_deg.toExponential(1)}° over its limit`;
const verdict = r => r.scenarios !== undefined ? `${r.scenarios_failed} of ${r.scenarios} scenarios fail (worst: scenario ${r.worst_scenario}) · ` + pfVerdict(r)
  : r.opf_n_buses !== undefined ? opfVerdict(r) : r.se_J !== undefined
  ? `step to the WLS optimum ${r.se_max_step.toExponential(1)} · J ${r.se_J.toPrecision(4)} (at the truth ${r.se_J_true.toPrecision(4)}) · max |ΔV| from the truth ${r.se_max_dvm_true_pu.toExponential(1)} p.u.`
  : r.se_n_buses !== undefined ? `only ${r.se_n_reported} of ${r.se_n_buses} buses reported`
  : pfVerdict(r);
function pfVerdict(r) { return `max |ΔP| ${r.residual_max_dp_mw.toExponential(2)} MW, |ΔQ| ${r.residual_max_dq_mvar.toExponential(2)} MVAr, |ΔV| setpoint ${r.residual_max_dvm_pu.toExponential(1)} p.u.${r.residual_min_vm_pu < VM_FLOOR ? `, lowest |V| ${r.residual_min_vm_pu.toFixed(3)} p.u. (low-voltage root)` : ""}, worst bus ${r.residual_worst_bus}`; }
const gridTitle = g => (D.grids.find(x => x[0] === g) || [g, g, g])[2];
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const dark = () => document.documentElement.dataset.theme === "dark" ||
  (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
const color = t => dark() ? t.colorDark : t.color;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = ms => ms < 10 ? ms.toFixed(3) : ms < 1000 ? ms.toFixed(1) : Math.round(ms).toLocaleString();
// A batch case has a solve record per thread count: one thread, or the fastest.
function row(t, c, op) {
  const rs = D.rows.filter(r => r.tool === t && r.case === c && r.op === op);
  if (rs.length < 2) return rs[0];
  return state.threads === "1" ? rs.find(r => r.threads === 1) : rs.reduce((a, b) => a.median <= b.median ? a : b);
}
const batch = () => state.problem === "batch" || state.problem === "n1";   // per-scenario times, a thread axis
const ms = r => r.scenarios ? r.median / r.scenarios : r.median;   // batch: per scenario
const fail = (t, c, op) => D.failures.find(f => f.tool === t && f.case === c && f.operation === op);
const casesOf = g => Object.keys(D.cases).filter(c => D.cases[c].grid === g).sort((a, b) => D.cases[a].order - D.cases[b].order);
const inputsOf = g => [...new Set(casesOf(g).map(c => D.cases[c].input))];
const reads = (t, c) => t.families.includes(D.cases[c].family);
const graded = cases => cases.every(c => D.cases[c].graded);   // tier-1 residual against a MATPOWER case
const verified = r => r.op !== "solve" || r.oracle_ok === undefined || r.oracle_ok;
const MEM_FLOOR = 1;   // MB; log axis
// Memory a record holds: power flow and the rest, the solve's peak if it
// solved, else the import's; CIM, the peak of the operation chosen (export's
// is over loading plus one export; validate's is a record of its own).
const MEM_KEY = {import: "rss_import_mb", export: "rss_export_mb", validate: "rss_validate_mb"};
const MEM_NAME = {import: "Import", export: "Import + export", validate: "Validate"};
const memKey = r => cim() ? MEM_KEY[state.memop] : r.rss_solve_mb !== undefined ? "rss_solve_mb" : "rss_import_mb";
const memAdded = r => r && r[memKey(r)] !== undefined ? r[memKey(r)] - r.rss_baseline_mb : undefined;
const validating = () => state.op === "validate" || (cim() && state.op === "memory" && state.memop === "validate");
// The record a view reads, and the value it plots: memory lives on the import record.
const recOp = () => state.op !== "memory" ? state.op : validating() ? "validate" : "import";
const rec = (t, c) => row(t, c, recOp());
const val = r => state.op === "memory" ? memAdded(r) : ms(r);

$("#meta").textContent = `Run ${D.run.date} · commit ${D.run.git} · ${D.run.cpu}, ${D.run.cores} logical CPUs · ${D.run.os}`;

function seg(el, key, options) {
  el.innerHTML = options.map(([v, l]) => `<button data-v="${v}" aria-pressed="${state[key] === v}">${l}</button>`).join("");
  el.onclick = e => { const b = e.target.closest("button"); if (!b) return; state[key] = b.dataset.v; render(); };
}
// A tool's colour key: a dot, or a dashed stroke for a tool drawn dashed.
const swatch = t => `<i style="${t.dash ? `width:16px;height:3px;border-radius:0;background:repeating-linear-gradient(90deg,${color(t)} 0 ${t.dash.split(" ")[0]}px,transparent ${t.dash.split(" ")[0]}px ${t.dash.split(" ").reduce((a, b) => a + +b, 0)}px)` : `background:${color(t)}`}"></i>`;
function chips() {
  $("#toolchips").innerHTML = D.tools.map(t => `<button class="chip" data-t="${t.name}" aria-pressed="${!state.off.has(t.name)}">${swatch(t)}${esc(t.display)}</button>`).join("");
}
$("#toolchips").onclick = e => { const b = e.target.closest(".chip"); if (!b) return;
  state.off.has(b.dataset.t) ? state.off.delete(b.dataset.t) : state.off.add(b.dataset.t); render(); };

function chart() {
  const svg = $("#chart"), W = 960, H = 440, m = {l: 64, r: 200, t: 20, b: 48};
  // Plotted: smoke and scaling cases, and the variants of those (an OPF case's
  // congested or small angle-difference version is the same grid, so it
  // extends no line across grid families; the feature cases that would are
  // in the tables only).
  const plotted = c => (D.cases[D.cases[c].base] || D.cases[c]).groups.some(g => g === "smoke" || g === "scaling");
  const cases = casesOf(state.grid).filter(c => D.cases[c].input === state.input && plotted(c));
  const tools = D.tools.filter(t => !state.off.has(t.name) && cases.some(c => reads(t, c)));
  const series = tools.map(t => ({t, pts: cases.map(c => ({c, r: rec(t.name, c)})).filter(p => p.r && val(p.r) !== undefined)
    .map(p => ({x: D.cases[p.c].size, y: state.op === "memory" ? Math.max(val(p.r), MEM_FLOOR) : val(p.r),
                ok: state.op === "memory" ? cim() || p.r.rss_solve_mb !== undefined : verified(p.r), c: p.c, r: p.r}))})).filter(s => s.pts.length);
  const all = series.flatMap(s => s.pts);
  if (!all.length) { svg.innerHTML = `<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="${css("--text2")}">No results for this selection</text>`; return; }
  const lg = Math.log10, x0 = Math.floor(lg(Math.min(...all.map(p => p.x)))), x1 = Math.ceil(lg(Math.max(...all.map(p => p.x))));
  const y0 = Math.floor(lg(Math.min(...all.map(p => p.y)))), y1 = Math.ceil(lg(Math.max(...all.map(p => p.y))));
  const X = v => m.l + (lg(v) - x0) / Math.max(1, x1 - x0) * (W - m.l - m.r);
  const Y = v => H - m.b - (lg(v) - y0) / Math.max(1, y1 - y0) * (H - m.t - m.b);
  let s = "";
  for (let e = y0; e <= y1; e++) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(10**e)}" y2="${Y(10**e)}" stroke="${css("--grid")}"/><text x="${m.l - 8}" y="${Y(10**e) + 4}" text-anchor="end" font-size="12" fill="${css("--text2")}">${10**e >= 1 ? (10**e).toLocaleString() : 10**e}</text>`;
  for (let e = x0; e <= x1; e++) s += `<line x1="${X(10**e)}" x2="${X(10**e)}" y1="${m.t}" y2="${H - m.b}" stroke="${css("--grid")}"/><text x="${X(10**e)}" y="${H - m.b + 18}" text-anchor="middle" font-size="12" fill="${css("--text2")}">${(10**e).toLocaleString()}</text>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 8}" text-anchor="middle" font-size="12" fill="${css("--text2")}">${graded(cases) ? "buses" : "published nodes"}</text>`;
  s += `<text transform="translate(16 ${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="${css("--text2")}">${state.op === "memory" ? "MB added over import baseline" : `median ${state.op} time${batch() && state.op === "solve" ? (state.problem === "n1" ? " per outage" : " per scenario") : ""} (ms)`}</text>`;
  const labels = [];
  for (const {t, pts} of series) {
    const c = color(t);
    s += `<polyline fill="none" stroke="${c}" stroke-width="2"${t.dash ? ` stroke-dasharray="${t.dash}"` : ""} stroke-linejoin="round" stroke-linecap="round" points="${pts.map(p => `${X(p.x)},${Y(p.y)}`).join(" ")}"/>`;
    for (const p of pts) s += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="4.5" fill="${p.ok ? c : css("--surface")}" stroke="${c}" stroke-width="2"/>`;
    const last = pts[pts.length - 1]; labels.push({y: Y(last.y), name: t.display});
  }
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 16);
  for (const l of labels) s += `<text x="${W - m.r + 10}" y="${l.y + 4}" font-size="12" fill="${css("--text")}">${esc(l.name)}</text>`;
  const hits = series.flatMap(({t, pts}) => pts.map(p => ({t, p, cx: X(p.x), cy: Y(p.y)})));
  s += hits.map((h, i) => `<circle cx="${h.cx}" cy="${h.cy}" r="12" fill="transparent" data-i="${i}"/>`).join("");
  svg.innerHTML = s;
  svg.onmousemove = e => { const i = e.target.dataset?.i; const tip = $("#tip");
    if (i === undefined) { tip.hidden = true; return; }
    const {t, p} = hits[+i], r = p.r;
    let acc = "";
    if (r.op === "solve" && r.oracle_ok !== undefined) acc = (r.oracle_ok ? "oracle: verified · " : "oracle: FAILED · ") + verdict(r);
    else if (r.sv_n) acc = `vs published SV: median ${(r.sv_dv_median * 100).toFixed(3)}%, max ${(r.sv_dv_max * 100).toFixed(2)}%`;
    const what = state.op === "memory"
      ? cim() ? `${MEM_NAME[state.memop]}: +${memAdded(r).toFixed(1)} MB peak over a ${Math.round(r.rss_baseline_mb)} MB baseline`
      : `+${memAdded(r).toFixed(1)} MB peak over a ${Math.round(r.rss_baseline_mb)} MB baseline (import peak +${(r.rss_import_mb - r.rss_baseline_mb).toFixed(1)} MB)`
      : r.scenarios ? `${fmt(ms(r))} ms per scenario on ${r.threads} thread${r.threads > 1 ? "s" : ""} (batch of ${r.scenarios}: median ${fmt(r.median)} ms, ${r.rounds} rounds)`
      : `median ${fmt(r.median)} ms (min ${fmt(r.min)}, ${r.rounds} rounds)${r.iterations ? ` · ${r.iterations} iterations` : ""}`;
    tip.innerHTML = `<b>${esc(t.display)} · ${esc(p.c)}</b><span>${p.x.toLocaleString()} ${D.cases[p.c].graded ? "buses" : "nodes"} · ${what}<br>${state.op === "memory" ? "" : acc}</span>`;
    tip.hidden = false; tip.style.left = Math.min(e.clientX + 14, innerWidth - 330) + "px"; tip.style.top = (e.clientY + 14) + "px"; };
  svg.onmouseleave = () => { $("#tip").hidden = true; };
}

function sortable(table) {
  table.querySelectorAll("th").forEach((th, col) => th.onclick = () => {
    const body = table.tBodies[0], rows = [...body.rows], dir = th.dataset.dir === "asc" ? -1 : 1;
    table.querySelectorAll("th").forEach(h => delete h.dataset.dir); th.dataset.dir = dir === 1 ? "asc" : "desc";
    const key = r => { const c = r.cells[col]; const v = c.dataset.v; return v === undefined ? c.textContent : +v; };
    rows.sort((a, b) => { const x = key(a), y = key(b); return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y))) * dir; });
    rows.forEach(r => body.appendChild(r));
  });
}

function scoreboard() {
  const S = D.scoreboard, name = t => D.tools.find(x => x.name === t)?.display || t;
  $("#scoreboard").innerHTML = `<thead><tr><th>tool</th>${S.columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
    S.rows.filter(([t]) => !state.off.has(t)).map(([t, cells]) => `<tr><td>${esc(name(t))}</td>${cells.map(v => `<td${v === "·" ? ' class="na"' : ""}>${esc(v)}</td>`).join("")}</tr>`).join("") + "</tbody>";
}

// Case name and size on a case's first row; its conversions below show only the input.
function head(c, cases, withInput) {
  const k = D.cases[c], first = k.base === c || !cases.includes(k.base);
  return `<td title="${esc(k.note || k.source)}">${esc(first ? k.base : c)}</td><td data-v="${k.size}">${first ? k.size.toLocaleString() : ""}</td>` + (withInput ? `<td style="text-align:left">${esc(k.input)}</td>` : "");
}

function tables() {
  const cases = casesOf(state.grid), unit = graded(cases) ? "buses" : "nodes";
  const tools = D.tools.filter(t => !state.off.has(t.name) && cases.some(c => reads(t, c)));
  const withInput = inputsOf(state.grid).length > 1;
  const best = c => { if (state.op !== "solve") return null;
    const ok = tools.map(t => row(t.name, c, "solve")).filter(r => r && r.oracle_ok);
    return ok.length ? ok.reduce((a, b) => a.median <= b.median ? a : b).tool : null; };
  $("#t-title").textContent = state.op === "memory" ? `Peak memory${cim() ? `, ${MEM_NAME[state.memop].toLowerCase()}` : ""}: ${gridTitle(state.grid)} (MB added)`
    : `${state.op === "solve" ? (se() ? "Warm estimate" : state.problem === "opf" ? "Warm OPF solve" : batch() ? `${state.problem === "n1" ? "N-1" : "Batch"}, ${state.threads === "1" ? "one thread" : "fastest thread count"}` : "Warm solve") : state.op === "export" ? "Warm export" : OP_NAME[state.op]}: ${gridTitle(state.grid)} (median ms${batch() && state.op === "solve" ? (state.problem === "n1" ? " per outage" : " per scenario") : ""})`;
  $("#t-desc").textContent = state.op === "memory"
    ? (cim() ? {import: "Peak RSS of a fresh process loading the case", export: "Peak RSS of a fresh process loading the case and exporting it once", validate: "Peak RSS of a fresh process validating the case's files once"}[state.memop] + ", minus the RSS after importing the tool (hover for the baseline; for the Java libraries it includes the JVM, and the heap it has taken counts, collected or not)."
      : "Peak RSS of a fresh process loading the case and solving it once, minus the peak after importing the tool (hover for the baseline). The Python process only: cgmes2pgm's Fuseki server is not included.")
    : state.op === "solve" && state.problem === "opf"
    ? "✓: feasible for the case (balance and every limit, from the .m) and at most 0.01% above PGLib's reference cost (oracle, independent of every tool). ✗: a limit is broken or the cost is higher; hover the cell for which. Bold: the fastest ✓ in the row."
    : state.op === "solve" && batch()
    ? (state.problem === "n1" ? "Median time of the whole call (base case included) over its outages. ✓: every outage's solution satisfies the case with that branch out of service (tier 1 per outage). ✗: at least one does not; hover for how many. @n: the thread count. Bold: the fastest ✓ in the row." : "Median time of the whole batch over its 100 scenarios. ✓: every scenario's solution satisfies the case with that scenario's demand (tier 1 per scenario). ✗: at least one does not; hover for how many. @n: the thread count. Bold: the fastest ✓ in the row.")
    : state.op === "solve" && se()
    ? "✓: the estimate is the weighted least-squares optimum of its measurement set (oracle, independent of every tool). ✗: it is not; hover the cell for how far off. Bold: the fastest ✓ in the row."
    : state.op === "solve"
    ? (graded(cases) ? "✓: the solution satisfies the original MATPOWER case's equations at every bus (tier 1), whatever the input: a CGMES row is graded against the .m it was converted from. ✗: converged to a different problem; hover the cell for where. Bold: the fastest ✓ in the row. ·: the tool does not read this input." : "Accuracy for CGMES fixtures is judged against the published SV solution, below.")
    : state.op === "export"
    ? "The model the import built, written back to CGMES RDF/XML into an empty directory: median of repeated exports after one warm-up. Hover for files and bytes written."
    : state.op === "validate"
    ? "Files to violation report in one call, parsing included: median of 3 after one warm-up. cimoxide: its own rules; triplets and OpenCGMES: the ENTSO-E CGMES 3.0 SHACL shapes. ·: no validator. Hover for what was found."
    : "File to model: median of 3 cold loads after one warm-up load. Memory: hover a cell.";
  let h = `<thead><tr><th>case</th><th>${unit}</th>${withInput ? '<th style="text-align:left">input</th>' : ""}${tools.map(t => `<th>${esc(t.display)}</th>`).join("")}</tr></thead><tbody>`;
  for (const c of cases) {
    const b = best(c);
    h += `<tr>${head(c, cases, withInput)}`;
    for (const t of tools) {
      if (!reads(t, c) || (validating() && !validator(t))) { h += `<td class="na" data-v="1e99">·</td>`; continue; }
      const r = rec(t.name, c);
      if (!r) { const op = recOp(), f = fail(t.name, c, op); h += `<td class="fail" data-v="1e98" title="${esc(f ? f.error : "not run")}">${f ? "FAILED" : "not run"}</td>`; continue; }
      if (state.op === "memory") {
        const mb = memAdded(r);
        h += mb === undefined ? `<td class="na" data-v="1e97">—</td>` : `<td data-v="${mb}" title="${esc(`peak over a ${Math.round(r.rss_baseline_mb)} MB baseline after importing the tool`)}">${mb.toFixed(mb < 10 ? 1 : 0)}</td>`;
        continue;
      }
      let cls = "", title = r.scenarios ? `batch of ${r.scenarios} on ${r.threads} thread${r.threads > 1 ? "s" : ""}: median ${fmt(r.median)} ms, ${r.rounds} rounds` : `${r.rounds} rounds, min ${fmt(r.min)} ms`;
      if (r.op === "solve" && r.oracle_ok !== undefined) { cls = r.oracle_ok ? (t.name === b ? "ok best" : "ok") : "bad";
        if (!r.oracle_ok || se()) title += ` · ${verdict(r)}`; }
      if (r.violations !== undefined) title += ` · ${r.violations.toLocaleString()} found (${Object.entries(r.by_severity).map(([k, v]) => `${v.toLocaleString()} ${k}`).join(", ")})`;
      if (r.files_written !== undefined) title += ` · ${r.files_written} files, ${(r.bytes_written / 1e6).toFixed(1)} MB written`;
      if (r.read !== undefined) title += ` · read ${r.read.toLocaleString()} ${r.unit}`;
      if (r.op === "import" && r.rss_import_mb) title += ` · peak memory +${Math.round((r.rss_solve_mb || r.rss_import_mb) - r.rss_baseline_mb)} MB over the ${Math.round(r.rss_baseline_mb)} MB import baseline`;
      h += `<td class="${cls}" data-v="${ms(r)}" title="${esc(title)}">${fmt(ms(r))}${r.op === "solve" && r.threads && state.threads !== "1" ? ` <span class="meta">@${r.threads}</span>` : ""}</td>`;
    }
    h += "</tr>";
  }
  $("#timing").innerHTML = h + "</tbody>"; sortable($("#timing"));

  const mp = graded(cases);
  $("#a-title").textContent = !cim() ? "Accuracy" : state.op === "validate" ? "What each validator found" : "What each tool read";
  if (cim()) { whatRead(cases, tools, unit, withInput); } else {
  $("#a-desc").textContent = state.problem === "opf"
    ? "Cost against PGLib's reference, relative (negative: cheaper, which a solution breaking a limit can be). Hover for balance and every limit."
    : se()
    ? "Largest entry of one Gauss-Newton step from the estimate (p.u./rad; 0 at the WLS optimum). Hover for J against J at the true state, and the distance from the true state (information only: with noise the right answer is the optimum, not the truth)."
    : mp
    ? "Tier 1: largest |ΔP| or |ΔQ| in MVA of V·conj(Ybus·V) − S over the buses where it is specified; Ybus built from the .m file. Every tool was asked for 1e-8 p.u."
    : "Tier 2: |ΔV|/V against the published SvVoltage, median / max, with matched / published TopologicalNodes. The published solution is a reference, not ground truth.";
  let a = `<thead><tr><th>case</th><th>${unit}</th>${withInput ? '<th style="text-align:left">input</th>' : ""}${tools.map(t => `<th>${esc(t.display)}</th>`).join("")}</tr></thead><tbody>`;
  for (const c of cases) {
    a += `<tr>${head(c, cases, withInput)}`;
    for (const t of tools) {
      const r = row(t.name, c, "solve");
      if (!reads(t, c)) { a += `<td class="na" data-v="1e99">·</td>`; continue; }
      if (!r) { a += `<td class="fail" data-v="1e98">failed</td>`; continue; }
      if (r.opf_n_buses !== undefined) a += r.opf_gap === undefined || r.opf_gap === null ? `<td class="bad" data-v="1e96">incomplete</td>`
        : `<td class="${r.oracle_ok ? "ok" : "bad"}" data-v="${r.opf_gap}" title="${esc(verdict(r))}">${(r.opf_gap * 100).toFixed(4)}%</td>`;
      else if (r.se_n_buses !== undefined) a += r.se_J === undefined ? `<td class="bad" data-v="1e96">${r.se_n_reported}/${r.se_n_buses} buses</td>`
        : `<td class="${r.oracle_ok ? "ok" : "bad"}" data-v="${r.se_max_step}" title="${esc(verdict(r))}">${r.se_max_step.toExponential(1)}</td>`;
      else if (mp) { const w = Math.max(r.residual_max_dp_mw, r.residual_max_dq_mvar);
        a += `<td class="${r.oracle_ok ? "ok" : "bad"}" data-v="${w}">${w.toExponential(1)}</td>`; }
      else a += r.sv_n ? `<td data-v="${r.sv_dv_max}">${(r.sv_dv_median * 100).toFixed(3)}% / ${(r.sv_dv_max * 100).toFixed(2)}% <span class="meta">(${r.sv_n}/${r.sv_n_published})</span></td>` : `<td data-v="1e97">n=0</td>`;
    }
    a += "</tr>";
  }
  $("#accuracy").innerHTML = a + "</tbody>"; sortable($("#accuracy"));
  }

  const fs = D.failures.filter(f => !state.off.has(f.tool) && D.cases[f.case]?.grid === state.grid);
  $("#failures").innerHTML = `<thead><tr><th>tool</th><th>case</th><th>operation</th><th style="text-align:left">error</th></tr></thead><tbody>` +
    (fs.map(f => `<tr><td>${esc(D.tools.find(t => t.name === f.tool)?.display || f.tool)}</td><td style="text-align:left">${esc(f.case)}</td><td>${f.operation}</td><td style="text-align:left;white-space:normal"><code>${esc(f.error)}</code></td></tr>`).join("") || `<tr><td colspan="4">None for this selection.</td></tr>`) + "</tbody>";
  sortable($("#failures"));
}

// Speedup of each API that takes a thread count over its own one-thread run:
// one panel per case on a shared y-axis, one line per tool, and the same as a table.
function scaling() {
  $("#scaling-sec").hidden = !batch();
  if (!batch()) return;
  const cases = casesOf(state.grid), unit = state.problem === "n1" ? "outage" : "scenario", c = D.run.cpus;
  const counts = [...new Set(D.rows.filter(r => r.op === "solve" && r.threads).map(r => r.threads))].sort((a, b) => a - b);
  $("#s-desc").textContent = `Speedup of each ${state.problem === "n1" ? "contingency" : "batch"} API that takes a thread count over its own one-thread run, same case, same ${unit}s, on ${c.available ?? "?"} of ${c.logical ?? "?"} logical CPUs available to the run. Loops and APIs without a thread count are not shown. Each thread count's solution is graded on its own.`;
  const series = [];
  for (const k of cases) for (const t of D.tools) {
    const rs = D.rows.filter(r => r.tool === t.name && r.case === k && r.op === "solve" && r.threads).sort((a, b) => a.threads - b.threads);
    const one = rs.find(r => r.threads === 1);
    if (!state.off.has(t.name) && one && rs.length > 1) series.push({k, t, one, rs});
  }
  const speedup = (s, r) => s.one.median / r.median;
  const tip = (s, r) => `${r.threads} thread${r.threads > 1 ? "s" : ""}: ${speedup(s, r).toFixed(2)}× · ${fmt(ms(r))} ms per ${unit}` + (r.oracle_ok ? " · oracle: verified" : ` · oracle: FAILED · ${verdict(r)}`);

  const panels = cases.map(k => [k, series.filter(s => s.k === k)]).filter(([, ss]) => ss.length);
  $("#s-legend").innerHTML = [...new Set(series.map(s => s.t))].map(t => `<span class="chip">${swatch(t)}${esc(t.display)}</span>`).join("");
  const W = 300, H = 210, m = {l: 36, r: 10, t: 10, b: 34}, nMax = counts[counts.length - 1] || 2;
  const yMax = Math.max(2, Math.ceil(Math.max(...series.flatMap(s => s.rs.map(r => speedup(s, r))))));
  const step = yMax <= 4 ? 1 : yMax <= 10 ? 2 : 4;
  const X = n => m.l + Math.log2(n) / Math.log2(nMax) * (W - m.l - m.r);
  const Y = v => H - m.b - v / yMax * (H - m.t - m.b);
  const hits = [];
  $("#scaling-plots").innerHTML = panels.map(([k, ss]) => {
    let g = "";
    for (let v = 0; v <= yMax; v += step) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${css("--grid")}"/><text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end" font-size="11" fill="${css("--text2")}">${v}×</text>`;
    for (const n of counts) g += `<text x="${X(n)}" y="${H - m.b + 16}" text-anchor="middle" font-size="11" fill="${css("--text2")}">${n}</text>`;
    g += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle" font-size="11" fill="${css("--text2")}">threads</text>`;
    const ideal = counts.filter(n => n <= yMax).concat(yMax < nMax ? [yMax] : []);
    g += `<polyline fill="none" stroke="${css("--axis")}" stroke-width="1.5" stroke-dasharray="4 4" points="${ideal.map(n => `${X(n)},${Y(n)}`).join(" ")}"/>`;
    for (const s of ss) {
      const col = color(s.t);
      g += `<polyline fill="none" stroke="${col}" stroke-width="2"${s.t.dash ? ` stroke-dasharray="${s.t.dash}"` : ""} stroke-linejoin="round" stroke-linecap="round" points="${s.rs.map(r => `${X(r.threads)},${Y(speedup(s, r))}`).join(" ")}"/>`;
      for (const r of s.rs) {
        g += `<circle cx="${X(r.threads)}" cy="${Y(speedup(s, r))}" r="4" fill="${r.oracle_ok ? col : css("--surface")}" stroke="${col}" stroke-width="2"/>`;
        hits.push({s, r, x: X(r.threads), y: Y(speedup(s, r))});
      }
    }
    g += hits.filter(h => h.s.k === k).map(h => `<circle cx="${h.x}" cy="${h.y}" r="10" fill="transparent" data-i="${hits.indexOf(h)}"/>`).join("");
    return `<div class="card"><h3>${esc(D.cases[k].base)} <span class="meta">${D.cases[k].size.toLocaleString()} buses</span></h3><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Speedup against threads for ${esc(D.cases[k].base)}">${g}</svg></div>`;
  }).join("") || `<p>None for this selection.</p>`;
  $("#scaling-plots").onmousemove = e => { const i = e.target.dataset?.i, el = $("#tip");
    if (i === undefined) { el.hidden = true; return; }
    const {s, r} = hits[+i];
    el.innerHTML = `<b>${esc(s.t.display)} · ${esc(D.cases[s.k].base)}</b><span>${esc(tip(s, r))}</span>`;
    el.hidden = false; el.style.left = Math.min(e.clientX + 14, innerWidth - 330) + "px"; el.style.top = (e.clientY + 14) + "px"; };
  $("#scaling-plots").onmouseleave = () => { $("#tip").hidden = true; };

  $("#scaling").innerHTML = `<thead><tr><th>case</th><th>buses</th><th style="text-align:left">tool</th>${counts.map(n => `<th>${n} thread${n > 1 ? "s" : ""}</th>`).join("")}</tr></thead><tbody>` +
    (series.map(s => `<tr>${head(s.k, cases, false)}<td style="text-align:left">${esc(s.t.display)}</td>` + counts.map(n => {
      const r = s.rs.find(x => x.threads === n);
      return r ? `<td${r.oracle_ok ? "" : ' class="bad"'} data-v="${speedup(s, r)}" title="${esc(tip(s, r))}">${speedup(s, r).toFixed(2)}×</td>` : `<td class="na" data-v="0">·</td>`;
    }).join("") + "</tr>").join("") || `<tr><td colspan="${counts.length + 3}">None for this selection.</td></tr>`) + "</tbody>";
  sortable($("#scaling"));
}

// CIM: not graded, so in place of accuracy, what each library read (import)
// and what its validator found (validate).
function whatRead(cases, tools, unit, withInput) {
  const v = state.op === "validate";
  $("#a-desc").textContent = v
    ? "Findings of each validator, by severity. Not graded: cimoxide runs its own rules, triplets and OpenCGMES the ENTSO-E shapes on the merged model, each engine with its own coverage, so the counts differ by rule set as much as by data."
    : "What the import holds: triples (triplets, OpenCGMES, PowSyBl's triplestore), typed objects (cimoxide) or network elements (pypowsybl, which converts CGMES into its network model). Hover for lines, generators, loads and substations. Not graded.";
  let a = `<thead><tr><th>case</th><th>${unit}</th>${withInput ? '<th style="text-align:left">input</th>' : ""}${tools.map(t => `<th>${esc(t.display)}</th>`).join("")}</tr></thead><tbody>`;
  for (const c of cases) {
    a += `<tr>${head(c, cases, withInput)}`;
    for (const t of tools) {
      if (!reads(t, c) || (v && !validator(t))) { a += `<td class="na" data-v="1e99">·</td>`; continue; }
      const r = row(t.name, c, v ? "validate" : "import");
      if (!r) { a += `<td class="fail" data-v="1e98">failed</td>`; continue; }
      a += v
        ? `<td data-v="${r.violations}" title="${esc(Object.entries(r.by_severity).map(([k, n]) => `${n.toLocaleString()} ${k}`).join(", "))}">${r.violations.toLocaleString()}</td>`
        : `<td data-v="${r.read}" title="${esc(`${r.lines} lines, ${r.generators} generators, ${r.loads} loads, ${r.substations} substations`)}">${r.read.toLocaleString()} <span class="meta">${esc(r.unit)}</span></td>`;
    }
    a += "</tr>";
  }
  $("#accuracy").innerHTML = a + "</tbody>"; sortable($("#accuracy"));
}

function legendNote() {
  $("#legend-note").textContent = "Log-log, scaling cases and their variants only (feature cases of different grid families are in the tables), one input at a time. " + (cim() ? "Timed, not graded: every point is a completed run. No point: the tool failed or has no such operation; the table says which."
      + (state.op === "memory" ? " Values below 1 MB are drawn at 1 MB." : "")
    : state.op === "memory"
    ? "Hollow point: memory of loading only, because the solve failed. Values below 1 MB are drawn at 1 MB."
    : "Filled point: the oracle verified the solution. Hollow: the tool converged, but to a solution of a different problem. No point: the tool failed; the table says why.");
}

const SB_DESC = {pf: $("#sb-desc").textContent,
  opf: "AC optimal power flow on PGLib-OPF cases: ✓ / ✗ / FAILED per grid. ✓: feasible for the case and at most 0.01% above PGLib's reference cost.",
  se: "Weighted least-squares state estimation on the default cases, both measurement scenarios: ✓ / ✗ / FAILED per grid. ✓: the estimate is the optimum of its measurement set.",
  batch: "Batch power flow, 100 operating points per case, at one thread: ✓ / ✗ / FAILED per grid. ✓: every scenario's solution satisfies the case with that scenario's demand.",
  n1: "N-1 contingency analysis, 200 branch outages per case (19 for case14), at one thread: ✓ / ✗ / FAILED. ✓: every outage's solution satisfies the case with that branch out of service.",
  cim: "CIM libraries reading, writing and validating CGMES: cases done of all, per operation. Timed, not graded (see How this is measured). ·: the tool has no validator."};
function render() {
  const problems = [["pf", "Power flow"], ["se", "State estimation"], ["opf", "Optimal power flow"], ["batch", "Batch power flow"], ["n1", "N-1 contingencies"], ["cim", "CIM import/export"]].filter(([p]) => ALL[p]);
  $("#problem").hidden = problems.length < 2;
  seg($("#problem"), "problem", problems);
  D = ALL[state.problem];
  if (!D.grids.some(([g]) => g === state.grid)) state.grid = D.grids[0][0];
  $("#sb-desc").textContent = SB_DESC[state.problem];
  const ops = (cim() ? ["import", "export", "validate", "memory"] : ["solve", "import", "memory"]).map(o => [o, OP_NAME[o]]);
  if (!ops.some(([o]) => o === state.op)) state.op = ops[0][0];
  seg($("#op"), "op", ops);
  seg($("#grid"), "grid", D.grids.map(([g, label]) => [g, label]));
  const inputs = inputsOf(state.grid);
  if (!inputs.includes(state.input)) state.input = inputs[0];
  $("#input").hidden = inputs.length < 2;
  seg($("#input"), "input", inputs.map(i => [i, `Chart: ${i}`]));
  $("#threads").hidden = !batch() || state.op !== "solve";
  seg($("#threads"), "threads", [["1", "1 thread"], ["best", "Fastest thread count"]]);
  $("#memop").hidden = !cim() || state.op !== "memory";
  seg($("#memop"), "memop", Object.entries(MEM_NAME));
  scoreboard();
  chips(); chart(); tables(); scaling(); legendNote();
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
new MutationObserver(render).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
render();
</script>
</body>
</html>
"""
