"""Writes `docs/index.html`: one self-contained page (GitHub Pages), from
result JSON only. Data is embedded; the chart and tables are drawn in the
browser so the filters apply to both. No external requests.
"""
import html
import json
from pathlib import Path

from cases.registry import CASES
from tools.benchmark_data import FAMILY_TITLES, Results, case_size
from tools.palette import DARK, LIGHT, LIGHT_TO_DARK

KEEP = ("iterations", "oracle_ok", "residual_max_dp_mw", "residual_max_dq_mvar", "residual_max_dvm_pu",
        "residual_worst_bus", "residual_n_checked", "residual_n_buses", "sv_n", "sv_n_published",
        "sv_dv_median", "sv_dv_max", "sv_da_max_deg", "rss_baseline_mb", "rss_import_mb", "rss_solve_mb")


def payload(res: Results) -> dict:
    tools = [{"name": t, "display": m["display_name"], "color": m["color"],
              "colorDark": LIGHT_TO_DARK.get(m["color"], m["color"]), "version": m["version"],
              "language": m["language"], "families": m["families"], "settings": m["settings"]}
             for t in res.tool_order() for m in [res.tools[t]]]
    cases = {c: {"family": CASES[c]["family"], "size": case_size(c), "groups": CASES[c]["groups"],
                 "source": CASES[c]["source"], "note": CASES[c]["note"]}
             for fam in res.families() for rob in (False, True) for c in res.cases(fam, rob)}
    rows = [{"tool": r.tool, "case": r.case, "op": r.operation, "median": r.median_ms, "min": r.min_ms,
             "rounds": r.rounds, **{k: r.extra[k] for k in KEEP if k in r.extra}} for r in res.records]
    run = next((r for r in res.runs if r["machine"]), {})
    cpu = run.get("machine", {}).get("cpu", {})
    families = [[f, {"matpower": "MATPOWER", "cgmes": "CGMES fixtures", "converted-cimoxide": "Converted (cimoxide)",
                      "converted-pypowsybl": "Converted (pypowsybl)"}[f], FAMILY_TITLES[f]] for f in res.families()]
    return {"tools": tools, "cases": cases, "rows": rows, "failures": res.failures, "families": families,
            "run": {"cpu": cpu.get("brand_raw", "?"), "cores": cpu.get("count", "?"),
                    "os": f"{run.get('machine', {}).get('system', '?')} {run.get('machine', {}).get('release', '')}",
                    "git": str(run.get("git_sha", "?"))[:12], "date": str(run.get("datetime", "?"))[:10]}}


def generate(directory: Path, res: Results) -> str:
    data = json.dumps(payload(res), separators=(",", ":")).replace("</", "<\\/")
    return (TEMPLATE.replace("__DATA__", data)
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
.legend-note{font-size:13px;color:var(--text2);margin:8px 0 0}
details{margin:6px 0}summary{cursor:pointer;color:var(--text)}
code{font-size:13px;background:var(--chip);padding:1px 5px;border-radius:4px}
</style>
</head>
<body>
<main>
<h1>grid-bench</h1>
<p class="lede">Power-flow speed of open-source power system tools, where every timing is graded by an oracle that no tool under test takes part in. MATPOWER cases are also converted to CGMES by two converters, and tools are graded on those against the original case.</p>
<p class="meta" id="meta"></p>

<div class="controls">
  <div class="seg" role="group" aria-label="Operation" id="op"></div>
  <div class="seg" role="group" aria-label="Case family" id="fam"></div>
  <div class="chips" id="toolchips" aria-label="Tools"></div>
</div>
<div class="card"><svg id="chart" viewBox="0 0 960 440" role="img" aria-label="Time versus case size, one line per tool"></svg>
<p class="legend-note" id="legend-note"></p></div>

<h2 id="t-title"></h2>
<p id="t-desc"></p>
<div class="scroll"><table id="timing"></table></div>

<h2>Accuracy</h2>
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
<li><b>Memory</b> is the peak RSS of a fresh process loading and solving the case, minus the peak after importing the tool (values below 1 MB are drawn at 1 MB on the log axis). Only the benchmark process counts: cgmes2pgm's Fuseki server is not included.</li>
</ul>
<div class="tip" id="tip" hidden></div>
</main>
<script>
const D = __DATA__;
const state = {op: "solve", fam: (D.families[0] || ["matpower"])[0], off: new Set()};
const famTitle = f => (D.families.find(x => x[0] === f) || [f, f, f])[2];
const graded = f => f !== "cgmes";   // tier-1 residual against a MATPOWER case
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const dark = () => document.documentElement.dataset.theme === "dark" ||
  (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
const color = t => dark() ? t.colorDark : t.color;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = ms => ms < 10 ? ms.toFixed(3) : ms < 1000 ? ms.toFixed(1) : Math.round(ms).toLocaleString();
const row = (t, c, op) => D.rows.find(r => r.tool === t && r.case === c && r.op === op);
const fail = (t, c, op) => D.failures.find(f => f.tool === t && f.case === c && f.operation === op);
const casesOf = fam => Object.keys(D.cases).filter(c => D.cases[c].family === fam).sort((a, b) => D.cases[a].size - D.cases[b].size);
const verified = r => r.op !== "solve" || r.oracle_ok === undefined || r.oracle_ok;
const MEM_FLOOR = 1;   // MB; log axis
const memAdded = r => r && r.rss_import_mb !== undefined ? (r.rss_solve_mb ?? r.rss_import_mb) - r.rss_baseline_mb : undefined;
// The record a view reads, and the value it plots: memory lives on the import record.
const rec = (t, c) => row(t, c, state.op === "memory" ? "import" : state.op);
const val = r => state.op === "memory" ? memAdded(r) : r.median;

$("#meta").textContent = `Run ${D.run.date} · commit ${D.run.git} · ${D.run.cpu}, ${D.run.cores} logical CPUs · ${D.run.os}`;

function seg(el, key, options) {
  el.innerHTML = options.map(([v, l]) => `<button data-v="${v}" aria-pressed="${state[key] === v}">${l}</button>`).join("");
  el.onclick = e => { const b = e.target.closest("button"); if (!b) return; state[key] = b.dataset.v; render(); };
}
function chips() {
  $("#toolchips").innerHTML = D.tools.map(t => `<button class="chip" data-t="${t.name}" aria-pressed="${!state.off.has(t.name)}"><i style="background:${color(t)}"></i>${esc(t.display)}</button>`).join("");
}
$("#toolchips").onclick = e => { const b = e.target.closest(".chip"); if (!b) return;
  state.off.has(b.dataset.t) ? state.off.delete(b.dataset.t) : state.off.add(b.dataset.t); render(); };

function chart() {
  const svg = $("#chart"), W = 960, H = 440, m = {l: 64, r: 200, t: 20, b: 48};
  const tools = D.tools.filter(t => !state.off.has(t.name) && t.families.includes(state.fam));
  const cases = casesOf(state.fam).filter(c => D.cases[c].groups.some(g => g === "smoke" || g === "scaling"));
  const series = tools.map(t => ({t, pts: cases.map(c => ({c, r: rec(t.name, c)})).filter(p => p.r && val(p.r) !== undefined)
    .map(p => ({x: D.cases[p.c].size, y: state.op === "memory" ? Math.max(val(p.r), MEM_FLOOR) : val(p.r),
                ok: state.op === "memory" ? p.r.rss_solve_mb !== undefined : verified(p.r), c: p.c, r: p.r}))})).filter(s => s.pts.length);
  const all = series.flatMap(s => s.pts);
  if (!all.length) { svg.innerHTML = `<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="${css("--text2")}">No results for this selection</text>`; return; }
  const lg = Math.log10, x0 = Math.floor(lg(Math.min(...all.map(p => p.x)))), x1 = Math.ceil(lg(Math.max(...all.map(p => p.x))));
  const y0 = Math.floor(lg(Math.min(...all.map(p => p.y)))), y1 = Math.ceil(lg(Math.max(...all.map(p => p.y))));
  const X = v => m.l + (lg(v) - x0) / Math.max(1, x1 - x0) * (W - m.l - m.r);
  const Y = v => H - m.b - (lg(v) - y0) / Math.max(1, y1 - y0) * (H - m.t - m.b);
  let s = "";
  for (let e = y0; e <= y1; e++) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(10**e)}" y2="${Y(10**e)}" stroke="${css("--grid")}"/><text x="${m.l - 8}" y="${Y(10**e) + 4}" text-anchor="end" font-size="12" fill="${css("--text2")}">${10**e >= 1 ? (10**e).toLocaleString() : 10**e}</text>`;
  for (let e = x0; e <= x1; e++) s += `<line x1="${X(10**e)}" x2="${X(10**e)}" y1="${m.t}" y2="${H - m.b}" stroke="${css("--grid")}"/><text x="${X(10**e)}" y="${H - m.b + 18}" text-anchor="middle" font-size="12" fill="${css("--text2")}">${(10**e).toLocaleString()}</text>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 8}" text-anchor="middle" font-size="12" fill="${css("--text2")}">${state.fam === "cgmes" ? "published nodes" : "buses"}</text>`;
  s += `<text transform="translate(16 ${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="${css("--text2")}">${state.op === "memory" ? "MB added over import baseline" : `median ${state.op} time (ms)`}</text>`;
  const labels = [];
  for (const {t, pts} of series) {
    const c = color(t);
    s += `<polyline fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${pts.map(p => `${X(p.x)},${Y(p.y)}`).join(" ")}"/>`;
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
    if (r.op === "solve" && r.oracle_ok !== undefined) acc = r.oracle_ok ? "oracle: verified" : `oracle: FAILED, worst |ΔS| ${Math.max(r.residual_max_dp_mw, r.residual_max_dq_mvar).toExponential(1)} MVA at bus ${r.residual_worst_bus}`;
    else if (r.sv_n) acc = `vs published SV: median ${(r.sv_dv_median * 100).toFixed(3)}%, max ${(r.sv_dv_max * 100).toFixed(2)}%`;
    const what = state.op === "memory"
      ? `+${memAdded(r).toFixed(1)} MB peak over a ${Math.round(r.rss_baseline_mb)} MB baseline (import peak +${(r.rss_import_mb - r.rss_baseline_mb).toFixed(1)} MB)`
      : `median ${fmt(r.median)} ms (min ${fmt(r.min)}, ${r.rounds} rounds)${r.iterations ? ` · ${r.iterations} iterations` : ""}`;
    tip.innerHTML = `<b>${esc(t.display)} · ${esc(p.c)}</b><span>${p.x.toLocaleString()} ${state.fam === "cgmes" ? "nodes" : "buses"} · ${what}<br>${state.op === "memory" ? "" : acc}</span>`;
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

function tables() {
  const tools = D.tools.filter(t => !state.off.has(t.name));
  const cases = casesOf(state.fam), unit = state.fam === "cgmes" ? "nodes" : "buses";
  $("#t-title").textContent = state.op === "memory" ? `Peak memory: ${famTitle(state.fam)} (MB added)`
    : `${state.op === "solve" ? "Warm solve" : "Import"}: ${famTitle(state.fam)} (median ms)`;
  $("#t-desc").textContent = state.op === "memory"
    ? "Peak RSS of a fresh process loading the case and solving it once, minus the peak after importing the tool (hover for the baseline). The Python process only: cgmes2pgm's Fuseki server is not included."
    : state.op === "solve"
    ? (graded(state.fam) ? "✓: the solution satisfies the original MATPOWER case's equations at every bus (tier 1). ✗: converged to a different problem; hover the cell for where." : "Accuracy for CGMES fixtures is judged against the published SV solution, below.")
    : "File to model: median of 3 cold loads after one warm-up load. Memory: hover a cell.";
  let h = `<thead><tr><th>case</th><th>${unit}</th>${tools.map(t => `<th>${esc(t.display)}</th>`).join("")}</tr></thead><tbody>`;
  for (const c of cases) {
    h += `<tr><td title="${esc(D.cases[c].note || D.cases[c].source)}">${esc(c)}</td><td data-v="${D.cases[c].size}">${D.cases[c].size.toLocaleString()}</td>`;
    for (const t of tools) {
      if (!t.families.includes(state.fam)) { h += `<td class="na" data-v="1e99">—</td>`; continue; }
      const r = rec(t.name, c);
      if (!r) { const op = state.op === "memory" ? "import" : state.op, f = fail(t.name, c, op); h += `<td class="fail" data-v="1e98" title="${esc(f ? f.error : "not run")}">${f ? "FAILED" : "not run"}</td>`; continue; }
      if (state.op === "memory") {
        const mb = memAdded(r);
        h += mb === undefined ? `<td class="na" data-v="1e97">—</td>` : `<td data-v="${mb}" title="${esc(`peak over a ${Math.round(r.rss_baseline_mb)} MB baseline after importing the tool`)}">${mb.toFixed(mb < 10 ? 1 : 0)}</td>`;
        continue;
      }
      let cls = "", title = `${r.rounds} rounds, min ${fmt(r.min)} ms`;
      if (r.op === "solve" && r.oracle_ok !== undefined) { cls = r.oracle_ok ? "ok" : "bad";
        if (!r.oracle_ok) title += ` · max |ΔP| ${r.residual_max_dp_mw.toExponential(2)} MW, |ΔQ| ${r.residual_max_dq_mvar.toExponential(2)} MVAr, |ΔV| setpoint ${r.residual_max_dvm_pu.toExponential(1)} p.u., worst bus ${r.residual_worst_bus}`; }
      if (r.op === "import" && r.rss_import_mb) title += ` · peak memory +${Math.round((r.rss_solve_mb || r.rss_import_mb) - r.rss_baseline_mb)} MB over the ${Math.round(r.rss_baseline_mb)} MB import baseline`;
      h += `<td class="${cls}" data-v="${r.median}" title="${esc(title)}">${fmt(r.median)}</td>`;
    }
    h += "</tr>";
  }
  $("#timing").innerHTML = h + "</tbody>"; sortable($("#timing"));

  const mp = graded(state.fam);
  $("#a-desc").textContent = mp
    ? "Tier 1: largest |ΔP| or |ΔQ| in MVA of V·conj(Ybus·V) − S over the buses where it is specified; Ybus built from the .m file. Every tool was asked for 1e-8 p.u."
    : "Tier 2: |ΔV|/V against the published SvVoltage, median / max, with matched / published TopologicalNodes. The published solution is a reference, not ground truth.";
  let a = `<thead><tr><th>case</th>${tools.map(t => `<th>${esc(t.display)}</th>`).join("")}</tr></thead><tbody>`;
  for (const c of cases) {
    a += `<tr><td>${esc(c)}</td>`;
    for (const t of tools) {
      const r = row(t.name, c, "solve");
      if (!t.families.includes(state.fam)) { a += `<td class="na" data-v="1e99">—</td>`; continue; }
      if (!r) { a += `<td class="fail" data-v="1e98">failed</td>`; continue; }
      if (mp) { const w = Math.max(r.residual_max_dp_mw, r.residual_max_dq_mvar);
        a += `<td class="${r.oracle_ok ? "ok" : "bad"}" data-v="${w}">${w.toExponential(1)}</td>`; }
      else a += r.sv_n ? `<td data-v="${r.sv_dv_max}">${(r.sv_dv_median * 100).toFixed(3)}% / ${(r.sv_dv_max * 100).toFixed(2)}% <span class="meta">(${r.sv_n}/${r.sv_n_published})</span></td>` : `<td data-v="1e97">n=0</td>`;
    }
    a += "</tr>";
  }
  $("#accuracy").innerHTML = a + "</tbody>"; sortable($("#accuracy"));

  const fs = D.failures.filter(f => !state.off.has(f.tool) && D.cases[f.case]?.family === state.fam);
  $("#failures").innerHTML = `<thead><tr><th>tool</th><th>case</th><th>operation</th><th style="text-align:left">error</th></tr></thead><tbody>` +
    (fs.map(f => `<tr><td>${esc(D.tools.find(t => t.name === f.tool)?.display || f.tool)}</td><td style="text-align:left">${esc(f.case)}</td><td>${f.operation}</td><td style="text-align:left;white-space:normal"><code>${esc(f.error)}</code></td></tr>`).join("") || `<tr><td colspan="4">None for this selection.</td></tr>`) + "</tbody>";
  sortable($("#failures"));
}

function legendNote() {
  $("#legend-note").textContent = "Log-log, scaling cases only (feature cases of different grid families are in the tables). " + (state.op === "memory"
    ? "Hollow point: memory of loading only, because the solve failed. Values below 1 MB are drawn at 1 MB."
    : "Filled point: the oracle verified the solution. Hollow: the tool converged, but to a solution of a different problem. No point: the tool failed; the table says why.");
}

function render() {
  seg($("#op"), "op", [["solve", "Solve"], ["import", "Import"], ["memory", "Memory"]]);
  seg($("#fam"), "fam", D.families.map(([f, label]) => [f, label]));
  chips(); chart(); tables(); legendNote();
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
new MutationObserver(render).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
render();
</script>
</body>
</html>
"""
