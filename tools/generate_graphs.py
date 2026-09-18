"""Static SVG charts for the README, from result JSON only.

One chart per (operation, family): time (solve, import) or peak memory vs
case size, log-log, one line per tool in its fixed colour. Only `smoke` and `scaling` cases are plotted: a
line through feature cases of different grid families (RTE, PEGASE, Polish)
at similar sizes would draw a trend that is not there. They are in the tables. Filled markers: the oracle accepted the solution;
hollow markers: the tool converged to a solution of a different problem
(tier 1 failed). Failed runs have no point at all. Each chart is written in
a light and a dark variant (`<name>.svg`, `<name>-dark.svg`) for the
README's <picture> element.
"""
import math
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

from cases.registry import CASES  # noqa: E402
from tools.benchmark_data import Results, case_size  # noqa: E402

PLOTTED_GROUPS = {"smoke", "scaling"}
from tools.palette import DARK, LIGHT, LIGHT_TO_DARK  # noqa: E402

from tools.benchmark_data import FAMILY_TITLES  # noqa: E402

def _lower_first(text: str) -> str:
    """'MATPOWER cases' stays as is; 'CGMES ...' too; 'Other' -> 'other'."""
    word = text.split()[0]
    return text if word.isupper() else text[0].lower() + text[1:]


TITLES = {(op, fam): f"{label}, {_lower_first(FAMILY_TITLES[fam])}"
          for op, label in (("solve", "Warm AC power-flow solve"), ("import", "Import from file"),
                            ("memory", "Peak memory of loading and solving"))
          for fam in FAMILY_TITLES}
MEMORY_FLOOR_MB = 1.0   # log axis: smaller additions are drawn at this line


def memory_added_mb(extra: dict) -> float | None:
    """Peak RSS over loading and one solve (or loading alone, if the solve
    failed), minus the baseline after importing the tool (adapters/memory.py)."""
    if "rss_import_mb" not in extra:
        return None
    return extra.get("rss_solve_mb", extra["rss_import_mb"]) - extra["rss_baseline_mb"]


def _spread(ys: list[float], min_gap: float) -> list[float]:
    """Nudge label positions (in log10 space) apart so none overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def chart(res: Results, operation: str, family: str, path: Path, dark: bool) -> bool:
    theme = DARK if dark else LIGHT
    series = []
    for tool in res.tool_order():
        pts = []
        for case in res.cases(family):
            if not PLOTTED_GROUPS & set(CASES[case]["groups"]):
                continue
            rec = res.get(tool, case, "import" if operation == "memory" else operation)
            if rec is None:
                continue
            if operation == "memory":
                mb = memory_added_mb(rec.extra)
                if mb is not None:   # hollow: loading only, the solve failed
                    pts.append((case_size(case), max(mb, MEMORY_FLOOR_MB), "rss_solve_mb" in rec.extra, case))
                continue
            ok = rec.extra.get("oracle_ok", True) if operation == "solve" else True
            pts.append((case_size(case), rec.median_ms, ok, case))
        if pts:
            color = res.tools[tool]["color"]
            series.append((res.tools[tool]["display_name"], LIGHT_TO_DARK.get(color, color) if dark else color, pts))
    if not series:
        return False

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=100)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    ends = []
    for name, color, pts in series:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=2, zorder=2, label=name, solid_capstyle="round")
        for x, y, ok, _ in pts:
            ax.plot([x], [y], marker="o", markersize=7, zorder=3, linestyle="none",
                    markerfacecolor=color if ok else theme["surface"], markeredgecolor=color, markeredgewidth=2)
        ends.append((xs[-1], ys[-1], name))
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("published nodes" if family == "cgmes" else "buses", color=theme["text2"])
    ax.set_ylabel("MB added over the import baseline" if operation == "memory" else "median time (ms)",
                  color=theme["text2"])
    ax.set_title(TITLES[(operation, family)], loc="left", color=theme["text"], fontsize=12, pad=12)
    ax.grid(True, which="major", color=theme["grid"], linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(colors=theme["text2"], which="both")

    # Direct labels, in text ink, only for lines that reach the right edge:
    # a label there for a line that stopped earlier would sit beside the
    # wrong data. The legend names every line.
    xmax = max(x for x, _, _ in ends)
    ends = [e for e in ends if e[0] == xmax]
    lo, hi = ax.get_ylim()
    gap = (math.log10(hi) - math.log10(lo)) * 0.055
    for (_, _, name), ly in zip(ends, _spread([math.log10(y) for _, y, _ in ends], gap)):
        ax.annotate(name, xy=(xmax * 1.15, 10 ** ly), color=theme["text"], fontsize=9, va="center",
                    annotation_clip=False)
    ax.set_xlim(right=xmax * 1.1)
    legend = fig.legend(loc="lower left", bbox_to_anchor=(0.08, 0.075), frameon=False, fontsize=9, ncols=4,
                        handlelength=1.5, columnspacing=1.2)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    note = ("Peak RSS of a fresh process loading the case and solving it once, minus the peak after importing the tool.\n"
            f"Hollow: loading only (the solve failed). Below {MEMORY_FLOOR_MB:g} MB drawn at {MEMORY_FLOOR_MB:g} MB. "
            "Exact values: comparison.md."
            if operation == "memory" else
            "Filled: solution verified by the oracle. Hollow: converged to a different problem.\n"
            "Missing: the tool failed (see comparison.md).")
    fig.text(0.09, 0.01, note, color=theme["text2"], fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.76, top=0.9, bottom=0.27)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)
    return True


def generate(directory: Path, res: Results) -> list[Path]:
    out_dir = Path(directory) / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for operation, family in TITLES:
        for dark in (False, True):
            path = out_dir / f"{operation}_{family}{'-dark' if dark else ''}.svg"
            if chart(res, operation, family, path, dark):
                written.append(path)
    return written
