"""Colours for reports and the site. Hue says the language a tool's solver
is written in; within a language, a lighter or darker step of that hue and
the line pattern tell the tools apart. Colour follows the tool, never its
rank: a filter never repaints the survivors.

Languages (the adapter's `language`) and their hues, from the dataviz
reference palette: Python blue, C++ yellow, Julia magenta (near Julia's
purple), Java green, and Rust a rust brown (light #8b3200, dark #e4a063),
added off the palette because no slot passes next to blue and magenta in
both modes. Of the 70 ways to give four slots to the four languages that
share the busiest chart (power flow: Python, C++, Julia, Java), blue,
yellow, magenta and green is the best (dataviz validator, all pairs, worst
normal-vision dE 19.6 light / 19.3 dark). Julia takes magenta and Java green
and not the other way round: under red-green colour blindness yellow and
green collapse, and Java has one tool on that chart where Julia has four.

Steps: each language's base and three OKLCH lightness steps of its hue
(light mode L 0.45 to 0.76, dark 0.50 to 0.83), at least about 0.08 apart.
Measured on every chart with the tools it shows, both modes:
- across languages, worst normal-vision dE 15.6 (cgmes2pgm against ExaPF.jl,
  power flow, light): two languages never read as one;
- two tools that also share a line pattern stay at CVD dE 6.9 or more
  (lightsim2grid against pypowsybl, dark), the band that is legal with
  direct labels, which every chart has; pairs below it (pypowsybl against
  p3s, 0.6 under protanopia) differ in pattern;
- within a language, steps are dE 7 to 9 apart (normal vision): the pattern,
  the direct label and the hover highlight carry identity there.
lightsim2grid's yellow base is 2.1:1 on the light surface (it was
power-grid-model's before): solid, labelled and in the tables, as the
relief rule requires. Dotted lines take steps of at least 3:1.

Patterns: solid, dashed, dotted, long-dashed for a language's first four
tools; dash-dot marks a GPU tool, which shares a chart only with other GPU
tools and its CPU baselines. A reference implementation is no language:
neutral ink, MATPOWER dashed, PowerModels.jl (PGLib-OPF's reference) dotted.

The site keys a line's pattern and its dark step on the light hex, so two
tools on one step differ in the last digit (one unit off): lightsim2grid,
gpusim2grid and its LODF DC N-1; pandapower, ToOp and triplets; Sparlectra,
ExaPF.jl on a GPU and ExaModelsPower.jl; ExaPF.jl and ExaModelsPower.jl on a GPU;
cimoxide, gridoxide (#8b3201, its CPU tool, PR #28) and gridoxide on a GPU.
gridoxide on a GPU's dark step is its own: cimoxide's
(#e4a063) is 9.8 from the C++ yellow (#c98500) under normal vision, below the
floor of 15, and the GPU tabs put the two side by side, which no CIM chart
does. #9e5220 against the yellow and ExaPF.jl's GPU magenta, all pairs, dark
surface: normal vision 15.8, CVD 12.8 (deutan), contrast 3:1 or more; the
light step against the same, 19.8 and 16.4.
"""
SOLID, DASHED, DOTTED, LONG, DASHDOT = None, "6 4", "1.5 3.5", "12 4", "6 3 1.5 3"

# light hex: (dark hex, pattern); comments name the tool, the language and the step.
STYLES = {
    "#2a78d6": ("#3987e5", SOLID),     # pandapower       python  base
    "#4996f6": ("#5aa3ff", DASHED),    # PyPSA            python  1
    "#035eb9": ("#176ac6", DOTTED),    # VeraGrid         python  2
    "#2a78d7": ("#3987e6", DASHDOT),   # ToOp (GPU)       python  base
    "#2a78d8": ("#3987e7", DASHED),    # triplets         python  base
    "#eda100": ("#c98500", SOLID),     # lightsim2grid    c++     base
    "#c78600": ("#a97004", DASHED),    # p3s              c++     1
    "#a36e04": ("#e7a137", DOTTED),    # power-grid-model c++     2
    "#825600": ("#ffb851", LONG),      # cgmes2pgm        c++     3
    "#eda101": ("#c98501", DASHDOT),   # gpusim2grid      c++     base
    "#eda102": ("#c98502", DOTTED),    # lightsim2grid LODF DC N-1, c++, base
    "#e87ba4": ("#d55181", SOLID),     # Sparlectra.jl    julia   base
    "#ca608a": ("#f46d9b", DASHED),    # JuliaGrid.jl     julia   1
    "#af4872": ("#b53266", DOTTED),    # Sienna           julia   2
    "#95305b": ("#ff9bb9", LONG),      # ExaPF.jl         julia   3
    "#e87ba5": ("#d55182", DASHDOT),   # ExaPF.jl (GPU)   julia   base
    "#e87ba6": ("#d55183", DOTTED),    # ExaModelsPower.jl julia  base
    "#95305c": ("#ff9bba", DASHDOT),   # ExaModelsPower.jl (GPU) julia 3
    "#008300": ("#008300", SOLID),     # pypowsybl        java    base
    "#33a02e": ("#33a02e", DASHED),    # OpenCGMES        java    1
    "#026802": ("#71da6a", DOTTED),    # PowSyBl CgmesModel java  3
    "#8b3200": ("#e4a063", SOLID),     # cimoxide         rust    base
    "#8b3201": ("#e4a064", SOLID),     # gridoxide        rust    base
    "#8b3202": ("#9e5220", DASHDOT),   # gridoxide (GPU)  rust    base, own dark step (below)
    "#a94d25": ("#c68447", DASHED),    # rustpower        rust    1
    "#52514e": ("#c3c2b7", DASHED),    # MATPOWER         reference
    "#52514f": ("#c3c2b8", DOTTED),    # PowerModels.jl   reference
}
LIGHT_TO_DARK = {light: dark for light, (dark, _) in STYLES.items()}
DASH = {light: dash for light, (_, dash) in STYLES.items() if dash}
LIGHT = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df", "axis": "#8a8984"}
DARK = {"surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "grid": "#33332f", "axis": "#6f6e69"}
