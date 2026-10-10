"""Colours for reports and the site. Each tool keeps one categorical slot for
life (colour follows the tool, never its rank). The eight slots below are
the dataviz reference palette in its validated order; checked with its
validator in both modes (worst adjacent CVD dE 9.1 light / 8.4 dark,
normal-vision 19.6 / 19.3). Slot 9 (teal, Sparlectra) was added on purpose
past the validated eight, accepted as imperfect: in the adjacent order the
nine still pass both modes, but all-pairs teal sits close to slot 1 blue
(normal-vision dE 10.8 light / 10.4 dark) and slot 3 aqua (13.5 / 10.9),
and under protanopia next to slot 5 magenta (2.9 light). A tenth tool needs a
different encoding (small multiples, "other"), never another hue. p3s (the
parallel pandapower solver, from pandapower's makers) is drawn as a variant
of slot 1: pandapower's blue, dashed, keyed one unit off like
REFERENCE_DOTTED; the dash and its label tell the two apart. ExaPF.jl, the
tenth tool, is drawn the same way as a variant of slot 8: Sienna's red,
dotted (both Julia), which is the encoding rule above, not a new hue.
ExaPF.jl on its GPU backend is the same tool on other hardware: ExaPF's red
again, dash-dotted, so that its two rows read as one tool.
Checked all-pairs, as a chart showing every tool at once needs, the palette
fails: slot 8 red vs slot 2 orange is dE 7.1 even with full colour vision,
and several pairs sit below 4 under colour-vision deficiency. The legend,
the site's hover and tool toggles, and comparison.md carry identity; small
multiples would fix it properly. Three light-mode slots sit below 3:1 contrast on
the surface, so every chart carries direct labels and a table view.
"""
LIGHT_TO_DARK = {
    "#2a78d6": "#3987e5",   # 1 blue    pandapower
    "#eb6834": "#d95926",   # 2 orange  lightsim2grid
    "#1baf7a": "#199e70",   # 3 aqua    PyPSA
    "#eda100": "#c98500",   # 4 yellow  power-grid-model
    "#e87ba4": "#d55181",   # 5 magenta pypowsybl
    "#008300": "#008300",   # 6 green   VeraGrid
    "#4a3aa7": "#9085e9",   # 7 violet  PGM via cgmes2pgm
    "#e34948": "#e66767",   # 8 red     Sienna (PowerFlows.jl)
    "#0f8fa8": "#1b9cb6",   # 9 teal    Sparlectra.jl (past the validated eight, see above)
}
# A reference implementation is not another hue: neutral ink, so it reads as
# a reference line in both modes and without colour vision. MATPOWER (the
# case format's) is dashed; PowerModels.jl (whose results with Ipopt are
# PGLib-OPF's published references) is dotted, in the same ink, the pattern
# telling the two apart. `DASH` maps a colour to its line pattern.
REFERENCE = "#52514e"
REFERENCE_DOTTED = "#52514f"   # the same ink, one unit off: a distinct key for its pattern
LIGHT_TO_DARK[REFERENCE] = "#c3c2b7"
LIGHT_TO_DARK[REFERENCE_DOTTED] = "#c3c2b8"
P3S = "#2a78d7"   # slot 1's blue, one unit off: pandapower's variant, dashed
LIGHT_TO_DARK[P3S] = "#3987e6"
EXAPF = "#e34949"   # slot 8's red, one unit off: drawn as Sienna's variant, dotted
LIGHT_TO_DARK[EXAPF] = "#e66768"
EXAPF_GPU = "#e3494a"   # slot 8's red, two units off: ExaPF on a GPU, dash-dotted
LIGHT_TO_DARK[EXAPF_GPU] = "#e66769"
DASH = {REFERENCE: "6 4", REFERENCE_DOTTED: "1.5 3.5", P3S: "6 4", EXAPF: "1.5 3.5", EXAPF_GPU: "6 3 1.5 3"}
LIGHT = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df", "axis": "#8a8984"}
DARK = {"surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "grid": "#33332f", "axis": "#6f6e69"}
