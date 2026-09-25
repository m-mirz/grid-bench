"""Colours for reports and the site. Each tool keeps one categorical slot for
life (colour follows the tool, never its rank). The eight slots below are
the dataviz reference palette in its validated order; checked with its
validator in both modes (worst adjacent CVD dE 9.1 light / 8.4 dark,
normal-vision 19.6 / 19.3). Slot 9 (teal, Sparlectra) was added on purpose
past the validated eight, accepted as imperfect: in the adjacent order the
nine still pass both modes, but all-pairs teal sits close to slot 1 blue
(normal-vision dE 10.8 light / 10.4 dark) and slot 3 aqua (13.5 / 10.9),
and under protanopia next to slot 5 magenta (2.9 light). A tenth tool needs a
different encoding (small multiples, "other"), never another hue.
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
# A reference implementation (MATPOWER) is not a ninth hue: neutral ink,
# drawn dashed, so it reads as a reference line in both modes and without
# colour vision.
REFERENCE = "#52514e"
LIGHT_TO_DARK[REFERENCE] = "#c3c2b7"
LIGHT = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df", "axis": "#8a8984"}
DARK = {"surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "grid": "#33332f", "axis": "#6f6e69"}
