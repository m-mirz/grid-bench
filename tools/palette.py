"""Colours for reports and the site. Each tool keeps one categorical slot for
life (colour follows the tool, never its rank). The seven slots below are the
first seven of the dataviz reference palette in its validated order; checked
with its validator in both modes (worst adjacent CVD dE 9.1 light / 8.4 dark,
normal-vision 19.6 / 19.3). Three light-mode slots sit below 3:1 contrast on
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
}
LIGHT = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df", "axis": "#8a8984"}
DARK = {"surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "grid": "#33332f", "axis": "#6f6e69"}
