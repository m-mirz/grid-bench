"""Tool registry. Imported lazily: each container installs only its own tool."""
from importlib import import_module

ADAPTERS = {
    "pandapower": "adapters.pandapower_adapter:PandapowerAdapter",
    "p3s": "adapters.p3s_adapter:P3sAdapter",   # slot 1's variant (tools/palette.py P3S)
    "lightsim2grid": "adapters.lightsim2grid_adapter:Lightsim2gridAdapter",
    "pypsa": "adapters.pypsa_adapter:PypsaAdapter",
    "pgm": "adapters.pgm_adapter:PgmAdapter",
    "pypowsybl": "adapters.pypowsybl_adapter:PypowsyblAdapter",
    "veragrid": "adapters.veragrid_adapter:VeragridAdapter",
    "cgmes2pgm": "adapters.cgmes2pgm_adapter:Cgmes2pgmAdapter",
    "sienna": "adapters.sienna_adapter:SiennaAdapter",
    "sparlectra": "adapters.sparlectra_adapter:SparlectraAdapter",
    "matpower": "adapters.matpower_adapter:MatpowerAdapter",
}

# State estimators, by the same tool names (a tool's colour and container are
# its power-flow adapter's). Order: ADAPTERS order.
ESTIMATORS = {
    "pandapower": "adapters.pandapower_se_adapter:PandapowerEstimator",
    "pgm": "adapters.pgm_se_adapter:PgmEstimator",
    "veragrid": "adapters.veragrid_se_adapter:VeragridEstimator",
    "sparlectra": "adapters.sparlectra_se_adapter:SparlectraEstimator",
}


def _instantiate(path: str):
    module, cls = path.split(":")
    return getattr(import_module(module), cls)()


# AC optimal power flow, by the same tool names.
OPTIMIZERS = {
    "pandapower": "adapters.pandapower_opf_adapter:PandapowerOptimizer",
    "veragrid": "adapters.veragrid_opf_adapter:VeragridOptimizer",
    "matpower": "adapters.matpower_opf_adapter:MatpowerOptimizer",
    "powermodels": "adapters.powermodels_opf_adapter:PowermodelsOptimizer",
}

REGISTRIES = {"pf": ADAPTERS, "se": ESTIMATORS, "opf": OPTIMIZERS}   # by the registry's `problem`


def get(name: str, problem: str = "pf"):
    """The tool's adapter for one problem ("pf", "se", "opf")."""
    return _instantiate(REGISTRIES[problem][name])


def get_adapter(name: str):
    return get(name, "pf")
