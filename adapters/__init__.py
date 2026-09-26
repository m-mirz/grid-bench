"""Tool registry. Imported lazily: each container installs only its own tool."""
from importlib import import_module

ADAPTERS = {
    "pandapower": "adapters.pandapower_adapter:PandapowerAdapter",
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


def get_adapter(name: str):
    return _instantiate(ADAPTERS[name])


def get_estimator(name: str):
    return _instantiate(ESTIMATORS[name])
