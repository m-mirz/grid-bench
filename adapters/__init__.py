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
}


def get_adapter(name: str):
    module, cls = ADAPTERS[name].split(":")
    return getattr(import_module(module), cls)()
