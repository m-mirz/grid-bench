"""Tool registry. Imported lazily: each container installs only its own tool."""
from importlib import import_module

ADAPTERS = {
    "pandapower": "adapters.pandapower_adapter:PandapowerAdapter",
    "p3s": "adapters.p3s_adapter:P3sAdapter",   # slot 1's variant (tools/palette.py P3S)
    "lightsim2grid": "adapters.lightsim2grid_adapter:Lightsim2gridAdapter",
    "gpusim2grid": "adapters.gpusim2grid_adapter:Gpusim2gridAdapter",   # slot 2's variant (GPUSIM2GRID)
    "pypsa": "adapters.pypsa_adapter:PypsaAdapter",
    "pgm": "adapters.pgm_adapter:PgmAdapter",
    "pypowsybl": "adapters.pypowsybl_adapter:PypowsyblAdapter",
    "veragrid": "adapters.veragrid_adapter:VeragridAdapter",
    "cgmes2pgm": "adapters.cgmes2pgm_adapter:Cgmes2pgmAdapter",
    "sienna": "adapters.sienna_adapter:SiennaAdapter",
    "exapf": "adapters.exapf_adapter:ExapfAdapter",   # slot 8's variant (tools/palette.py EXAPF)
    "exapf_gpu": "adapters.exapf_gpu_adapter:ExapfGpuAdapter",   # slot 8's second variant (EXAPF_GPU)
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

# Batch power flow (operating-point sweeps), by the same tool names: a native
# batch API or a loop of single solves (adapters/batch_adapter.py).
BATCHES = {
    "pandapower": "adapters.pandapower_batch_adapter:PandapowerBatch",
    "p3s": "adapters.p3s_batch_adapter:P3sBatch",
    "lightsim2grid": "adapters.lightsim2grid_batch_adapter:Lightsim2gridBatch",
    "gpusim2grid": "adapters.gpusim2grid_batch_adapter:Gpusim2gridBatch",
    "pypsa": "adapters.pypsa_batch_adapter:PypsaBatch",
    "pgm": "adapters.pgm_batch_adapter:PgmBatch",
    "pypowsybl": "adapters.pypowsybl_batch_adapter:PypowsyblBatch",
    "veragrid": "adapters.veragrid_batch_adapter:VeragridBatch",
    "sienna": "adapters.sienna_batch_adapter:SiennaBatch",
    "exapf": "adapters.exapf_batch_adapter:ExapfBatch",
    "exapf_gpu": "adapters.exapf_gpu_batch_adapter:ExapfGpuBatch",
    "sparlectra": "adapters.sparlectra_batch_adapter:SparlectraBatch",
    "matpower": "adapters.matpower_batch_adapter:MatpowerBatch",
}

# N-1 contingency analysis, by the same tool names: a native contingency API
# or a loop of single solves (adapters/batch_adapter.py, ContingencyAdapter).
CONTINGENCIES = {
    "pandapower": "adapters.pandapower_n1_adapter:PandapowerN1",
    "p3s": "adapters.p3s_n1_adapter:P3sN1",
    "lightsim2grid": "adapters.lightsim2grid_n1_adapter:Lightsim2gridN1",
    "gpusim2grid": "adapters.gpusim2grid_n1_adapter:Gpusim2gridN1",
    "pypsa": "adapters.pypsa_n1_adapter:PypsaN1",
    "pgm": "adapters.pgm_n1_adapter:PgmN1",
    "pypowsybl": "adapters.pypowsybl_n1_adapter:PypowsyblN1",
    "veragrid": "adapters.veragrid_n1_adapter:VeragridN1",
    "sienna": "adapters.sienna_n1_adapter:SiennaN1",
    "exapf": "adapters.exapf_n1_adapter:ExapfN1",
    "exapf_gpu": "adapters.exapf_gpu_n1_adapter:ExapfGpuN1",
    "sparlectra": "adapters.sparlectra_n1_adapter:SparlectraN1",
    "matpower": "adapters.matpower_n1_adapter:MatpowerN1",
}

# CIM import, export and validation (adapters/cim_adapter.py): CIM libraries,
# most with no power-flow adapter. pypowsybl is the one tool in both.
CIM = {
    "cimoxide": "adapters.cimoxide_cim_adapter:CimoxideCim",
    "triplets": "adapters.triplets_cim_adapter:TripletsCim",
    "opencgmes": "adapters.opencgmes_cim_adapter:OpencgmesCim",
    "powsybl": "adapters.powsybl_cim_adapter:PowsyblCim",
    "pypowsybl": "adapters.pypowsybl_cim_adapter:PypowsyblCim",
}

REGISTRIES = {"pf": ADAPTERS, "se": ESTIMATORS, "opf": OPTIMIZERS, "batch": BATCHES,
              "n1": CONTINGENCIES, "cim": CIM}   # by the registry's `problem`


def get(name: str, problem: str = "pf"):
    """The tool's adapter for one problem ("pf", "se", "opf", "batch", "n1", "cim")."""
    return _instantiate(REGISTRIES[problem][name])


def get_adapter(name: str):
    return get(name, "pf")
