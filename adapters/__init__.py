"""Tool registry. Imported lazily: each container installs only its own tool."""
from importlib import import_module

# Order: by language, as tools/palette.py colours them (Python, C++, Julia,
# Java, then the reference); within one, the palette's step order.
ADAPTERS = {
    "pandapower": "adapters.pandapower_adapter:PandapowerAdapter",
    "pypsa": "adapters.pypsa_adapter:PypsaAdapter",
    "veragrid": "adapters.veragrid_adapter:VeragridAdapter",
    "lightsim2grid": "adapters.lightsim2grid_adapter:Lightsim2gridAdapter",
    "p3s": "adapters.p3s_adapter:P3sAdapter",
    "pgm": "adapters.pgm_adapter:PgmAdapter",
    "cgmes2pgm": "adapters.cgmes2pgm_adapter:Cgmes2pgmAdapter",
    "gpusim2grid": "adapters.gpusim2grid_adapter:Gpusim2gridAdapter",
    "sparlectra": "adapters.sparlectra_adapter:SparlectraAdapter",
    "juliagrid": "adapters.juliagrid_adapter:JuliagridAdapter",
    "sienna": "adapters.sienna_adapter:SiennaAdapter",
    "exapf": "adapters.exapf_adapter:ExapfAdapter",
    "exapf_gpu": "adapters.exapf_gpu_adapter:ExapfGpuAdapter",
    "pypowsybl": "adapters.pypowsybl_adapter:PypowsyblAdapter",
    "matpower": "adapters.matpower_adapter:MatpowerAdapter",
}

# State estimators, by the same tool names (a tool's colour and container are
# its power-flow adapter's). Order: ADAPTERS order.
ESTIMATORS = {
    "pandapower": "adapters.pandapower_se_adapter:PandapowerEstimator",
    "veragrid": "adapters.veragrid_se_adapter:VeragridEstimator",
    "pgm": "adapters.pgm_se_adapter:PgmEstimator",
    "sparlectra": "adapters.sparlectra_se_adapter:SparlectraEstimator",
    "juliagrid": "adapters.juliagrid_se_adapter:JuliagridEstimator",
}


def _instantiate(path: str):
    module, cls = path.split(":")
    return getattr(import_module(module), cls)()


# AC optimal power flow, by the same tool names.
OPTIMIZERS = {
    "pandapower": "adapters.pandapower_opf_adapter:PandapowerOptimizer",
    "veragrid": "adapters.veragrid_opf_adapter:VeragridOptimizer",
    "examodelspower": "adapters.examodelspower_opf_adapter:ExamodelspowerOptimizer",
    "examodelspower_gpu": "adapters.examodelspower_gpu_opf_adapter:ExamodelspowerGpuOptimizer",
    "juliagrid": "adapters.juliagrid_opf_adapter:JuliagridOptimizer",
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
    "juliagrid": "adapters.juliagrid_batch_adapter:JuliagridBatch",
    "matpower": "adapters.matpower_batch_adapter:MatpowerBatch",
}

# N-1 contingency analysis, by the same tool names: a native contingency API
# or a loop of single solves (adapters/batch_adapter.py, ContingencyAdapter).
CONTINGENCIES = {
    "pandapower": "adapters.pandapower_n1_adapter:PandapowerN1",
    "p3s": "adapters.p3s_n1_adapter:P3sN1",
    "lightsim2grid": "adapters.lightsim2grid_n1_adapter:Lightsim2gridN1",
    "gpusim2grid": "adapters.gpusim2grid_n1_adapter:Gpusim2gridN1",
    "pgm": "adapters.pgm_n1_adapter:PgmN1",
    "pypowsybl": "adapters.pypowsybl_n1_adapter:PypowsyblN1",
    "veragrid": "adapters.veragrid_n1_adapter:VeragridN1",
    "sienna": "adapters.sienna_n1_adapter:SiennaN1",
    "exapf": "adapters.exapf_n1_adapter:ExapfN1",
    "exapf_gpu": "adapters.exapf_gpu_n1_adapter:ExapfGpuN1",
    "sparlectra": "adapters.sparlectra_n1_adapter:SparlectraN1",
    "juliagrid": "adapters.juliagrid_n1_adapter:JuliagridN1",
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

# DC power flow and DC N-1 (adapters/dc_adapter.py), by the same tool names.
DC = {
    "lightsim2grid": "adapters.lightsim2grid_dc_adapter:Lightsim2gridDc",
    "pypsa": "adapters.pypsa_dc_adapter:PypsaDc",
    "pypowsybl": "adapters.pypowsybl_dc_adapter:PypowsyblDc",
    "toop": "adapters.toop_dc_adapter:ToopDc",
    "matpower": "adapters.matpower_dc_adapter:MatpowerDc",
}
DC_CONTINGENCIES = {
    "lightsim2grid": "adapters.lightsim2grid_dc_adapter:Lightsim2gridDcN1",
    "lightsim2grid_lodf": "adapters.lightsim2grid_dc_adapter:Lightsim2gridDcN1Lodf",
    "pypowsybl": "adapters.pypowsybl_dc_adapter:PypowsyblDcN1",
    "toop": "adapters.toop_dc_adapter:ToopDcN1",
    "matpower": "adapters.matpower_dc_adapter:MatpowerDcN1",
}

REGISTRIES = {"pf": ADAPTERS, "se": ESTIMATORS, "opf": OPTIMIZERS, "batch": BATCHES,
              "n1": CONTINGENCIES, "dc": DC, "dcn1": DC_CONTINGENCIES, "cim": CIM}   # by the registry's `problem`


def get(name: str, problem: str = "pf"):
    """The tool's adapter for one problem ("pf", "se", "opf", "batch", "n1", "dc", "dcn1", "cim")."""
    return _instantiate(REGISTRIES[problem][name])


def get_adapter(name: str):
    return get(name, "pf")
