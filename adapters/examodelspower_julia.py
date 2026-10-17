"""Starts Julia in this process and loads GridBenchExaModelsPower (see
adapters/examodelspower_opf_adapter.py). A module of its own so that the memory
baseline (`SolverAdapter.modules`) includes the Julia runtime and the loaded
packages, as it includes an imported Python tool."""
from juliacall import Main as _jl

_jl.seval("using GridBenchExaModelsPower")
GB = _jl.GridBenchExaModelsPower
GB.quiet()
JULIA_VERSION = str(_jl.VERSION)
