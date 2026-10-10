"""ExaPF (GPU): N-1 contingency analysis through ExaPF.jl's line-contingency
formulation on its CUDA backend (`native`).

adapters/exapf_n1_adapter.py on adapters/exapf_gpu_adapter.py's backend:
the base case from flat start, then every outage block from its solution,
in one timed call, with every Newton step one uniform batch of cuDSS.
`BatchJacobian` gives every block the same sparsity pattern (an outage's
admittances are zeroed, not removed), which a uniform batch requires; the
oracle grades every outage, so a block solved on another block's pattern
would show there.

Memory: `load` refuses, before building anything, a system whose
ForwardDiff duals alone exceed the device's free memory (`CUDA.free_memory`):
at least 26.4 GB for 201 blocks of case9241pegase.
"""
from adapters.exapf_gpu_adapter import GPU_SETTINGS, ExapfGpuAdapter
from adapters.exapf_n1_adapter import ExapfN1


class ExapfGpuN1(ExapfN1):
    name = ExapfGpuAdapter.name
    display_name = ExapfGpuAdapter.display_name
    color = ExapfGpuAdapter.color
    julia = ExapfGpuAdapter.julia
    modules = ExapfGpuAdapter.modules
    settings = ExapfN1.settings | GPU_SETTINGS
