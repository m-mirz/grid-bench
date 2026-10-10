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
ForwardDiff duals alone exceed the device's free memory (after reclaiming
CUDA.jl's pool, see adapters/exapf_gpu_julia.py): at least 26.4 GB for 201
blocks of case9241pegase, which fits on a 40 GB A100 and solves there.

Known loss, reported rather than worked around (cuDSS 0.8.0 through
CUDSS.jl 0.8.0): case2869pegase#n1 fails, most outages not below tolerance,
where the CPU adapter passes all 200. The Jacobian and residual equal the
CPU's (2e-9, 8e-12) and every block has block 1's sparsity pattern, which
ExaPF's uniform batch requires; but one cuDSS factorization and solve of
that same input returns NaN in a different 30 to 39 of the 201 blocks each
time (block 1, the base case, among them) and every other block within
3e-13 of KLU. On this case it starts between 100 and 128 blocks (100:
none, 128: 5 to 8) and grows with their number; case1354pegase at 201
blocks and case9241pegase at 201 blocks of 18 000 rows are unaffected.

Results (A100, every other outage accepted): 0.52 ms per outage for
case1354pegase and 6.9 ms for case9241pegase, 5.5x and 4.5x lightsim2grid's
contingency engine on one thread of the same machine, 2.3x and 3.3x behind
it on 30 threads (0.23, 2.1 ms).
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
