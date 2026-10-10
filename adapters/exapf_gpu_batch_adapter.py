"""ExaPF (GPU): batch power flow as ExaPF.jl's block formulation on its CUDA
backend (`native`).

adapters/exapf_batch_adapter.py on adapters/exapf_gpu_adapter.py's backend:
the same `BlockPolarForm`, scenario joins and convergence test, with the
`BatchJacobian` evaluated by GPU kernels and every Newton step one uniform
batch of cuDSS (one symbolic analysis shared by all blocks, as every block
has the same sparsity pattern). This is the workload ExaPF's block
formulation is built for: on the CPU the blocks run one after another, on
the GPU together. The thread count is not a setting (one GPU).

Memory: the ForwardDiff duals of every block live on the device (15.3 GB
for 100 blocks of case9241pegase, measured on the CPU), so a GPU with less
free memory records an out-of-memory failure for that case.

Results (A100, 100 scenarios from the base case, every scenario accepted):
0.86 ms per scenario for case1354pegase, 2.0 ms for case2869pegase, 10.2 ms
for case9241pegase, 3.9 to 4.2x lightsim2grid on one thread of the same
machine (3.3, 8.4, 42.5 ms) but about 3x behind it on 30 threads (0.27,
0.65, 3.3 ms), and 6 to 12x ExaPF's own CPU batch there. Per scenario the
batch is barely faster than the GPU's single solve from the flat start
(10.2 against 13.6 ms on case9241pegase; on mvlv10616 slower, 8.6 against
6.0 ms): the block formulation's extra dual arithmetic and its hardest
scenario's step count stay, so the gain is the device's factorization, not
the blocks running together.
"""
from adapters.exapf_batch_adapter import ExapfBatch
from adapters.exapf_gpu_adapter import GPU_SETTINGS, ExapfGpuAdapter


class ExapfGpuBatch(ExapfBatch):
    name = ExapfGpuAdapter.name
    display_name = ExapfGpuAdapter.display_name
    color = ExapfGpuAdapter.color
    julia = ExapfGpuAdapter.julia
    modules = ExapfGpuAdapter.modules
    settings = ExapfBatch.settings | GPU_SETTINGS
