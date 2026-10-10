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

Results (A100, 100 scenarios, every scenario accepted): 1.0 ms per scenario
for case1354pegase, 2.4 ms for case2869pegase, 12.0 ms for case9241pegase,
4.6x lightsim2grid on one thread of the same machine but 3x behind it on
30 threads (0.30, 0.82, 3.7 ms), and 6 to 13x ExaPF's own CPU batch there.
Per scenario the batch is barely faster than the GPU's single solve
(12.0 against 13.6 ms on case9241pegase; on mvlv10616 slower, 8.5 against
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
