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
for case9241pegase, 3.5 to 3.8x lightsim2grid on one thread of the same
machine (3.0, 7.6, 36.9 ms) but 5 to 6x behind it on 30 threads (0.15,
0.36, 2.0 ms), and 6 to 12x ExaPF's own CPU batch there. Per scenario the
batch is barely faster than the GPU's single solve from the flat start
(10.2 against 13.6 ms on case9241pegase; on mvlv10616 slower, 8.5 against
6.0 ms): the block formulation's extra dual arithmetic and its hardest
scenario's step count stay, so the gain is the device's factorization, not
the blocks running together.

Large sweeps (A100, 40 GB): 1 000 scenarios of case1354pegase and
case2869pegase fit, every scenario accepted, 0.22 and 0.55 ms per scenario
(4x the 100-scenario batch per scenario; lightsim2grid on 30 threads 0.16
and 0.33 ms, gpusim2grid 0.15 ms with 485 scenarios NaN and 0.37 ms with
every scenario accepted). The memory check refuses the rest before building:
case9241pegase and mvlv10616 at 1 000 (at least 131.5 and 49.7 GB of
duals) and every case at 10 000 (50.9 GB for case1354pegase).
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
