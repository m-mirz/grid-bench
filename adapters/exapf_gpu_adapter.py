"""ExaPF (GPU): ExaPF.jl's polar Newton-Raphson on its CUDA backend, the
backend it is written for. The same model, input and Julia code as
adapters/exapf_adapter.py, which this subclasses; only the backend and the
linear solver differ, so the two rows of a case compare the hardware and
the solver, not two problems. A separate tool (own image, own JSON) because
the image carries CUDA.jl and cuDSS, which the CPU image does not need.

Not yet run: written and built without a GPU (the CPU path it shares was
checked to be bit-identical to before), so its first run on an NVIDIA GPU
is its test (`docker/build.sh exapf_gpu && docker/run_single.sh exapf_gpu
--groups smoke`). A correct run shows the CPU adapter's oracle residuals
on case14, around 1e-9 MVA or below.

Input, bus-id mapping, flat start, tolerance, iteration limit and known
losses: the CPU adapter's, unchanged (its docstring). Here:

- Backend: `CUDABackend()` (KernelAbstractions), with ExaPF's CUDA extension
  (`ExaPFCUDAExt`), loaded by adapters/exapf_gpu_julia.py, which raises if
  CUDA is not functional rather than fall back to the CPU. One GPU, the one
  docker-compose passes in (`GRID_BENCH_GPU`, default device 0). Float64
  throughout, as on the CPU: a GPU's FP64 throughput is its own (a fraction
  of FP32 on consumer cards), which belongs to the result, so
  `grid_bench.tools.exapf_gpu.dependencies.gpu` records the device.
- Linear solver: ExaPF's GPU default, `DirectSolver` on cuDSS (`CudssSolver`,
  asserted at load), refactorized in place each step on the symbolic
  analysis of the first. A direct solver like KLU on the CPU, so both rows
  solve each Newton step exactly. Not ExaPF's Krylov solvers (BiCGSTAB,
  DQGMRES): they solve the step to `1e-3` of the mismatch, an inexact
  Newton method, and their convergence depends on a preconditioner.
- Timing: each timed call ends with a device synchronize
  (`GridBenchExaPF.sync`), as kernels run asynchronously; without it the
  timer would stop at the last launch. The flat start is restored on the
  device, inside the call. The single solve leaves its voltages on the
  device, and `solution` (untimed, as for every tool) copies them out; the
  batch and N-1 calls copy theirs to the host inside the call, as they do on
  the CPU.
- Compilation: the image is built without a GPU, so its precompile workload
  runs on the CPU only, and CUDA kernels are compiled in each process on
  first use: in the untimed warm-up load and solve every test has. The
  first case of a process is therefore slower to import, not to solve.
- CUDA: the runtime is fixed at build time (`CUDA_Runtime_jll` version
  preference, tool-configs/exapf_gpu/julia/LocalPreferences.toml), since
  the build has no driver to choose it from; the host needs a driver for
  that CUDA major version. `dependencies` records runtime and driver.
- Memory: `rss_*` in the records is host memory, as for every tool. Device
  memory, where ExaPF keeps the model, is not measured.
- Julia single-threaded, logs off: as on the CPU.
"""
from adapters.exapf_adapter import ExapfAdapter

GPU_SETTINGS = {"backend": "cuda", "linear_solver": "cudss"}


class ExapfGpuAdapter(ExapfAdapter):
    name = "exapf_gpu"
    display_name = "ExaPF.jl (GPU)"
    color = "#e3494a"   # tools/palette.py EXAPF_GPU: Sienna's red, dash-dotted
    julia = "adapters.exapf_gpu_julia"
    modules = (julia,)
    settings = ExapfAdapter.settings | GPU_SETTINGS
