"""ExaModelsPower.jl (GPU): the same polar AC-OPF on ExaModels' CUDA backend,
solved by MadNLP with MadNLPGPU, the setup the package is built for. The
same model, input, joins, flat start and Julia code as
adapters/examodelspower_opf_adapter.py, which this subclasses; only the
backend and MadNLP's linear algebra differ, so the two rows of a case
compare the hardware and the KKT solve, not two problems. A separate tool
(own image, own JSON, the GPU tabs) because the image carries CUDA.jl, cuDSS
and MadNLPGPU, which the CPU image does not need.

Input, joins, what the model reads differently from MATPOWER, flat start,
`tol = OPF_TOLERANCE`, `max_iter = OPF_MAX_ITERATIONS`, a new solver per
solve, statuses: the CPU adapter's, unchanged (its docstring). Here:
- Backend: `CUDABackend()`. The parsed data is moved to the device
  (`convert_data`, as `ac_opf_model(path; backend)` does) and the model,
  its derivatives and every MadNLP vector live there; the flat start is
  written into the device's starting point in two bulk copies. One GPU, the
  one docker-compose passes in (`GRID_BENCH_GPU`, default device 0). Float64
  throughout, as on the CPU; `dependencies.gpu` records the device.
- MadNLP on the device, MadNLPGPU's own choices for a model there, set
  explicitly so a version bump cannot change them: `linear_solver =
  CUDSSSolver` (cuDSS, NVIDIA's sparse direct solver, an LDL^T/LU like MUMPS
  on the CPU); `kkt_system = SparseCondensedKKTSystem` (the inequality
  slacks condensed out, so the system cuDSS factorizes is positive definite
  near the solution); `equality_treatment = RelaxEquality` and
  `fixed_variable_treatment = RelaxBounds`, which the condensed system needs:
  each equality h(x) = 0 becomes -tol <= h(x) <= tol, fixed variables a
  band of that width. So the GPU solves the problem with its equalities
  relaxed by MadNLP's tolerance (scaled, 1e-6), where the CPU solves them
  exactly; the oracle grades both against the raw `.m`, and a relaxation
  that shows above its 1e-3 MVA balance limit is a finding, reported.
- Timing: each timed solve ends with a device synchronize, as kernels run
  asynchronously; MadNLP reads its convergence test back to the host every
  iteration anyway. `solution` (untimed) copies the result to the host.
- Compilation: the image is built without a GPU, so its precompile workload
  runs on the CPU only; CUDA kernels are compiled in each process on first
  use, in the untimed warm-up every test has.
- CUDA: CUDA.jl 5 and CUDSS 0.6 (cuDSS 0.7.1), the newest MadNLPGPU 0.9
  takes, and MadNLP 0.9, the newest ExaModelsPower 0.3.1 takes: an older
  stack than exapf_gpu's, on the same runtime (12.9, fixed at build time in
  tool-configs/examodelspower_gpu/julia/LocalPreferences.toml).
- Memory: `rss_*` is host memory, as for every tool; device memory is not
  measured.
- Julia single-threaded, ExaPowerIO's log off: as on the CPU.
"""
from adapters.examodelspower_opf_adapter import ExamodelspowerOptimizer

GPU_SETTINGS = {"backend": "cuda", "linear_solver": "cudss", "kkt_system": "sparse condensed",
                "equality_treatment": "relaxed by tol", "fixed_variable_treatment": "relaxed bounds"}


class ExamodelspowerGpuOptimizer(ExamodelspowerOptimizer):
    name = "examodelspower_gpu"
    display_name = "ExaModelsPower.jl (MadNLP, GPU)"
    color = "#e3494c"   # tools/palette.py EXAMODELSPOWER_GPU: ExaModelsPower's red, dash-dotted
    julia = "adapters.examodelspower_gpu_julia"
    modules = (julia,)
    settings = ExamodelspowerOptimizer.settings | GPU_SETTINGS
