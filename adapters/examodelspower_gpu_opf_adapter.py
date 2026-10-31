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
  `fixed_variable_treatment = RelaxBound`, which the condensed system needs:
  every equality h(x) = 0 becomes a band -e <= h(x) <= e, every bound and
  fixed variable is widened by e, with e = `bound_relax_factor` x max(1,
  |bound|), in the model's units (p.u. for the balances), before scaling.
- `bound_relax_factor = 1e-8`: MadNLP's default, the one the CPU adapter
  runs with, so the bounds are widened alike; the equalities are not, as
  the CPU keeps MadNLP's `EnforceEquality` (balances to 1e-7 to 1e-10 MW
  there, 1e-5 to 7e-5 MW here, both well inside the oracle's 1e-3 MVA).
  Not MadNLPGPU's default for this system, which ties it to `tol` (1e-6). The balance
  mismatch is that band, scaled one to one with the factor (about 12x it in
  p.u., several relaxed rows adding up at a bus): at 1e-6 every case was
  rejected by the oracle, 1.2e-3 to 9.4e-3 MW against its 1e-3 MVA.
  Measured on an A100 (one process, the factor set per model; median of 3
  solves after a warm-up, in seconds, and max P
  mismatch in MW, "stall" = MAXIMUM_ITERATIONS_EXCEEDED at 200):

      bound_relax_factor     1e-6               1e-7              1e-8              1e-10
      case14                 0.078, 1.2e-3      0.067, 1.2e-4     0.074, 1.1e-5     0.084, 1.7e-8
      case1354pegase         0.421, 3.6e-3      0.356, 3.5e-4     0.394, 3.0e-5     0.407, 3.1e-8
      case2869pegase         0.589, 3.6e-3      0.562, 3.5e-4     0.542, 2.9e-5     0.599, 4.7e-8
      case9241pegase         1.887, 9.4e-3      stall             stall             stall
      case30000_goc          4.285, 3.0e-3      stall             stall             stall

  Up to case2869pegase every factor below 1e-6 is accepted at the same
  speed; case9241pegase and case30000_goc stall at every one of them, and
  still at 1000 iterations (1e-8), so the condensed system's GPU speed on
  those two exists only at a relaxation the oracle rejects. Not a matter of
  size: the four GOC cases between them in size converge at 1e-8 (Results).
  Not tuned past MadNLP's default: the factor is the CPU's, and the
  iteration limit everyone's. No exact-equality
  alternative runs on the device in this stack: `SparseKKTSystem` with
  `EnforceEquality` fails with `MakeParameter` (its multipliers need
  `jtprod!`, which ExaModelsPower's model is not built for, `prod=true`)
  and with `RelaxBound` (MadNLPGPU implements it for the condensed system
  only).
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

Results (A100, PGLib cases, results-docker/gpu/, against ExaModelsPower on
the same machine's CPU, which passes every case): 15 of 18 cases accepted,
balanced to 1e-5 to 7e-5 MW (the 1e-8 band), objectives within 5e-5 of
PGLib's reference, about as many iterations as on the CPU.
case9241pegase and case30000_goc fail: MAXIMUM_ITERATIONS_EXCEEDED (above).
So does case13659pegase in this run (de91c8b), accepted in 62 iterations
in the previous one with the same pins: the iteration counts on the device
vary from run to run (by 1 or 2 on the cases that converge), and this case
is on the edge. A stall is quick (case9241pegase's 200 iterations take
8.2 s); a process's first solve is not (89 s on case14, compiling the CUDA
kernels), which every fresh process pays: the memory measurement's, once
per case. Per solve the GPU is 15 to 18x slower on case14's variants (88
and 141 against 5.8 and 7.8 ms; case14 itself, the session's first solve,
got one round, 385 ms, as its warm-up included the compilation), behind
up to case300, and ahead from case1354pegase on: 3.0x there, 5.8x on
case2869pegase, and on the large cases 15x on case10000_goc and
case10480_goc, 17x on case19402_goc, 18x on case24464_goc (3.6 against
65 s).
"""
from adapters.examodelspower_opf_adapter import ExamodelspowerOptimizer

GPU_SETTINGS = {"backend": "cuda", "linear_solver": "cudss", "kkt_system": "sparse condensed",
                "equality_treatment": "relaxed", "fixed_variable_treatment": "relaxed bounds",
                "bound_relax_factor": 1e-8}


class ExamodelspowerGpuOptimizer(ExamodelspowerOptimizer):
    name = "examodelspower_gpu"
    display_name = "ExaModelsPower.jl (MadNLP, GPU)"
    color = "#95305c"   # tools/palette.py: Julia magenta, step 3, dash-dotted (GPU)
    julia = "adapters.examodelspower_gpu_julia"
    modules = (julia,)
    settings = ExamodelspowerOptimizer.settings | GPU_SETTINGS
