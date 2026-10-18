"""Starts Julia in this process and loads GridBenchExaModelsPower on ExaModels'
CUDA backend with MadNLPGPU (see adapters/examodelspower_gpu_opf_adapter.py):
the counterpart of adapters/examodelspower_julia.py, with the same names.

Loading CUDA and CUDSS next to MadNLPGPU activates its CUDA extension
(cuDSS as the KKT solver, device arrays throughout). Without a usable GPU
importing this raises CUDA.jl's reason, so every case records it instead
of falling back to the CPU. Scalar indexing of device arrays is disallowed
(`allowscalar(false)`), as in adapters/exapf_gpu_julia.py: an element
copied between host and device one at a time raises instead of silently
dominating the timing."""
from juliacall import Main as _jl

_jl.seval("using CUDA, CUDSS, MadNLP, MadNLPGPU, GridBenchExaModelsPower")
_jl.seval("CUDA.functional(true)")   # raises CUDA.jl's reason when there is no usable GPU
_jl.seval("CUDA.allowscalar(false)")
GB = _jl.GridBenchExaModelsPower
GB.quiet()
JULIA_VERSION = str(_jl.VERSION)

# MadNLPGPU's own choices for a model on the device (MadNLPGPUCUDAExt's
# MadNLPOptions, and MadNLP's defaults for the condensed system), set
# explicitly so that a version bump cannot change them silently (the
# adapter's docstring).
_load = _jl.seval("""(path, tol, max_iterations) -> GridBenchExaModelsPower.load(path, tol, max_iterations;
    backend = CUDA.CUDABackend(),
    options = (linear_solver = MadNLPGPU.CUDSSSolver, kkt_system = MadNLP.SparseCondensedKKTSystem,
               equality_treatment = MadNLP.RelaxEquality, fixed_variable_treatment = MadNLP.RelaxBound,
               bound_relax_factor = tol))""")
# Kernels run asynchronously: the timed solve ends when the device is done.
_solve = _jl.seval("m -> (r = GridBenchExaModelsPower.solve!(m); CUDA.synchronize(); r)")


def load(path: str, tol: float, max_iterations: int):
    return _load(path, tol, max_iterations)


def solve(model) -> tuple[int, str]:
    return _solve(model)


def versions() -> dict[str, str]:
    """The Julia packages, plus the CUDA stack the numbers were measured on:
    the runtime the image carries, the host's driver and the device."""
    cuda = _jl.seval("""Dict(
        "MadNLPGPU" => string(pkgversion(MadNLPGPU)),
        "CUDA.jl" => string(pkgversion(CUDA)), "CUDSS.jl" => string(pkgversion(CUDSS)),
        "cuda_runtime" => string(CUDA.runtime_version()),
        "cuda_driver" => string(CUDA.driver_version()),
        "gpu" => CUDA.name(CUDA.device()))""")
    return ({k: str(v) for k, v in GB.versions().items()} | {k: str(v) for k, v in cuda.items()}
            | {"julia": JULIA_VERSION})
