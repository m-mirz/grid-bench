"""Starts Julia in this process and loads GridBenchExaPF on ExaPF's CUDA
backend (see adapters/exapf_gpu_adapter.py): the counterpart of
adapters/exapf_julia.py, with the same names.

Loading CUDA and CUDSS next to ExaPF activates ExaPF's CUDA extension
(`ExaPFCUDAExt`: `CUDABackend` models, cuDSS as the direct solver). Without
a usable GPU importing this raises CUDA.jl's reason, so every case records
it instead of falling back to the CPU. Scalar indexing of device arrays is
disallowed (`allowscalar(false)`, as ExaPF's own GPU tests do): an element
copied between host and device one at a time would raise instead of
silently dominating the timing."""
from juliacall import Main as _jl

_jl.seval("using CUDA, CUDSS, GridBenchExaPF")
_jl.seval("CUDA.functional(true)")   # raises CUDA.jl's reason when there is no usable GPU
_jl.seval("CUDA.allowscalar(false)")
GB = _jl.GridBenchExaPF
GB.quiet()
JULIA_VERSION = str(_jl.VERSION)
BACKEND = _jl.seval("CUDA.CUDABackend()")
FACTORIZATION = _jl.seval("CUDSS.CudssSolver")


def versions() -> dict[str, str]:
    """The Julia packages, plus the CUDA stack the numbers were measured on:
    the runtime the image carries, the host's driver and the device."""
    cuda = _jl.seval("""Dict(
        "CUDA.jl" => string(pkgversion(CUDA)), "CUDSS.jl" => string(pkgversion(CUDSS)),
        "cuda_runtime" => string(CUDA.CUDACore.runtime_version()),
        "cuda_driver" => string(CUDA.CUDACore.driver_version()),
        "gpu" => CUDA.name(CUDA.device()))""")
    return ({k: str(v) for k, v in GB.versions().items()} | {k: str(v) for k, v in cuda.items()}
            | {"julia": JULIA_VERSION})


def available_bytes() -> int:
    """Memory a model's arrays can still take: on a GPU, the device's free
    memory, which is where ExaPF keeps them."""
    return int(_jl.seval("CUDA.free_memory()"))
