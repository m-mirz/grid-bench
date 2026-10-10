"""Starts Julia in this process and loads GridBenchExaPF on ExaPF's CPU
backend (see adapters/exapf_adapter.py). A module of its own so that the
memory baseline (`SolverAdapter.modules`) includes the Julia runtime and the
loaded packages, as it includes an imported Python tool.
adapters/exapf_gpu_julia.py is its CUDA counterpart: the same names, the
adapters use nothing else."""
from pathlib import Path

from juliacall import Main as _jl

_jl.seval("using GridBenchExaPF")
GB = _jl.GridBenchExaPF
GB.quiet()
JULIA_VERSION = str(_jl.VERSION)
BACKEND = _jl.seval("GridBenchExaPF.ExaPF.CPU()")
FACTORIZATION = _jl.seval("GridBenchExaPF.KLU.KLUFactorization")


def versions() -> dict[str, str]:
    return {k: str(v) for k, v in GB.versions().items()} | {"julia": JULIA_VERSION}


def available_bytes() -> int:
    """Memory a model's arrays can still take: on the CPU backend, what the
    kernel reports available (`MemAvailable`)."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not in /proc/meminfo")
