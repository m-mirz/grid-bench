"""Starts Julia in this process and loads GridBenchExaModelsPower (see
adapters/examodelspower_opf_adapter.py). A module of its own so that the memory
baseline (`SolverAdapter.modules`) includes the Julia runtime and the loaded
packages, as it includes an imported Python tool. The CPU backend: its
counterpart adapters/examodelspower_gpu_julia.py has the same names."""
from juliacall import Main as _jl

_jl.seval("using GridBenchExaModelsPower")
GB = _jl.GridBenchExaModelsPower
GB.quiet()
JULIA_VERSION = str(_jl.VERSION)


def load(path: str, tol: float, max_iterations: int):
    return GB.load(path, tol, max_iterations)   # its defaults: no backend, MUMPS


def solve(model) -> tuple[int, str]:
    return GB.solve_b(model)   # juliacall spells solve! as solve_b


def versions() -> dict[str, str]:
    return {k: str(v) for k, v in GB.versions().items()} | {"julia": JULIA_VERSION}
