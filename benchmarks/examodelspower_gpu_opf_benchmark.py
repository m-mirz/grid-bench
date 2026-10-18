"""ExaModelsPower.jl (GPU) AC-OPF: see adapters/examodelspower_gpu_opf_adapter.py for inputs and settings."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("examodelspower_gpu", "opf")
