"""ExaPF (GPU) N-1 (line-contingency block formulation): see adapters/exapf_gpu_n1_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("exapf_gpu", "n1")
