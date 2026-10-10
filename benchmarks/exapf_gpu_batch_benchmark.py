"""ExaPF (GPU) batch power flow (BlockPolarForm): see adapters/exapf_gpu_batch_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("exapf_gpu", "batch")
