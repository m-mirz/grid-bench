"""gridoxide (GPU) batch power flow (GpuBatchSolver.solve_batch, base-case start): see adapters/gridoxide_gpu_batch_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("gridoxide_gpu", "batch")
