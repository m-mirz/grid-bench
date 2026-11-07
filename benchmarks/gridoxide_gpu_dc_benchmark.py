"""gridoxide (GPU) DC power flow (GpuDcSolver.solve_single): see adapters/gridoxide_gpu_dc_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("gridoxide_gpu", "dc")
