"""p3s batch power flow (C++/KLU solve_batch, OpenMP): see adapters/p3s_batch_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("p3s", "batch")
