"""JuliaGrid.jl batch power flow (loop): see adapters/juliagrid_batch_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("juliagrid", "batch")
