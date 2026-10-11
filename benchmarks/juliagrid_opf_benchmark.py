"""JuliaGrid.jl AC-OPF (Ipopt): see adapters/juliagrid_opf_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("juliagrid", "opf")
