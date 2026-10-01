"""PyPSA batch power flow (pf over snapshots): see adapters/pypsa_batch_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("pypsa", "batch")
