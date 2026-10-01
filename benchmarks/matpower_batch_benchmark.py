"""MATPOWER batch power flow (a loop of runpf in Octave): see adapters/matpower_batch_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("matpower", "batch")
