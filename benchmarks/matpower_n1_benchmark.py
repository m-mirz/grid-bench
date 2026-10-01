"""MATPOWER N-1 (a loop of runpf in Octave): see adapters/matpower_n1_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("matpower", "n1")
