"""Sparlectra.jl N-1 (a loop of runpf_rectangular! from the base solution): see adapters/sparlectra_n1_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("sparlectra", "n1")
