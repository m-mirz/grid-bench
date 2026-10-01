"""Sparlectra.jl batch power flow (a loop of runpf_rectangular!): see adapters/sparlectra_batch_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("sparlectra", "batch")
