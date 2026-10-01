"""pypowsybl batch power flow (a loop of run_ac): see adapters/pypowsybl_batch_adapter.py."""
from benchmarks.benchmark_template import create_benchmarks

create_benchmarks("pypowsybl", "batch")
