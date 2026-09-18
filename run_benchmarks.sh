#!/usr/bin/env bash
# Native run of every tool from one environment, for development.
# Usage: ./run_benchmarks.sh [tool ...] [-- pytest args, e.g. --groups smoke]
set -uo pipefail
cd "$(dirname "$0")"
tools=(); extra=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
    tools+=("$1"); shift
done
[ ${#tools[@]} -eq 0 ] && tools=(pandapower lightsim2grid pypsa pgm pypowsybl veragrid)
export GRID_BENCH_RESULTS=results GIT_SHA="$(git rev-parse HEAD)$(git diff --quiet || echo -dirty)"
mkdir -p results
uv run python -m cases.prep
uv run python -m oracle.check_conversion results
for t in "${tools[@]}"; do
    uv run pytest "benchmarks/${t}_benchmark.py" "--benchmark-json=results/$t.json" "${extra[@]}"
done
uv run python -m tools.generate_all results
