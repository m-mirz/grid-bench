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
export GRID_BENCH_RESULTS=results GIT_SHA="$(git rev-parse HEAD)$(git diff --quiet HEAD -- . ":!results-docker" ":!results" ":!docs" || echo -dirty)"
mkdir -p results
# The case selection of pytest's arguments (`--groups g` or `--cases a,b`), if
# any: a case only in an opt-in group (`--groups large`, cases.registry.OPT_IN_GROUPS)
# is prepared only when a run asks for it.
selection_of() { while [ $# -gt 0 ]; do case "$1" in --groups|--cases) echo "$1 ${2:-}"; return;;
    --groups=*|--cases=*) echo "${1%%=*} ${1#*=}"; return;; esac; shift; done; }
uv run python -m cases.prep
selection="$(selection_of "${extra[@]}")"
[ -n "$selection" ] && uv run python -m cases.prep $selection
uv run python -m oracle.check_conversion results
for t in "${tools[@]}"; do
    uv run pytest "benchmarks/${t}_benchmark.py" "--benchmark-json=results/$t.json" "${extra[@]}"
done
uv run python -m tools.generate_all results
