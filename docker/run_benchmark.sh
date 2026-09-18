#!/usr/bin/env bash
# Full run: prepare inputs, test the oracle, benchmark every tool one at a
# time, then regenerate reports. A tool that fails does not stop the sweep.
# Usage: docker/run_benchmark.sh [tool ...] [-- extra pytest args, e.g. --groups smoke]
#
# Publish only numbers from one sweep on one otherwise idle machine. For an
# A/B comparison, interleave the runs (A, B, A, B): two sweeps taken minutes
# apart also measure the machine's thermal state.
set -uo pipefail
cd "$(dirname "$0")/.."

tools=(); extra=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
    tools+=("$1"); shift
done
[ ${#tools[@]} -eq 0 ] && tools=($(ls tool-configs | grep -v '^harness$'))

export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo -dirty)"
compose="docker compose -f docker/docker-compose.yml"
mkdir -p results-docker data/.case-cache

$compose run --rm prep || { echo "input preparation failed"; exit 1; }
$compose run --rm oracle-tests || { echo "oracle tests failed; refusing to benchmark"; exit 1; }

failed=()
for t in "${tools[@]}"; do
    echo "=== $t"
    $compose run --rm "$t" pytest "benchmarks/${t}_benchmark.py" "--benchmark-json=/output/$t.json" "${extra[@]}" \
        || failed+=("$t")
done

$compose run --rm reports
# pytest exits non-zero whenever any case fails; that is a result, not an
# error of the sweep. Only a missing JSON means the tool did not run.
for t in "${failed[@]}"; do
    [ -f "results-docker/$t.json" ] || echo "WARNING: $t produced no results"
done
