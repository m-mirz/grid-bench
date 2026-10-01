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
export GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet HEAD -- . ":!results-docker" ":!results" ":!docs" 2>/dev/null || echo -dirty)"
compose="docker compose -f docker/docker-compose.yml"
# Batch benchmarks measure a tool's own parallelism: BLAS stays on one thread.
blas_env() { [[ "$1" = _batch || "$1" = _n1 ]] && echo "-e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1"; }
mkdir -p results-docker data/.case-cache

$compose run --rm prep || { echo "input preparation failed"; exit 1; }
$compose run --rm prep-pypowsybl || { echo "pypowsybl conversion failed"; exit 1; }
$compose run --rm conversion-check || { echo "conversion check failed"; exit 1; }
$compose run --rm oracle-tests || { echo "oracle tests failed; refusing to benchmark"; exit 1; }

failed=()
for t in "${tools[@]}"; do
    echo "=== $t"
    # Every problem the tool has a benchmark for (power flow, state estimation,
    # OPF, batch power flow, N-1), in the same container, a JSON each: <tool>.json, <tool>-se.json, ...
    for suffix in "" _se _opf _batch _n1; do
        if [ -f "benchmarks/${t}${suffix}_benchmark.py" ]; then
            $compose run --rm $(blas_env "$suffix") "$t" pytest "benchmarks/${t}${suffix}_benchmark.py" \
                "--benchmark-json=/output/$t${suffix/_/-}.json" "${extra[@]}" || failed+=("$t${suffix/_/-}")
        fi
    done
done

$compose run --rm reports
# pytest exits non-zero whenever any case fails; that is a result, not an
# error of the sweep. Only a missing JSON means the tool did not run.
for t in "${failed[@]}"; do
    [ -f "results-docker/$t.json" ] || echo "WARNING: $t produced no results"
done

# Stop the Fuseki sidecar that `run` starts for cgmes2pgm.
$compose stop fuseki >/dev/null 2>&1
$compose rm -f fuseki >/dev/null 2>&1
