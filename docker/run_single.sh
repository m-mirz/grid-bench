#!/usr/bin/env bash
# One tool, optionally restricted: docker/run_single.sh pandapower --cases case14,case300
set -uo pipefail
cd "$(dirname "$0")/.."
tool="$1"; shift
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet HEAD -- . ":!results-docker" ":!results" ":!docs" 2>/dev/null || echo -dirty)"
mkdir -p results-docker data/.case-cache
compose="docker compose -f docker/docker-compose.yml"
# Batch benchmarks measure a tool's own parallelism: BLAS stays on one thread.
blas_env() { [[ "$1" = _batch || "$1" = _n1 ]] && echo "-e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1"; }
$compose run --rm prep
# pypowsybl's MATPOWER -> CGMES conversion needs its image; without it (CI
# builds only what the tool under test needs) the converted-pypowsybl cases
# are not prepared, rather than Compose trying to pull a local-only image.
if docker image inspect grid-bench/pypowsybl:latest >/dev/null 2>&1; then
    $compose run --rm prep-pypowsybl
else
    echo "no grid-bench/pypowsybl image: pypowsybl conversions not prepared"
fi
$compose run --rm conversion-check
# Every problem the tool has a benchmark for (power flow, state estimation,
# OPF, batch power flow, N-1, CIM), in the same container, a JSON each: <tool>.json, <tool>-se.json, ...
for suffix in "" _se _opf _batch _n1 _cim; do
    if [ -f "benchmarks/${tool}${suffix}_benchmark.py" ]; then
        $compose run --rm $(blas_env "$suffix") "$tool" pytest "benchmarks/${tool}${suffix}_benchmark.py" \
            "--benchmark-json=/output/$tool${suffix/_/-}.json" "$@"
    fi
done

# Stop the Fuseki sidecar that `run` starts for cgmes2pgm.
$compose stop fuseki >/dev/null 2>&1
$compose rm -f fuseki >/dev/null 2>&1
