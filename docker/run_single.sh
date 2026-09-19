#!/usr/bin/env bash
# One tool, optionally restricted: docker/run_single.sh pandapower --cases case14,case300
set -uo pipefail
cd "$(dirname "$0")/.."
tool="$1"; shift
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet HEAD -- . ":!results-docker" ":!results" ":!docs" 2>/dev/null || echo -dirty)"
mkdir -p results-docker data/.case-cache
compose="docker compose -f docker/docker-compose.yml"
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
$compose run --rm "$tool" pytest "benchmarks/${tool}_benchmark.py" "--benchmark-json=/output/$tool.json" "$@"

# Stop the Fuseki sidecar that `run` starts for cgmes2pgm.
$compose stop fuseki >/dev/null 2>&1
$compose rm -f fuseki >/dev/null 2>&1
