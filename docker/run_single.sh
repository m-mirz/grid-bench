#!/usr/bin/env bash
# One tool, optionally restricted: docker/run_single.sh pandapower --cases case14,case300
set -uo pipefail
cd "$(dirname "$0")/.."
tool="$1"; shift
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo -dirty)"
mkdir -p results-docker data/.case-cache
compose="docker compose -f docker/docker-compose.yml"
$compose run --rm prep
$compose run --rm prep-pypowsybl
$compose run --rm conversion-check
$compose run --rm "$tool" pytest "benchmarks/${tool}_benchmark.py" "--benchmark-json=/output/$tool.json" "$@"
