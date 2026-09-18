#!/usr/bin/env bash
# Builds the base image, the harness image and one image per tool.
# Usage: docker/build.sh [tool ...]   (default: harness + every tool-configs/ entry)
set -euo pipefail
cd "$(dirname "$0")/.."
export DOCKER_BUILDKIT=1

docker build -f docker/base.dockerfile -t grid-bench/base:latest .
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    targets=(harness $(ls tool-configs | grep -v '^harness$'))
fi
for t in "${targets[@]}"; do
    echo "=== building grid-bench/$t"
    docker build -f docker/tool.dockerfile --build-arg TOOL="$t" -t "grid-bench/$t:latest" .
done
