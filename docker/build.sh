#!/usr/bin/env bash
# Builds the base image, the harness image and one image per tool.
# Usage: docker/build.sh [tool ...]   (default: harness + every tool-configs/ entry)
set -euo pipefail
cd "$(dirname "$0")/.."
export DOCKER_BUILDKIT=1

docker build -f docker/base.dockerfile -t grid-bench/base:latest .
# Fuseki, the SPARQL store cgmes2pgm reads from (see docker/fuseki/PROVENANCE.md).
docker build -t grid-bench/fuseki:latest docker/fuseki
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    targets=(harness $(ls tool-configs | grep -v '^harness$'))
fi
for t in "${targets[@]}"; do
    echo "=== building grid-bench/$t"
    # A tool that needs more than Python packages (sienna: Julia) brings its own
    # Dockerfile; one on the JVM (a pom.xml) shares docker/java-tool.dockerfile.
    dockerfile=docker/tool.dockerfile
    [ -f "tool-configs/$t/pom.xml" ] && dockerfile=docker/java-tool.dockerfile
    [ -f "tool-configs/$t/Dockerfile" ] && dockerfile="tool-configs/$t/Dockerfile"
    extra=()
    # gridoxide is compiled from a commit, not a release (tool-configs/gridoxide/
    # Dockerfile): GRIDOXIDE_REPO is a gridoxide checkout, GRIDOXIDE_COMMIT the
    # commit to build (default HEAD). `git archive` exports exactly that commit,
    # never uncommitted work, into the `gridoxide` build context.
    if [ "$t" = gridoxide ]; then
        : "${GRIDOXIDE_REPO:?set GRIDOXIDE_REPO to a gridoxide checkout}"
        commit=$(git -C "$GRIDOXIDE_REPO" rev-parse "${GRIDOXIDE_COMMIT:-HEAD}")
        src=$(mktemp -d)
        trap 'rm -rf "$src"' EXIT
        git -C "$GRIDOXIDE_REPO" archive "$commit" | tar -x -C "$src"
        extra=(--build-context "gridoxide=$src" --build-arg "GRIDOXIDE_COMMIT=$commit")
    fi
    docker build -f "$dockerfile" --build-arg TOOL="$t" "${extra[@]}" -t "grid-bench/$t:latest" .
done
