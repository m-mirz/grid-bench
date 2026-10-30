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
    # gridoxide is compiled from source (tool-configs/gridoxide/Dockerfile), as
    # its published wheel has neither KLU nor IPOPT. By default the source is
    # the pinned release's tarball from GitHub, checked against its sha256.
    # To benchmark unreleased work instead, set GRIDOXIDE_REPO to a gridoxide
    # checkout and GRIDOXIDE_COMMIT to the commit to build (default HEAD);
    # `git archive` exports exactly that commit, never uncommitted work. Either
    # way the tree becomes the `gridoxide` build context.
    if [ "$t" = gridoxide ]; then
        src=$(mktemp -d)
        trap 'rm -rf "$src"' EXIT
        if [ -n "${GRIDOXIDE_REPO:-}" ]; then
            commit=$(git -C "$GRIDOXIDE_REPO" rev-parse "${GRIDOXIDE_COMMIT:-HEAD}")
            git -C "$GRIDOXIDE_REPO" archive "$commit" | tar -x -C "$src"
        else
            version=0.3.0
            commit=e85f8082bcb15d494c9c3f4ce92647da079dac0b
            sha256=8e7456120a0929b4fe36c31df770d087b4acfd6a6f00e570d8cb19eeca18c804
            curl -sSfL -o "$src/gridoxide.tar.gz" \
                "https://github.com/m-mirz/gridoxide/archive/refs/tags/v$version.tar.gz"
            echo "$sha256  $src/gridoxide.tar.gz" | sha256sum -c --quiet
            tar -xzf "$src/gridoxide.tar.gz" -C "$src" --strip-components=1
            rm "$src/gridoxide.tar.gz"
        fi
        extra=(--build-context "gridoxide=$src" --build-arg "GRIDOXIDE_COMMIT=$commit")
    fi
    docker build -f "$dockerfile" --build-arg TOOL="$t" "${extra[@]}" -t "grid-bench/$t:latest" .
done
