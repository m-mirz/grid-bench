#!/usr/bin/env bash
# Re-resolves every lockfile (each tool image's and the root one), using the
# pinned uv inside the base image so the lockfile format matches the builds.
# Run it deliberately, review the diff, and commit it: images install exactly
# these lockfiles (`uv sync --frozen`). The `exclude-newer = "P7D"` rule in
# each pyproject.toml applies here, at resolution time.
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e UV_CACHE_DIR=/tmp/lockcache \
    -v "$PWD":/repo -w /repo grid-bench/base:latest bash -c '
    for d in tool-configs/*/; do (cd "$d" && uv lock && echo "locked $d"); done
    uv lock && echo "locked ./"'
# Java libraries (tool-configs/*/pom.xml): jars.sha256, the SHA-256 of every
# jar Maven resolves for the pom, in the Maven image docker/java-tool.dockerfile
# builds with. The build checks the jars against it.
maven_image=$(sed -n 's/^FROM \(maven:[^ ]*\) AS jars$/\1/p' docker/java-tool.dockerfile)
for pom in tool-configs/*/pom.xml; do
    d=$(dirname "$pom")
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD/$d":/proj:ro --entrypoint bash "$maven_image" -c '
        mvn -q -B -f /proj/pom.xml -Dmaven.repo.local=/tmp/m2 dependency:copy-dependencies \
            -DoutputDirectory=/tmp/lib -DincludeScope=runtime && cd /tmp/lib && sha256sum *' > "$d/jars.sha256" \
        && echo "locked $d/jars.sha256"
done
# Julia environments (tool-configs/*/julia/): Manifest.toml, resolved by the
# tool's own pinned Julia image against the registry snapshot in setup.jl.
# All of tool-configs/ is mounted, at the same relative layout: a project may
# take a package from another tool's directory (exapf_gpu: GridBenchExaPF).
for d in tool-configs/*/julia/; do
    julia_image=$(sed -n 's/^FROM \(julia:[^ ]*\) AS julia$/\1/p' "$(dirname "$d")/Dockerfile")
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e JULIA_DEPOT_PATH=/tmp/depot \
        -v "$PWD/tool-configs":/tool-configs "$julia_image" julia "/$d/setup.jl" lock "/$d" && echo "locked $d"
done
