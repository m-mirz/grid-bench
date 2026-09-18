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
