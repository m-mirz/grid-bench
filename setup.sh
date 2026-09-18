#!/usr/bin/env bash
# Native development setup: datasets, a uv-managed Python 3.13 environment
# with every tool, and the prepared tool inputs.
# Containers (docker/build.sh) are the path for publishable numbers.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
git submodule update --init --depth 1
uv sync --extra all --frozen   # exactly uv.lock; re-lock with `uv lock`
uv run python -m cases.prep
uv run pytest tests -q
