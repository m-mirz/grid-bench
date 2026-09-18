# One image per tool (and one for the harness), from tool-configs/<TOOL>/.
# Build: docker build -f docker/tool.dockerfile --build-arg TOOL=pandapower -t grid-bench/pandapower .
FROM grid-bench/base:latest
ARG TOOL
WORKDIR /env
# Installs exactly the committed lockfile: every package version and file
# hash is pinned. Re-lock deliberately with docker/lock.sh.
COPY tool-configs/${TOOL}/pyproject.toml tool-configs/${TOOL}/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --python 3.13.14
ENV PATH="/env/.venv/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/${TOOL}
WORKDIR /bench
