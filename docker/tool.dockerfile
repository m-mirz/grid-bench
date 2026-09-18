# One image per tool (and one for the harness), from tool-configs/<TOOL>/.
# Build: docker build -f docker/tool.dockerfile --build-arg TOOL=pandapower -t grid-bench/pandapower .
FROM grid-bench/base:latest
ARG TOOL
WORKDIR /env
COPY tool-configs/${TOOL}/pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --python 3.13
ENV PATH="/env/.venv/bin:${PATH}" \
    GRID_BENCH_IMAGE=grid-bench/${TOOL}
WORKDIR /bench
