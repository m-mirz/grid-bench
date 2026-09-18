# Shared base for every grid-bench image: Debian + uv + a uv-managed CPython.
# No source is baked in: the repository is mounted read-only at /bench at run
# time, so editing an adapter or the oracle never requires a rebuild.
FROM debian:bookworm-slim

# uv, pinned; bump deliberately.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PYTHONPATH=/bench \
    PYTHONDONTWRITEBYTECODE=1 \
    GRID_BENCH_RESULTS=/output \
    HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib
RUN uv python install 3.13

WORKDIR /bench
CMD ["bash"]
